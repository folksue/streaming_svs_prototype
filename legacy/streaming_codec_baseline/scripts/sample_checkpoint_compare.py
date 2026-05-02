from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Iterable
import sys
import random
from urllib.parse import unquote

import soundfile as sf
import torch
import torchaudio

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import SVSSequenceDataset
from infer import EncodecDecoder, build_model
from utils import get_device, masked_code_accuracy


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--indices-file", type=str, default=None)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Checkpoint Comparison")
    parser.add_argument("--dump-logits", action="store_true")
    parser.add_argument("--logits-topk", type=int, default=10)
    parser.add_argument("--no-infer-cache", action="store_true")
    parser.add_argument(
        "--link-style",
        choices=["path", "fileuri", "relative"],
        default="relative",
        help="How links are written inside the generated markdown.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    return data


def build_manifest_utt_map(manifest: list[dict]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in manifest:
        utt_id = item.get("utt_id")
        if not isinstance(utt_id, str) or not utt_id:
            continue
        if utt_id not in mapping:
            mapping[utt_id] = item
    return mapping


def checkpoint_label(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if stem == "best":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        return f"best_ep{epoch:03d}"
    if stem == "last":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        return f"last_ep{epoch:03d}"
    return stem


def resolve_audio_path(manifest_path: Path, audio_path: str) -> Path:
    candidate = Path(audio_path)
    if "://" in audio_path:
        return candidate
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_link_target(path: Path, style: str, base_dir: Path) -> str:
    resolved = path.resolve()
    if style == "fileuri":
        return resolved.as_uri()
    if style == "relative":
        return Path(os.path.relpath(resolved, base_dir.resolve())).as_posix()
    return str(resolved)


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def choose_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    if args.indices_file:
        idx_path = Path(args.indices_file).resolve()
        if idx_path.exists():
            payload = _load_json(idx_path)
            if not payload or not isinstance(payload.get("indices"), list):
                raise ValueError(f"Invalid indices file: {idx_path}")
            indices = [int(x) for x in payload["indices"]]
            if any(x < 0 or x >= dataset_len for x in indices):
                raise ValueError(f"indices file out-of-range for dataset size={dataset_len}: {idx_path}")
            return indices
    if args.indices:
        indices = list(args.indices)
        if args.indices_file:
            _save_json(Path(args.indices_file).resolve(), {"indices": [int(x) for x in indices]})
        return indices
    if not args.random_sample:
        raise ValueError("Provide --indices or set --random-sample.")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.sample_count > dataset_len:
        raise ValueError(f"--sample-count ({args.sample_count}) exceeds dataset size ({dataset_len}).")
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(dataset_len), args.sample_count))
    if args.indices_file:
        _save_json(Path(args.indices_file).resolve(), {"indices": [int(x) for x in indices], "seed": int(args.seed)})
    return indices


def save_codec_recon(
    decoder: EncodecDecoder,
    codes: torch.Tensor,
    out_path: Path,
    sample_rate: int,
) -> None:
    full_codes = codes.reshape(-1, codes.size(-1))
    wav = decoder.decode_from_codes(full_codes)
    sf.write(str(out_path), wav.squeeze().cpu().numpy(), sample_rate, format="WAV")


def load_manifest_audio(manifest_path: Path, audio_path: str, sample_rate: int) -> torch.Tensor:
    if audio_path.startswith("parquet://"):
        payload = audio_path[len("parquet://") :]
        parquet_path, row_idx_str = payload.rsplit("::", 1)
        parquet_path = unquote(parquet_path)
        row_idx = int(row_idx_str)
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path, columns=["audio"])
        rows = table.column("audio").to_pylist()
        item = rows[row_idx]
        if not isinstance(item, dict) or not isinstance(item.get("bytes"), (bytes, bytearray)):
            raise RuntimeError(f"Invalid parquet audio payload at {parquet_path} row {row_idx}")
        wav_np, sr = sf.read(io.BytesIO(bytes(item["bytes"])), always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1).float()
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav

    resolved = resolve_audio_path(manifest_path, audio_path)
    try:
        wav, sr = torchaudio.load(str(resolved))
    except Exception:
        wav_np, sr = sf.read(str(resolved), always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1).float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def save_gt_audio(
    manifest_path: Path,
    audio_path: str,
    out_path: Path,
    sample_rate: int,
) -> None:
    wav = load_manifest_audio(manifest_path, audio_path, sample_rate=sample_rate)
    sf.write(str(out_path), wav.squeeze().cpu().numpy(), sample_rate, format="WAV")


@torch.no_grad()
def generate_stream_with_debug(
    model,
    note_id: torch.Tensor,
    phoneme_id: torch.Tensor,
    slur: torch.Tensor,
    phone_progress: torch.Tensor,
    singer_id: torch.Tensor,
    codes_gt: torch.Tensor | None,
    temperature: float,
    dump_logits: bool,
    logits_topk: int,
) -> tuple[torch.Tensor, dict | None]:
    b, t = note_id.shape
    cond = model.cond_enc(note_id, phoneme_id, slur, phone_progress, singer_id=singer_id)
    kv_cache = [(None, None) for _ in range(len(model.blocks))]
    pos = 0
    prev_frame_codes = torch.full(
        (b, model.num_codebooks),
        fill_value=model.audio_bos_id,
        dtype=torch.long,
        device=note_id.device,
    )

    if model.prefix_pad_steps > 0:
        pad_control = model.prefix_control_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
        pad_frame = model.prefix_frame_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
        for _ in range(model.prefix_pad_steps):
            control_in = pad_control + model.token_type_emb.weight[0].view(1, 1, -1) + model.within_step_pos_emb[0].to(
                cond.dtype
            ).view(1, 1, -1)
            _, kv_cache = model._forward_one_token_incremental(control_in, pos, kv_cache)
            pos += 1
            for frame_slot in range(model.audio_slots_per_step):
                frame_in = pad_frame + model.token_type_emb.weight[1].view(1, 1, -1) + model.within_step_pos_emb[
                    1 + frame_slot
                ].to(cond.dtype).view(1, 1, -1)
                _, kv_cache = model._forward_one_token_incremental(frame_in, pos, kv_cache)
                pos += 1

    pred_steps = []
    topk_idx_steps = []
    topk_val_steps = []
    gt_token_steps = []
    gt_logit_steps = []
    gt_rank_steps = []
    topk = max(1, int(logits_topk))

    for step_idx in range(t):
        step_tokens = []
        step_topk_idx = []
        step_topk_val = []
        step_gt_tok = []
        step_gt_logit = []
        step_gt_rank = []

        control_tok = model._encode_control(cond[:, step_idx : step_idx + 1, :])
        control_in = control_tok + model.token_type_emb.weight[0].view(1, 1, -1) + model.within_step_pos_emb[0].to(
            cond.dtype
        ).view(1, 1, -1)
        _, kv_cache = model._forward_one_token_incremental(control_in, pos, kv_cache)
        pos += 1

        for frame_idx in range(model.audio_slots_per_step):
            frame_codes_in = prev_frame_codes.view(b, 1, 1, model.num_codebooks)
            frame_tok = model.audio_in(model._embed_frame_tokens(frame_codes_in)).view(b, 1, -1)
            frame_in = frame_tok + model.token_type_emb.weight[1].view(1, 1, -1) + model.within_step_pos_emb[
                1 + frame_idx
            ].to(cond.dtype).view(1, 1, -1)
            frame_hidden, kv_cache = model._forward_one_token_incremental(frame_in, pos, kv_cache)
            pos += 1

            frame_hidden_flat = frame_hidden[:, 0, :]
            logits_per_codebook = [head(frame_hidden_flat) for head in model.audio_heads]
            next_logits = torch.stack(logits_per_codebook, dim=1)
            next_tokens = model._sample_next(next_logits, temperature=temperature).long()
            prev_frame_codes = next_tokens
            step_tokens.append(next_tokens)

            if dump_logits:
                logits_0 = next_logits[0]
                tk_val, tk_idx = torch.topk(logits_0, k=min(topk, logits_0.size(-1)), dim=-1)
                step_topk_idx.append(tk_idx.cpu())
                step_topk_val.append(tk_val.cpu())
                if codes_gt is not None:
                    gt_tok = codes_gt[0, step_idx, frame_idx].long()
                    gathered = logits_0.gather(dim=-1, index=gt_tok.unsqueeze(-1)).squeeze(-1)
                    gt_tok_rank = (logits_0 > gathered.unsqueeze(-1)).sum(dim=-1)
                    step_gt_tok.append(gt_tok.cpu())
                    step_gt_logit.append(gathered.cpu())
                    step_gt_rank.append(gt_tok_rank.cpu())

        pred_steps.append(torch.stack(step_tokens, dim=1))
        if dump_logits:
            topk_idx_steps.append(torch.stack(step_topk_idx, dim=0))
            topk_val_steps.append(torch.stack(step_topk_val, dim=0))
            if codes_gt is not None:
                gt_token_steps.append(torch.stack(step_gt_tok, dim=0))
                gt_logit_steps.append(torch.stack(step_gt_logit, dim=0))
                gt_rank_steps.append(torch.stack(step_gt_rank, dim=0))

    codes_pred = torch.stack(pred_steps, dim=1)
    if not dump_logits:
        return codes_pred, None

    debug = {
        "pred_tokens": codes_pred[0].cpu().short().numpy(),
        "topk_idx": torch.stack(topk_idx_steps, dim=0).short().numpy(),
        "topk_val": torch.stack(topk_val_steps, dim=0).float().numpy(),
    }
    if codes_gt is not None and gt_token_steps:
        debug["gt_tokens"] = torch.stack(gt_token_steps, dim=0).short().numpy()
        debug["gt_logit"] = torch.stack(gt_logit_steps, dim=0).float().numpy()
        debug["gt_rank"] = torch.stack(gt_rank_steps, dim=0).short().numpy()
    return codes_pred, debug


@torch.no_grad()
def run_checkpoint_inference(
    checkpoint_path: Path,
    cache_path: Path,
    indices: Iterable[int],
    out_root: Path,
    dump_logits: bool = False,
    logits_topk: int = 10,
    use_infer_cache: bool = True,
) -> dict[int, dict]:
    device = get_device()
    dataset = SVSSequenceDataset(str(cache_path))
    ckpt_mtime = checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0.0
    ckpt_tag = checkpoint_label(checkpoint_path)
    results: dict[int, dict] = {}
    missing_indices: list[int] = []

    if use_infer_cache:
        for idx in indices:
            sample = dataset[idx]
            sample_dir = out_root / f"sample_{idx:04d}"
            metrics_json = sample_dir / f"{ckpt_tag}_metrics.json"
            cached = _load_json(metrics_json)
            if cached is None:
                missing_indices.append(idx)
                continue
            cache_ok = (
                str(cached.get("utt_id", "")) == str(sample.utt_id)
                and str(cached.get("checkpoint_path", "")) == str(checkpoint_path.resolve())
                and float(cached.get("checkpoint_mtime", -1.0)) == float(ckpt_mtime)
                and bool(cached.get("dump_logits", False)) == bool(dump_logits)
                and int(cached.get("logits_topk", -1)) == int(logits_topk)
                and Path(str(cached.get("wav_path", ""))).exists()
            )
            if dump_logits:
                cache_ok = cache_ok and Path(str(cached.get("logits_path", ""))).exists()
            if cache_ok:
                results[idx] = {
                    "utt_id": str(cached["utt_id"]),
                    "wav_path": str(cached["wav_path"]),
                    "token_acc": float(cached["token_acc"]),
                    "logits_path": str(cached.get("logits_path", "")),
                }
                print(f"[cache-hit] idx={idx} ckpt={ckpt_tag}")
            else:
                missing_indices.append(idx)
    else:
        missing_indices = list(indices)

    if not missing_indices:
        return results

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt, device)
    decoder = EncodecDecoder(
        model_name=ckpt["config"]["audio"]["encodec_model_name"],
        device=device,
    )
    sample_rate = int(ckpt["config"]["audio"]["sample_rate"])

    for idx in missing_indices:
        sample = dataset[idx]
        sample_dir = out_root / f"sample_{idx:04d}"
        ensure_dir(sample_dir)
        out_wav = sample_dir / f"{ckpt_tag}.wav"
        logits_npz = sample_dir / f"{ckpt_tag}_logits_debug.npz"
        metrics_json = sample_dir / f"{ckpt_tag}_metrics.json"

        note_id = sample.note_id.unsqueeze(0).to(device)
        phoneme = sample.phoneme_id.unsqueeze(0).to(device)
        slur = sample.slur.unsqueeze(0).to(device)
        phone_progress = sample.phone_progress.unsqueeze(0).to(device)
        singer_id = sample.singer_id.unsqueeze(0).to(device)
        codes_gt = sample.codes.unsqueeze(0).to(device)
        mask = torch.ones(1, codes_gt.size(1), device=device)

        print(f"[infer] idx={idx} ckpt={ckpt_tag}")
        codes_pred, debug = generate_stream_with_debug(
            model=model,
            note_id=note_id,
            phoneme_id=phoneme,
            slur=slur,
            phone_progress=phone_progress,
            singer_id=singer_id,
            codes_gt=codes_gt,
            temperature=0.0,
            dump_logits=dump_logits,
            logits_topk=logits_topk,
        )
        acc = float(masked_code_accuracy(codes_pred, codes_gt, mask).item())

        full_codes = codes_pred.squeeze(0).reshape(-1, codes_pred.size(-1))
        wav = decoder.decode_from_codes(full_codes)
        sf.write(out_wav, wav.numpy(), sample_rate)

        logits_path = ""
        if dump_logits and debug is not None:
            import numpy as np

            np.savez_compressed(
                str(logits_npz),
                utt_id=sample.utt_id,
                checkpoint_label=ckpt_tag,
                **debug,
            )
            logits_path = str(logits_npz.resolve())

        result_item = {
            "utt_id": sample.utt_id,
            "wav_path": str(out_wav.resolve()),
            "token_acc": acc,
            "logits_path": logits_path,
        }
        results[idx] = result_item
        _save_json(
            metrics_json,
            {
                **result_item,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_mtime": ckpt_mtime,
                "dump_logits": bool(dump_logits),
                "logits_topk": int(logits_topk),
            },
        )

    return results


def write_markdown(
    out_path: Path,
    title: str,
    sample_rows: list[dict],
    checkpoint_columns: list[dict],
) -> None:
    lines = [f"# {title}", ""]
    header = ["idx", "utt_id", "GT", "codec_recon"]
    header.extend(col["label"] for col in checkpoint_columns)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in sample_rows:
        cells = [
            str(row["idx"]),
            f"`{row['utt_id']}`",
            f"[gt]({row['gt_path']})",
            f"[codec]({row['codec_path']})",
        ]
        for col in checkpoint_columns:
            result = row["checkpoint_results"][col["label"]]
            logits_part = f"<br>[logits]({result['logits_path']})" if result.get("logits_path") else ""
            cells.append(f"[wav]({result['wav_path']})<br>`acc={result['token_acc']:.6f}`{logits_part}")
        lines.append("| " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(
    out_path: Path,
    title: str,
    sample_rows: list[dict],
    checkpoint_columns: list[dict],
) -> None:
    cols = "".join(f"<th>{html_escape(col['label'])}</th>" for col in checkpoint_columns)
    rows_html: list[str] = []
    for row in sample_rows:
        cells = [
            f"<td class='meta'>{row['idx']}</td>",
            f"<td class='meta'><code>{html_escape(row['utt_id'])}</code></td>",
            (
                "<td><div class='cell'>"
                f"<a href='{html_escape(row['gt_path'])}'>gt.wav</a>"
                f"<audio controls preload='none' src='{html_escape(row['gt_path'])}'></audio>"
                "</div></td>"
            ),
            (
                "<td><div class='cell'>"
                f"<a href='{html_escape(row['codec_path'])}'>codec_q2.wav</a>"
                f"<audio controls preload='none' src='{html_escape(row['codec_path'])}'></audio>"
                "</div></td>"
            ),
        ]
        for col in checkpoint_columns:
            result = row["checkpoint_results"][col["label"]]
            cells.append(
                "<td><div class='cell'>"
                f"<a href='{html_escape(result['wav_path'])}'>{html_escape(col['label'])}.wav</a>"
                f"<audio controls preload='none' src='{html_escape(result['wav_path'])}'></audio>"
                f"<div class='score'>acc={result['token_acc']:.6f}</div>"
                "</div></td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --panel: #fffdf7;
      --grid: #d7d0bf;
      --text: #1f1d18;
      --muted: #6e6758;
      --accent: #8e3b2e;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: linear-gradient(180deg, #f3efe6 0%, #ece5d7 100%);
      color: var(--text);
    }}
    .wrap {{ padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; }}
    p.note {{ margin: 0 0 20px; color: var(--muted); }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--grid);
      background: var(--panel);
      box-shadow: 0 12px 30px rgba(40, 30, 10, 0.08);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1800px; }}
    th, td {{
      border: 1px solid var(--grid);
      padding: 10px;
      vertical-align: top;
      background: rgba(255, 253, 247, 0.92);
    }}
    th {{
      position: sticky; top: 0; z-index: 1; background: #efe6d2; font-size: 13px; text-align: left;
    }}
    td.meta {{ white-space: nowrap; font-size: 13px; }}
    .cell {{ display: flex; flex-direction: column; gap: 8px; min-width: 170px; }}
    .score {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--accent); }}
    audio {{ width: 100%; min-width: 160px; height: 32px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html_escape(title)}</h1>
    <p class="note">浏览器直接打开即可试听。每个单元格都保留了 wav 文件链接和音频控件。</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>idx</th>
            <th>utt_id</th>
            <th>GT</th>
            <th>codec_recon</th>
            {cols}
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_root = Path(args.out_dir).resolve()
    ensure_dir(out_root)

    dataset = SVSSequenceDataset(str(cache_path))
    manifest = load_manifest(manifest_path)
    manifest_utt_map = build_manifest_utt_map(manifest)
    device = get_device()
    codec_decoder = None
    sample_rate = None

    def ensure_codec_decoder() -> tuple[EncodecDecoder, int]:
        nonlocal codec_decoder, sample_rate
        if codec_decoder is None or sample_rate is None:
            ref_ckpt = torch.load(Path(args.checkpoints[0]).resolve(), map_location=device, weights_only=False)
            codec_decoder = EncodecDecoder(
                model_name=ref_ckpt["config"]["audio"]["encodec_model_name"],
                device=device,
            )
            sample_rate = int(ref_ckpt["config"]["audio"]["sample_rate"])
        return codec_decoder, sample_rate

    indices = choose_indices(args, len(dataset))

    sample_rows: list[dict] = []
    for idx in indices:
        sample = dataset[idx]
        manifest_item = manifest_utt_map.get(sample.utt_id)
        if manifest_item is None:
            manifest_item = manifest[idx]
        sample_dir = out_root / f"sample_{idx:04d}"
        ensure_dir(sample_dir)
        gt_path = sample_dir / "gt.wav"
        if not gt_path.exists():
            decoder, sample_rate = ensure_codec_decoder()
            try:
                save_gt_audio(manifest_path, str(manifest_item["audio_path"]), gt_path, sample_rate)
            except Exception as exc:
                print(f"[warn] idx={idx} utt={sample.utt_id} failed to load GT audio: {exc}")
                save_codec_recon(decoder, sample.codes, gt_path, sample_rate)
        codec_path = sample_dir / f"codec_recon_k{sample.codes.size(-1)}.wav"
        if not codec_path.exists():
            decoder, sample_rate = ensure_codec_decoder()
            save_codec_recon(decoder, sample.codes, codec_path, sample_rate)
        sample_rows.append(
            {
                "idx": idx,
                "utt_id": sample.utt_id,
                "gt_path": format_link_target(gt_path, args.link_style, out_root),
                "codec_path": format_link_target(codec_path, args.link_style, out_root),
                "checkpoint_results": {},
            }
        )

    checkpoint_columns: list[dict] = []
    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint).resolve()
        label = checkpoint_label(checkpoint_path)
        checkpoint_columns.append({"label": label, "path": str(checkpoint_path)})
        results = run_checkpoint_inference(
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
            indices=indices,
            out_root=out_root,
            dump_logits=args.dump_logits,
            logits_topk=args.logits_topk,
            use_infer_cache=not args.no_infer_cache,
        )
        for row in sample_rows:
            result = dict(results[row["idx"]])
            result["wav_path"] = format_link_target(Path(result["wav_path"]), args.link_style, out_root)
            if result.get("logits_path"):
                result["logits_path"] = format_link_target(Path(result["logits_path"]), args.link_style, out_root)
            row["checkpoint_results"][label] = result

    md_path = out_root / "comparison.md"
    write_markdown(
        out_path=md_path,
        title=args.title,
        sample_rows=sample_rows,
        checkpoint_columns=checkpoint_columns,
    )
    html_path = out_root / "comparison.html"
    write_html(
        out_path=html_path,
        title=args.title,
        sample_rows=sample_rows,
        checkpoint_columns=checkpoint_columns,
    )
    print(md_path.resolve())


if __name__ == "__main__":
    main()
