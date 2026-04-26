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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Checkpoint Comparison")
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
    if args.indices:
        return list(args.indices)
    if not args.random_sample:
        raise ValueError("Provide --indices or set --random-sample.")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.sample_count > dataset_len:
        raise ValueError(f"--sample-count ({args.sample_count}) exceeds dataset size ({dataset_len}).")
    rng = random.Random(args.seed)
    return sorted(rng.sample(range(dataset_len), args.sample_count))


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
def run_checkpoint_inference(
    checkpoint_path: Path,
    cache_path: Path,
    indices: Iterable[int],
    out_root: Path,
) -> dict[int, dict]:
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt, device)
    decoder = EncodecDecoder(
        model_name=ckpt["config"]["audio"]["encodec_model_name"],
        device=device,
    )
    dataset = SVSSequenceDataset(str(cache_path))
    sample_rate = int(ckpt["config"]["audio"]["sample_rate"])
    results: dict[int, dict] = {}

    for idx in indices:
        sample = dataset[idx]
        note_id = sample.note_id.unsqueeze(0).to(device)
        phoneme = sample.phoneme_id.unsqueeze(0).to(device)
        slur = sample.slur.unsqueeze(0).to(device)
        phone_progress = sample.phone_progress.unsqueeze(0).to(device)
        codes_gt = sample.codes.unsqueeze(0).to(device)
        mask = torch.ones(1, codes_gt.size(1), device=device)

        pred_steps = list(
            model.generate_stream(
                note_id=note_id,
                phoneme_id=phoneme,
                slur=slur,
                phone_progress=phone_progress,
                temperature=0.0,
            )
        )
        codes_pred = torch.stack(pred_steps, dim=1)
        acc = float(masked_code_accuracy(codes_pred, codes_gt, mask).item())

        full_codes = codes_pred.squeeze(0).reshape(-1, codes_pred.size(-1))
        wav = decoder.decode_from_codes(full_codes)

        sample_dir = out_root / f"sample_{idx:04d}"
        ensure_dir(sample_dir)
        out_wav = sample_dir / f"{checkpoint_label(checkpoint_path)}.wav"
        sf.write(out_wav, wav.numpy(), sample_rate)

        results[idx] = {
            "utt_id": sample.utt_id,
            "wav_path": str(out_wav.resolve()),
            "token_acc": acc,
        }

    return results


def write_markdown(
    out_path: Path,
    title: str,
    sample_rows: list[dict],
    checkpoint_columns: list[dict],
) -> None:
    lines = [f"# {title}", ""]
    header = ["idx", "utt_id", "GT", "codec_recon_q2"]
    header.extend(col["label"] for col in checkpoint_columns)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in sample_rows:
        cells = [
            str(row["idx"]),
            f"`{row['utt_id']}`",
            f"[gt]({row['gt_path']})",
            f"[codec_q2]({row['codec_path']})",
        ]
        for col in checkpoint_columns:
            result = row["checkpoint_results"][col["label"]]
            cells.append(f"[wav]({result['wav_path']})<br>`acc={result['token_acc']:.6f}`")
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
            <th>codec_recon_q2</th>
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
    device = get_device()
    ref_ckpt = torch.load(Path(args.checkpoints[0]).resolve(), map_location=device, weights_only=False)
    codec_decoder = EncodecDecoder(
        model_name=ref_ckpt["config"]["audio"]["encodec_model_name"],
        device=device,
    )
    sample_rate = int(ref_ckpt["config"]["audio"]["sample_rate"])

    indices = choose_indices(args, len(dataset))

    sample_rows: list[dict] = []
    for idx in indices:
        sample = dataset[idx]
        manifest_item = manifest[idx]
        sample_dir = out_root / f"sample_{idx:04d}"
        ensure_dir(sample_dir)
        gt_path = sample_dir / "gt.wav"
        if not gt_path.exists():
            save_gt_audio(manifest_path, str(manifest_item["audio_path"]), gt_path, sample_rate)
        codec_path = sample_dir / "codec_recon_q2.wav"
        if not codec_path.exists():
            save_codec_recon(codec_decoder, sample.codes, codec_path, sample_rate)
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
        )
        for row in sample_rows:
            result = dict(results[row["idx"]])
            result["wav_path"] = format_link_target(Path(result["wav_path"]), args.link_style, out_root)
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
