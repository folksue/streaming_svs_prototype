from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
import sys

import soundfile as sf
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import SVSSequenceDataset
from infer import EncodecDecoder, build_model
from sample_checkpoint_compare import (
    _load_json,
    _save_json,
    build_manifest_utt_map,
    checkpoint_label,
    choose_indices,
    ensure_dir,
    format_link_target,
    load_manifest,
    masked_code_accuracy,
    save_codec_recon,
    save_gt_audio,
    write_html,
    write_markdown,
)
from utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-forced single-step reconstruction comparison.")
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)
    p.add_argument("--indices", type=int, nargs="+", default=None)
    p.add_argument("--indices-file", type=str, default=None)
    p.add_argument("--random-sample", action="store_true")
    p.add_argument("--sample-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--title", type=str, default="Teacher Forced Reconstruction Comparison")
    p.add_argument(
        "--link-style",
        choices=["path", "fileuri", "relative"],
        default="relative",
    )
    p.add_argument("--no-infer-cache", action="store_true")
    return p.parse_args()


@torch.no_grad()
def run_checkpoint_teacher_forced(
    checkpoint_path: Path,
    cache_path: Path,
    indices: Iterable[int],
    out_root: Path,
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
            metrics_json = sample_dir / f"{ckpt_tag}_tf_metrics.json"
            cached = _load_json(metrics_json)
            if cached is None:
                missing_indices.append(idx)
                continue
            cache_ok = (
                str(cached.get("utt_id", "")) == str(sample.utt_id)
                and str(cached.get("checkpoint_path", "")) == str(checkpoint_path.resolve())
                and float(cached.get("checkpoint_mtime", -1.0)) == float(ckpt_mtime)
                and Path(str(cached.get("wav_path", ""))).exists()
            )
            if cache_ok:
                results[idx] = {
                    "utt_id": str(cached["utt_id"]),
                    "wav_path": str(cached["wav_path"]),
                    "token_acc": float(cached["token_acc"]),
                }
                print(f"[cache-hit] idx={idx} ckpt={ckpt_tag} mode=tf")
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
        out_wav = sample_dir / f"{ckpt_tag}_teacher_forced.wav"
        metrics_json = sample_dir / f"{ckpt_tag}_tf_metrics.json"

        note_id = sample.note_id.unsqueeze(0).to(device)
        phoneme = sample.phoneme_id.unsqueeze(0).to(device)
        slur = sample.slur.unsqueeze(0).to(device)
        phone_progress = sample.phone_progress.unsqueeze(0).to(device)
        singer_id = sample.singer_id.unsqueeze(0).to(device)
        codes_gt = sample.codes.unsqueeze(0).to(device)
        mask = torch.ones(1, codes_gt.size(1), device=device)

        print(f"[infer] idx={idx} ckpt={ckpt_tag} mode=tf")
        logits = model.forward_train(
            codes=codes_gt,
            note_id=note_id,
            phoneme_id=phoneme,
            slur=slur,
            phone_progress=phone_progress,
            singer_id=singer_id,
            mask=mask,
        )
        codes_pred = logits.argmax(dim=-1)
        acc = float(masked_code_accuracy(codes_pred, codes_gt, mask).item())

        full_codes = codes_pred.squeeze(0).reshape(-1, codes_pred.size(-1))
        wav = decoder.decode_from_codes(full_codes)
        sf.write(str(out_wav), wav.numpy(), sample_rate, format="WAV")

        result_item = {
            "utt_id": sample.utt_id,
            "wav_path": str(out_wav.resolve()),
            "token_acc": acc,
        }
        results[idx] = result_item
        _save_json(
            metrics_json,
            {
                **result_item,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_mtime": ckpt_mtime,
            },
        )

    return results


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache).resolve()
    manifest_path = Path(args.manifest).resolve()
    checkpoint_paths = [Path(p).resolve() for p in args.checkpoints]
    out_root = Path(args.out_dir).resolve()
    ensure_dir(out_root)

    dataset = SVSSequenceDataset(str(cache_path))
    manifest = load_manifest(manifest_path)
    manifest_by_utt = build_manifest_utt_map(manifest)
    indices = choose_indices(args, len(dataset))

    fallback_decoder = EncodecDecoder("facebook/encodec_24khz", get_device())
    sample_rows: list[dict] = []
    for idx in indices:
        sample = dataset[idx]
        sample_dir = out_root / f"sample_{idx:04d}"
        ensure_dir(sample_dir)
        gt_path = sample_dir / "gt.wav"
        codec_path = sample_dir / "codec_recon_k8.wav"
        if not gt_path.exists():
            manifest_item = manifest_by_utt.get(sample.utt_id)
            audio_path = manifest_item.get("audio_path") if manifest_item else None
            if isinstance(audio_path, str) and audio_path:
                try:
                    save_gt_audio(manifest_path, audio_path, gt_path, sample_rate=int(dataset.meta["sample_rate"]))
                except Exception as exc:
                    print(f"[warn] failed to load GT audio for {sample.utt_id}: {exc}; falling back to codec reconstruction")
                    save_codec_recon(
                        decoder=fallback_decoder,
                        codes=sample.codes,
                        out_path=gt_path,
                        sample_rate=int(dataset.meta["sample_rate"]),
                    )
            else:
                save_codec_recon(
                    decoder=fallback_decoder,
                    codes=sample.codes,
                    out_path=gt_path,
                    sample_rate=int(dataset.meta["sample_rate"]),
                )
        if not codec_path.exists():
            save_codec_recon(
                decoder=fallback_decoder,
                codes=sample.codes,
                out_path=codec_path,
                sample_rate=int(dataset.meta["sample_rate"]),
            )
        sample_rows.append(
            {
                "idx": idx,
                "utt_id": sample.utt_id,
                "gt_path": format_link_target(gt_path, args.link_style, out_root),
                "codec_path": format_link_target(codec_path, args.link_style, out_root),
                "checkpoint_results": {},
            }
        )

    checkpoint_columns = []
    for ckpt_path in checkpoint_paths:
        label = f"{checkpoint_label(ckpt_path)}_tf"
        checkpoint_columns.append({"label": label})
        results = run_checkpoint_teacher_forced(
            checkpoint_path=ckpt_path,
            cache_path=cache_path,
            indices=indices,
            out_root=out_root,
            use_infer_cache=not args.no_infer_cache,
        )
        for row in sample_rows:
            res = dict(results[row["idx"]])
            res["wav_path"] = format_link_target(Path(res["wav_path"]), args.link_style, out_root)
            row["checkpoint_results"][label] = res

    md_path = out_root / "comparison.md"
    html_path = out_root / "comparison.html"
    write_markdown(md_path, args.title, sample_rows, checkpoint_columns)
    write_html(html_path, args.title, sample_rows, checkpoint_columns)
    print(f"saved {md_path}")
    print(f"saved {html_path}")


if __name__ == "__main__":
    main()
