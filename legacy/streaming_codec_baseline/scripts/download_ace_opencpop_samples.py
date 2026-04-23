#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from typing import Any, Dict, List
from urllib.parse import urlencode


HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/first-rows"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ACE-OpenCpop sample rows and wavs.")
    p.add_argument("--dataset", type=str, default="espnet/ace-opencpop-segments")
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_rows", type=int, default=20)
    p.add_argument(
        "--audio_dir",
        type=str,
        default="data/samples/ace_opencpop/audio",
        help="Directory to save downloaded sample wavs.",
    )
    p.add_argument(
        "--manifest_out",
        type=str,
        default="data/samples/ace_opencpop/manifest.json",
        help="Output JSON manifest compatible with adapter 'ace-opencpop-segments'.",
    )
    p.add_argument(
        "--skip_audio",
        action="store_true",
        help="Only export metadata manifest, do not download wav files.",
    )
    p.add_argument(
        "--allow_missing_audio",
        action="store_true",
        help="Continue even if some audio downloads fail (records still exported).",
    )
    return p.parse_args()


def fetch_rows(dataset: str, config: str, split: str) -> List[Dict[str, Any]]:
    query = urlencode({"dataset": dataset, "config": config, "split": split})
    url = f"{HF_ROWS_ENDPOINT}?{query}"
    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("rows", [])
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            out.append(item["row"])
        elif isinstance(item, dict):
            out.append(item)
    return out


def download_file(url: str, dst_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "streaming-svs-prototype/1.0"})
    with urllib.request.urlopen(req) as resp, open(dst_path, "wb") as f:
        f.write(resp.read())


def main() -> int:
    args = parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)

    rows = fetch_rows(dataset=args.dataset, config=args.config, split=args.split)
    rows = rows[: max(args.max_rows, 0)]
    if not rows:
        raise RuntimeError("No rows returned from HF datasets-server.")

    manifest: List[Dict[str, Any]] = []
    missing_audio = 0
    for idx, row in enumerate(rows):
        segment_id = str(row.get("segment_id", f"{args.split}_{idx:06d}"))
        audio_url = None
        audio_meta = row.get("audio")
        if isinstance(audio_meta, list) and audio_meta and isinstance(audio_meta[0], dict):
            audio_url = audio_meta[0].get("src")
        elif isinstance(audio_meta, dict):
            audio_url = audio_meta.get("src") or audio_meta.get("url")
        elif isinstance(audio_meta, str):
            audio_url = audio_meta

        if not audio_url:
            raise RuntimeError(f"Missing audio src for segment {segment_id}.")

        wav_rel = f"{segment_id}.wav"
        wav_path = os.path.join(args.audio_dir, wav_rel)
        audio_ok = True
        if not args.skip_audio:
            if not os.path.exists(wav_path):
                try:
                    download_file(audio_url, wav_path)
                except Exception as exc:
                    audio_ok = False
                    missing_audio += 1
                    if not args.allow_missing_audio:
                        raise RuntimeError(f"Failed to download audio for {segment_id}: {exc}") from exc
                    print(f"[warn] audio download failed for {segment_id}: {exc}")
            else:
                audio_ok = True
        else:
            audio_ok = os.path.exists(wav_path)
            if not audio_ok:
                missing_audio += 1

        manifest.append(
            {
                "segment_id": segment_id,
                "audio_path": wav_rel,
                "audio_downloaded": bool(audio_ok),
                "audio_url": audio_url,
                "transcription": row.get("transcription", ""),
                "singer": row.get("singer", 0),
                "label": row.get("label", ""),
                "tempo": row.get("tempo", 0),
                "note_midi": row.get("note_midi", []),
                "note_phns": row.get("note_phns", []),
                "note_lyrics": row.get("note_lyrics", []),
                "note_start_times": row.get("note_start_times", []),
                "note_end_times": row.get("note_end_times", []),
                "phn": row.get("phn", []),
                "phn_start_time": row.get("phn_start_time", []),
                "phn_end_time": row.get("phn_end_time", []),
            }
        )

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"saved manifest: {args.manifest_out} ({len(manifest)} entries)")
    print(f"saved audio dir: {args.audio_dir}")
    print(f"missing audio: {missing_audio}")
    print("adapter: ace-opencpop-segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
