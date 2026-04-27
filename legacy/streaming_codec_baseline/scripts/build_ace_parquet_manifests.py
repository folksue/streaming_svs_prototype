#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lightweight ACE manifests that reference audio bytes inside parquet shards.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-type", choices=["ace-opencpop", "ace-kising"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def make_audio_uri(parquet_path: Path, row_idx: int) -> str:
    return f"parquet://{quote(str(parquet_path.resolve()), safe='/:._-')}::{row_idx}"


def infer_singer_from_segment_id(segment_id: str, dataset_type: str) -> str:
    text = str(segment_id).strip()
    if "#" in text:
        return text.split("#", 1)[0].strip() or f"{dataset_type}_default"
    m = re.match(r"^(acesinger_\\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    if text.isdigit():
        return f"{dataset_type}_default"
    if "_" in text:
        prefix = text.split("_", 1)[0].strip()
        if prefix and prefix.lower() not in {"seg", "utt", "sample"}:
            return prefix
    return f"{dataset_type}_default"


def convert_row(row: dict, parquet_path: Path, row_idx: int, dataset_type: str) -> dict:
    if dataset_type == "ace-opencpop":
        phoneme_symbols = list(row["phn"])
        phoneme_intervals = [[float(s), float(e)] for s, e in zip(row["phn_start_time"], row["phn_end_time"])]
    else:
        phoneme_symbols = list(row["phns"])
        phoneme_intervals = [[float(s), float(e)] for s, e in zip(row["phn_start_times"], row["phn_end_times"])]

    note_pitch_midi = [float(x) for x in row["note_midi"]]
    note_intervals = [[float(s), float(e)] for s, e in zip(row["note_start_times"], row["note_end_times"])]
    note_slur = [0 for _ in note_intervals]
    duration = max(note_intervals[-1][1] if note_intervals else 0.0, phoneme_intervals[-1][1] if phoneme_intervals else 0.0)
    return {
        "utt_id": str(row["segment_id"]),
        "audio_path": make_audio_uri(parquet_path, row_idx),
        "singer": infer_singer_from_segment_id(str(row["segment_id"]), dataset_type=dataset_type),
        "phoneme_symbols": phoneme_symbols,
        "phoneme_intervals": phoneme_intervals,
        "note_pitch_midi": note_pitch_midi,
        "note_intervals": note_intervals,
        "note_slur": note_slur,
        "duration": duration,
    }


def build_split_manifest(dataset_root: Path, split: str, dataset_type: str) -> list[dict]:
    data_dir = dataset_root / "data"
    shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    entries: list[dict] = []
    for shard in shards:
        table = pq.read_table(
            shard,
            columns=[
                "segment_id",
                "note_midi",
                "note_start_times",
                "note_end_times",
                "phn" if dataset_type == "ace-opencpop" else "phns",
                "phn_start_time" if dataset_type == "ace-opencpop" else "phn_start_times",
                "phn_end_time" if dataset_type == "ace-opencpop" else "phn_end_times",
            ],
        )
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            entries.append(convert_row(row, shard, row_idx, dataset_type))
    return entries


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    for split in ["train", "validation", "test"]:
        entries = build_split_manifest(args.dataset_root, split, args.dataset_type)
        out_path = args.out_dir / f"{split}.json"
        save_json(out_path, entries)
        print(f"{split}: {len(entries)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
