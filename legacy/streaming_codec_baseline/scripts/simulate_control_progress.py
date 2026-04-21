#!/usr/bin/env python3
"""Quick simulator for legacy control-state progression.

This script mirrors the current preprocessing rule:

- derive step centers from frame rate and tokens_per_step
- sample phoneme/note/slur at each step center
- bucketize phoneme progress with the same implementation used by training

It is intended for debugging alignment behavior before running full preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = REPO_ROOT / "legacy" / "streaming_codec_baseline"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from metadata_adapters import load_manifest_entries, normalize_standard_entry  # type: ignore
from preprocess_encodec import (  # type: ignore
    bucketize_progress,
    find_active_interval,
    note_features,
    phoneme_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate legacy step-level control progression.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to a dataset manifest. Can be adapter-specific raw metadata.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="manifest",
        help="Manifest adapter name, for example: manifest, opencpop, m4singer.",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=None,
        help="Optional dataset audio root passed into the adapter.",
    )
    parser.add_argument(
        "--utt-id",
        type=str,
        default=None,
        help="Specific utterance id to simulate when using a manifest.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Path to a normalized single-utterance JSON spec.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=75.0,
        help="Codec frame rate in Hz. Example: 75 for EnCodec 24k.",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=1,
        help="How many codec frames belong to one model step.",
    )
    parser.add_argument(
        "--phone-progress-bins",
        type=int,
        default=8,
        help="Number of discrete phone-progress buckets.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit for printed steps.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON rows instead of a text table.",
    )
    return parser.parse_args()


def load_single_utterance(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec is not None:
        raw = json.loads(args.spec.read_text(encoding="utf-8"))
        if "audio_path" in raw:
            utt = normalize_standard_entry(raw, base_dir=str(args.spec.parent))
        else:
            utt = dict(raw)
            utt["utt_id"] = str(raw["utt_id"])
            utt["phoneme_intervals"] = [[float(a), float(b)] for a, b in raw["phoneme_intervals"]]
            utt["note_intervals"] = [[float(a), float(b)] for a, b in raw["note_intervals"]]
            utt["note_pitch_midi"] = [float(x) for x in raw["note_pitch_midi"]]
            utt["note_slur"] = [int(x) for x in raw.get("note_slur", [0 for _ in utt["note_intervals"]])]
        if "phoneme_ids" not in utt:
            utt["phoneme_ids"] = list(range(1, len(utt["phoneme_symbols"]) + 1))
        return utt

    if args.manifest is None:
        raise ValueError("Either --spec or --manifest must be provided.")

    entries = load_manifest_entries(
        str(args.manifest),
        adapter_name=args.adapter,
        audio_root=args.audio_root,
    )
    if not entries:
        raise ValueError(f"No entries found in manifest: {args.manifest}")

    if args.utt_id is not None:
        for entry in entries:
            if str(entry["utt_id"]) == args.utt_id:
                return entry
        raise ValueError(f"Could not find utt_id={args.utt_id!r} in {args.manifest}")
    return entries[0]


def estimate_duration_sec(utt: dict[str, Any]) -> float:
    duration = float(utt.get("duration", 0.0))
    if duration > 0.0:
        return duration
    max_note = max((float(x[1]) for x in utt["note_intervals"]), default=0.0)
    max_ph = max((float(x[1]) for x in utt["phoneme_intervals"]), default=0.0)
    return max(max_note, max_ph)


def simulate_rows(
    utt: dict[str, Any],
    frame_rate: float,
    tokens_per_step: int,
    phone_progress_bins: int,
) -> list[dict[str, Any]]:
    duration_sec = estimate_duration_sec(utt)
    total_frames = int(round(duration_sec * frame_rate))
    n_steps = total_frames // tokens_per_step
    step_duration_sec = tokens_per_step / frame_rate
    boundary_eps = 0.5 * step_duration_sec
    phoneme_ids = utt.get("phoneme_ids")
    if phoneme_ids is None:
        phoneme_ids = list(range(1, len(utt["phoneme_symbols"]) + 1))

    rows: list[dict[str, Any]] = []
    for step_idx in range(n_steps):
        start_frame = step_idx * tokens_per_step
        center_t = (start_frame + 0.5 * tokens_per_step) / frame_rate

        ph_idx = find_active_interval(center_t, utt["phoneme_intervals"])
        note_idx = find_active_interval(center_t, utt["note_intervals"])
        ph_id, ph_prog_bucket = phoneme_features(
            center_t,
            utt["phoneme_intervals"],
            phoneme_ids,
            phone_progress_bins=phone_progress_bins,
        )
        note_id, note_start, note_end, note_on, note_off, slur = note_features(
            center_t,
            utt["note_intervals"],
            utt["note_pitch_midi"],
            utt.get("note_slur", []),
            boundary_eps=boundary_eps,
        )

        raw_phone_progress = 0.0
        if ph_idx >= 0:
            ph_s, ph_e = utt["phoneme_intervals"][ph_idx]
            ph_dur = max(float(ph_e) - float(ph_s), 1e-4)
            raw_phone_progress = min(max((center_t - float(ph_s)) / ph_dur, 0.0), 1.0)
        raw_phone_progress_bucket = bucketize_progress(raw_phone_progress, phone_progress_bins)

        rows.append(
            {
                "step_idx": step_idx,
                "center_t": round(center_t, 6),
                "phoneme_idx": ph_idx,
                "phoneme_symbol": utt["phoneme_symbols"][ph_idx] if ph_idx >= 0 else "<none>",
                "phoneme_id": ph_id,
                "phone_progress": round(raw_phone_progress, 6),
                "phone_progress_bucket": ph_prog_bucket,
                "phone_progress_bucket_check": raw_phone_progress_bucket,
                "note_idx": note_idx,
                "note_midi": note_id,
                "note_start": round(float(note_start), 6),
                "note_end": round(float(note_end), 6),
                "note_slur": slur,
                "note_on_boundary": bool(note_on),
                "note_off_boundary": bool(note_off),
            }
        )
    return rows


def print_text_summary(utt: dict[str, Any], rows: list[dict[str, Any]], max_steps: int | None) -> None:
    print(f"utt_id: {utt['utt_id']}")
    print(f"phonemes: {utt['phoneme_symbols']}")
    print(f"phoneme_intervals: {utt['phoneme_intervals']}")
    print(f"note_pitch_midi: {utt['note_pitch_midi']}")
    print(f"note_intervals: {utt['note_intervals']}")
    print(f"note_slur: {utt.get('note_slur', [])}")
    print("")
    print("step_idx center_t phoneme_idx phoneme phone_prog prog_bin note_idx note_midi slur on_b off_b")
    for row in rows[:max_steps]:
        print(
            f"{row['step_idx']:>7} "
            f"{row['center_t']:>8.3f} "
            f"{row['phoneme_idx']:>11} "
            f"{row['phoneme_symbol']:<10} "
            f"{row['phone_progress']:>10.3f} "
            f"{row['phone_progress_bucket']:>8} "
            f"{row['note_idx']:>8} "
            f"{row['note_midi']:>9} "
            f"{row['note_slur']:>4} "
            f"{int(row['note_on_boundary']):>4} "
            f"{int(row['note_off_boundary']):>5}"
        )


def main() -> int:
    args = parse_args()
    utt = load_single_utterance(args)
    rows = simulate_rows(
        utt=utt,
        frame_rate=float(args.frame_rate),
        tokens_per_step=int(args.tokens_per_step),
        phone_progress_bins=int(args.phone_progress_bins),
    )
    max_steps = args.max_steps if args.max_steps is not None else len(rows)
    if args.json:
        payload = {
            "utt_id": utt["utt_id"],
            "rows": rows[:max_steps],
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    print_text_summary(utt, rows, max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
