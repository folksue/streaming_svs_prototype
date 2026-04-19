#!/usr/bin/env python3
"""Estimate dataset duration and equivalent token budgets for audio models.

This utility supports two common workflows:

1. Scan a local audio directory and sum durations from audio files.
2. Start from an already known duration (for example from a paper or model card).

The token estimate is intentionally simple and reusable:

    equivalent_frames = total_seconds * token_rate_hz
    flattened_tokens = equivalent_frames * num_codebooks
    model_steps = equivalent_frames / tokens_per_step

This is an estimate, not an exact codec frame count. It ignores codec-specific
padding, chunk overlap, and boundary effects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sf = None


DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


@dataclass
class DurationStats:
    num_files: int
    total_seconds: float
    skipped_files: int

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate audio duration and equivalent token budgets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio-root",
        type=Path,
        help="Recursively scan audio files under this directory.",
    )
    source.add_argument(
        "--audio-list",
        type=Path,
        help="Text file containing one audio path per line.",
    )
    source.add_argument(
        "--duration-hours",
        type=float,
        help="Known total duration in hours.",
    )
    source.add_argument(
        "--duration-seconds",
        type=float,
        help="Known total duration in seconds.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated audio extensions when scanning directories.",
    )
    parser.add_argument(
        "--token-rate-hz",
        type=float,
        required=True,
        help="Codec frame or token rate in Hz. Example: 75 for EnCodec 24k, 12.5 for semantic/audio tokenizers.",
    )
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=1,
        help="How many codebooks are emitted per frame. Use 1 for coarse-only AR accounting.",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=None,
        help="Optional grouping hyperparameter. If set, also estimate model steps.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label shown in output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def iter_audio_files_from_root(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def iter_audio_files_from_list(list_path: Path) -> Iterable[Path]:
    base = list_path.parent
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = base / path
        yield path


def get_audio_duration_seconds(path: Path) -> float:
    if sf is not None:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
        return float(frames) / float(sample_rate)

    raise RuntimeError(
        f"Cannot read non-WAV file without soundfile installed: {path}"
    )


def scan_audio_files(files: Iterable[Path]) -> DurationStats:
    num_files = 0
    skipped_files = 0
    total_seconds = 0.0
    for path in files:
        try:
            total_seconds += get_audio_duration_seconds(path)
            num_files += 1
        except Exception as exc:
            skipped_files += 1
            print(f"[warn] skipped {path}: {exc}", file=sys.stderr)
    return DurationStats(
        num_files=num_files,
        total_seconds=total_seconds,
        skipped_files=skipped_files,
    )


def build_result(
    stats: DurationStats,
    token_rate_hz: float,
    num_codebooks: int,
    tokens_per_step: int | None,
    label: str | None,
) -> dict[str, object]:
    coarse_frames_exact = stats.total_seconds * token_rate_hz
    flattened_tokens_exact = coarse_frames_exact * num_codebooks
    result: dict[str, object] = {
        "label": label,
        "num_files": stats.num_files,
        "skipped_files": stats.skipped_files,
        "total_seconds": stats.total_seconds,
        "total_hours": stats.total_hours,
        "token_rate_hz": token_rate_hz,
        "num_codebooks": num_codebooks,
        "coarse_frames_exact": coarse_frames_exact,
        "coarse_frames_rounded": int(round(coarse_frames_exact)),
        "flattened_tokens_exact": flattened_tokens_exact,
        "flattened_tokens_rounded": int(round(flattened_tokens_exact)),
    }
    if tokens_per_step is not None:
        steps_exact = coarse_frames_exact / float(tokens_per_step)
        result["tokens_per_step"] = tokens_per_step
        result["model_steps_exact"] = steps_exact
        result["model_steps_rounded"] = int(round(steps_exact))
    return result


def main() -> int:
    args = parse_args()
    extensions = {
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    }

    if args.audio_root is not None:
        stats = scan_audio_files(iter_audio_files_from_root(args.audio_root, extensions))
    elif args.audio_list is not None:
        stats = scan_audio_files(iter_audio_files_from_list(args.audio_list))
    elif args.duration_hours is not None:
        stats = DurationStats(
            num_files=0,
            total_seconds=args.duration_hours * 3600.0,
            skipped_files=0,
        )
    elif args.duration_seconds is not None:
        stats = DurationStats(
            num_files=0,
            total_seconds=args.duration_seconds,
            skipped_files=0,
        )
    else:  # pragma: no cover - argparse already enforces this
        raise AssertionError("No duration source provided.")

    result = build_result(
        stats=stats,
        token_rate_hz=args.token_rate_hz,
        num_codebooks=args.num_codebooks,
        tokens_per_step=args.tokens_per_step,
        label=args.label,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0

    label = result["label"] or "dataset"
    print(f"label: {label}")
    print(f"num_files: {result['num_files']}")
    print(f"skipped_files: {result['skipped_files']}")
    print(f"total_seconds: {result['total_seconds']:.3f}")
    print(f"total_hours: {result['total_hours']:.6f}")
    print(f"token_rate_hz: {result['token_rate_hz']}")
    print(f"num_codebooks: {result['num_codebooks']}")
    print(f"coarse_frames_exact: {result['coarse_frames_exact']:.3f}")
    print(f"coarse_frames_rounded: {result['coarse_frames_rounded']}")
    print(f"flattened_tokens_exact: {result['flattened_tokens_exact']:.3f}")
    print(f"flattened_tokens_rounded: {result['flattened_tokens_rounded']}")
    if args.tokens_per_step is not None:
        print(f"tokens_per_step: {result['tokens_per_step']}")
        print(f"model_steps_exact: {result['model_steps_exact']:.3f}")
        print(f"model_steps_rounded: {result['model_steps_rounded']}")
    print("note: estimates ignore codec padding/chunk-boundary effects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
