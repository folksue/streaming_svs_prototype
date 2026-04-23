#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from metadata_adapters import load_manifest_entries  # type: ignore
from utils import ensure_dir, save_json  # type: ignore


def stable_hash_score(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16)


def rebase_audio_paths(entries: list[dict], out_dir: Path) -> list[dict]:
    """
    Mixed manifests are saved under artifacts/, but upstream adapters may return
    audio paths relative to other working directories (often BASELINE_ROOT).
    Standard-manifest loading resolves relative paths w.r.t. the manifest file
    directory, so we rebase them here to be relative to the output dir.
    """
    out_dir_abs = out_dir.resolve()
    rebased: list[dict] = []
    for entry in entries:
        audio_path = str(entry.get("audio_path", "")).strip()
        if not audio_path:
            raise ValueError(f"Entry {entry.get('utt_id', '<unknown>')} is missing audio_path")

        audio_abs = Path(audio_path)
        if not audio_abs.is_absolute():
            audio_abs = (BASELINE_ROOT / audio_abs).resolve()

        rel = None
        try:
            rel = audio_abs.relative_to(out_dir_abs)
        except ValueError:
            rel = Path(os.path.relpath(str(audio_abs), str(out_dir_abs)))

        entry2 = dict(entry)
        entry2["audio_path"] = str(rel)
        rebased.append(entry2)
    return rebased


def pick_deterministic_subset(items: list[dict], seed: int, n: int) -> tuple[list[dict], list[dict]]:
    if n <= 0:
        return [], list(items)
    if n >= len(items):
        return list(items), []

    scored = sorted(items, key=lambda it: stable_hash_score(str(it.get("utt_id", "")), seed))
    chosen = scored[:n]
    rest = scored[n:]
    return chosen, rest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mixed OpenCPOP+M4Singer manifests for legacy baseline.")
    p.add_argument("--opencpop-train", type=Path, required=True)
    p.add_argument("--opencpop-valid", type=Path, required=True)
    p.add_argument("--m4singer-meta", type=Path, required=True)
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-valid", type=Path, required=True)
    p.add_argument("--m4singer-valid-ratio", type=float, default=0.05)
    p.add_argument("--min-m4-valid", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    opencpop_train = load_manifest_entries(str(args.opencpop_train), adapter_name="opencpop", audio_root=None)
    opencpop_valid = load_manifest_entries(str(args.opencpop_valid), adapter_name="opencpop", audio_root=None)

    m4_entries = load_manifest_entries(str(args.m4singer_meta), adapter_name="m4singer", audio_root=None)
    m4_total = len(m4_entries)
    ratio = max(float(args.m4singer_valid_ratio), 0.0)
    n_m4_valid = int(round(m4_total * ratio))
    n_m4_valid = max(n_m4_valid, int(args.min_m4_valid))
    if m4_total > 0:
        n_m4_valid = min(n_m4_valid, m4_total - 1)
    else:
        n_m4_valid = 0

    m4_valid, m4_train = pick_deterministic_subset(m4_entries, seed=int(args.seed), n=n_m4_valid)

    mixed_train = list(opencpop_train) + list(m4_train)
    mixed_valid = list(opencpop_valid) + list(m4_valid)

    ensure_dir(str(args.out_train.parent))
    ensure_dir(str(args.out_valid.parent))
    # Rebase audio paths to be resolvable from the manifest location.
    mixed_train = rebase_audio_paths(mixed_train, out_dir=args.out_train.parent)
    mixed_valid = rebase_audio_paths(mixed_valid, out_dir=args.out_valid.parent)

    save_json(str(args.out_train), mixed_train)
    save_json(str(args.out_valid), mixed_valid)

    print(
        "Built mixed manifests:\n"
        f"- OpenCPOP train: {len(opencpop_train)}\n"
        f"- OpenCPOP valid: {len(opencpop_valid)}\n"
        f"- M4Singer total: {m4_total}\n"
        f"- M4Singer train: {len(m4_train)}\n"
        f"- M4Singer valid: {len(m4_valid)}\n"
        f"- Mixed train: {len(mixed_train)} -> {args.out_train}\n"
        f"- Mixed valid: {len(mixed_valid)} -> {args.out_valid}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
