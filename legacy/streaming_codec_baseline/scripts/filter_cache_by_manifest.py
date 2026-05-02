#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from dataset import load_cache_payload  # type: ignore
from metadata_adapters import load_manifest_entries  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter raw/step cache items by manifest utt_id whitelist.")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--adapter", type=str, default="manifest")
    p.add_argument("--audio-root", type=str, default=None)
    p.add_argument("--cache-in", type=str, required=True)
    p.add_argument("--cache-out", type=str, required=True)
    return p.parse_args()


def collect_keep_ids(manifest: str, adapter: str, audio_root: str | None) -> Set[str]:
    entries = load_manifest_entries(manifest_path=manifest, adapter_name=adapter, audio_root=audio_root)
    keep = {str(x["utt_id"]) for x in entries}
    if not keep:
        raise ValueError(f"No utt_id found from manifest: {manifest}")
    return keep


def filter_raw_cache(cache_in: str, cache_out: str, keep: Set[str]) -> None:
    payload = torch.load(cache_in, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "items" not in payload or "meta" not in payload:
        raise ValueError(f"Invalid raw cache: {cache_in}")
    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError(f"Invalid raw cache items: {cache_in}")
    out_items = [x for x in items if str(x.get("utt_id", "")) in keep]
    if not out_items:
        raise ValueError("Filtering removed all raw cache items.")
    out_payload = {"items": out_items, "meta": payload["meta"]}
    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, cache_out)
    print(f"raw cache filtered: {cache_in} -> {cache_out} | {len(items)} -> {len(out_items)}")


def filter_step_cache_single(cache_in: str, cache_out: str, keep: Set[str]) -> None:
    payload = load_cache_payload(cache_in)
    items = payload["items"]
    out_items = [x for x in items if str(x.get("utt_id", "")) in keep]
    if not out_items:
        raise ValueError("Filtering removed all step cache items.")
    out_payload = {
        "items": out_items,
        "meta": payload["meta"],
        "storage": "single",
    }
    Path(cache_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, cache_out)
    print(f"step cache filtered: {cache_in} -> {cache_out} | {len(items)} -> {len(out_items)}")


def filter_step_cache_sharded(cache_in: str, cache_out: str, keep: Set[str]) -> None:
    in_dir = Path(cache_in)
    out_dir = Path(cache_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_payload = torch.load(in_dir / "index.pt", map_location="cpu", weights_only=False)
    shards = index_payload["shards"]
    meta = index_payload["meta"]

    out_shards: List[Dict[str, Any]] = []
    out_index: List[Dict[str, int]] = []
    total = 0
    shard_counter = 0
    for shard in shards:
        shard_path = Path(shard["path"])
        shard_payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_items = shard_payload["items"]
        filtered = [x for x in shard_items if str(x.get("utt_id", "")) in keep]
        if not filtered:
            continue
        out_shard_path = out_dir / f"shard_{shard_counter:05d}.pt"
        torch.save({"items": filtered}, out_shard_path)
        out_shards.append(
            {
                "path": str(out_shard_path),
                "start_idx": total,
                "length": len(filtered),
            }
        )
        for i in range(len(filtered)):
            out_index.append({"shard_idx": shard_counter, "item_idx": i})
        total += len(filtered)
        shard_counter += 1

    if total == 0:
        raise ValueError("Filtering removed all sharded step cache items.")
    out_index_payload = {
        "items": [],
        "meta": meta,
        "storage": "sharded",
        "num_items": total,
        "index": out_index,
        "shards": out_shards,
    }
    torch.save(out_index_payload, out_dir / "index.pt")
    print(f"sharded step cache filtered: {cache_in} -> {cache_out} | items={total} shards={len(out_shards)}")


def is_raw_cache(payload: Dict[str, Any]) -> bool:
    meta = payload.get("meta", {})
    return str(meta.get("cache_format", "")) == "encodec_codes_only"


def main() -> None:
    args = parse_args()
    keep = collect_keep_ids(args.manifest, args.adapter, args.audio_root)
    print(f"manifest keep ids: {len(keep)}")

    if os.path.isdir(args.cache_in) and (Path(args.cache_in) / "index.pt").exists():
        filter_step_cache_sharded(args.cache_in, args.cache_out, keep)
        return

    payload = torch.load(args.cache_in, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "meta" not in payload:
        raise ValueError(f"Invalid cache payload: {args.cache_in}")

    if is_raw_cache(payload):
        filter_raw_cache(args.cache_in, args.cache_out, keep)
        return

    filter_step_cache_single(args.cache_in, args.cache_out, keep)


if __name__ == "__main__":
    main()
