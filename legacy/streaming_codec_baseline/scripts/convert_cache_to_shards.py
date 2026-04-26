from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to legacy single-file cache .pt")
    p.add_argument("--output-dir", required=True, help="Directory to write sharded cache")
    p.add_argument("--items-per-shard", type=int, default=2048)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    items_per_shard = max(1, int(args.items_per_shard))

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir already exists: {output_dir}. Pass --force to overwrite.")
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise SystemExit(f"Refusing to overwrite nested directory inside {output_dir}: {child}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = torch.load(input_path, map_location="cpu", weights_only=False)
    items = payload["items"]
    meta = dict(payload["meta"])
    meta["cache_format_version"] = 4

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, int]] = []
    shards: list[dict[str, Any]] = []
    total_items = len(items)
    for shard_idx, start in enumerate(range(0, total_items, items_per_shard)):
        shard_items = items[start : start + items_per_shard]
        shard_path = shard_dir / f"shard_{shard_idx:04d}.pt"
        torch.save({"items": shard_items}, shard_path)
        shards.append(
            {
                "path": str(shard_path),
                "start_idx": start,
                "length": len(shard_items),
            }
        )
        for item_idx in range(len(shard_items)):
            index.append({"shard_idx": shard_idx, "item_idx": item_idx})

    index_payload = {
        "items": [],
        "meta": meta,
        "storage": "sharded",
        "num_items": total_items,
        "index": index,
        "shards": shards,
    }
    torch.save(index_payload, output_dir / "index.pt")

    print(
        f"Converted {input_path} -> {output_dir} | items={total_items} "
        f"| shards={len(shards)} | items_per_shard={items_per_shard}"
    )


if __name__ == "__main__":
    main()
