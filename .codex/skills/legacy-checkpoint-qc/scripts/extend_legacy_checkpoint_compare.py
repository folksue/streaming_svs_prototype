from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
LEGACY_DIR = REPO_ROOT / "legacy" / "streaming_codec_baseline"
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from scripts.sample_checkpoint_compare import (  # type: ignore
    checkpoint_label,
    format_link_target,
    run_checkpoint_inference,
    write_html,
    write_markdown,
)


ROW_RE = re.compile(r"^sample_(\d+)$")
CELL_RE = re.compile(r"\[wav\]\(([^)]+)\)<br>`acc=([0-9.]+)`")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend an existing legacy checkpoint comparison page without re-running completed columns.")
    parser.add_argument("--config", required=True, help="Legacy baseline YAML config.")
    parser.add_argument("--split", required=True, choices=["train", "valid"])
    parser.add_argument("--comparison-dir", required=True, help="Directory that already contains comparison.md and sample_* folders.")
    parser.add_argument("--title", default=None, help="Optional title override for regenerated comparison page.")
    return parser.parse_args()


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_existing_md(md_path: Path) -> tuple[list[str], list[dict]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("| idx |"))
    columns = [c.strip() for c in header.strip("|").split("|")]
    checkpoint_columns = columns[4:]

    rows: list[dict] = []
    for line in lines:
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        if line == header:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        idx = int(cells[0])
        utt_id = cells[1].strip("`")
        gt_path = re.search(r"\[gt\]\(([^)]+)\)", cells[2]).group(1)
        codec_path = re.search(r"\[codec_q2\]\(([^)]+)\)", cells[3]).group(1)
        checkpoint_results = {}
        for label, cell in zip(checkpoint_columns, cells[4:]):
            m = CELL_RE.search(cell)
            if not m:
                raise ValueError(f"Failed to parse cell for {label}: {cell}")
            checkpoint_results[label] = {
                "wav_path": m.group(1),
                "token_acc": float(m.group(2)),
            }
        rows.append(
            {
                "idx": idx,
                "utt_id": utt_id,
                "gt_path": gt_path,
                "codec_path": codec_path,
                "checkpoint_results": checkpoint_results,
            }
        )
    return checkpoint_columns, rows


def collect_candidate_checkpoints(ckpt_dir: Path) -> list[Path]:
    paths = []
    best = ckpt_dir / "best.pt"
    if best.exists():
        paths.append(best)
    paths.extend(sorted(ckpt_dir.glob("epoch_*.pt")))
    last = ckpt_dir / "last.pt"
    if last.exists():
        paths.append(last)
    return paths


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)
    comparison_dir = Path(args.comparison_dir).resolve()
    md_path = comparison_dir / "comparison.md"
    html_path = comparison_dir / "comparison.html"
    checkpoint_labels, sample_rows = parse_existing_md(md_path)

    ckpt_dir = (cfg_path.parent / cfg["paths"]["checkpoints"]).resolve()
    cache_rel = cfg["data"]["train_cache"] if args.split == "train" else cfg["data"]["valid_cache"]
    cache_path = (cfg_path.parent / cache_rel).resolve()

    candidate_checkpoints = collect_candidate_checkpoints(ckpt_dir)
    missing_paths = [p for p in candidate_checkpoints if checkpoint_label(p) not in checkpoint_labels]

    indices = [row["idx"] for row in sample_rows]
    for checkpoint_path in missing_paths:
        label = checkpoint_label(checkpoint_path)
        results = run_checkpoint_inference(
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
            indices=indices,
            out_root=comparison_dir,
        )
        for row in sample_rows:
            result = dict(results[row["idx"]])
            result["wav_path"] = format_link_target(Path(result["wav_path"]), "relative", comparison_dir)
            row["checkpoint_results"][label] = result
        checkpoint_labels.append(label)

    checkpoint_columns = [{"label": label} for label in checkpoint_labels]
    title = args.title or f"{cfg_path.stem} {args.split} checkpoint comparison"

    write_markdown(
        out_path=md_path,
        title=title,
        sample_rows=sample_rows,
        checkpoint_columns=checkpoint_columns,
    )
    write_html(
        out_path=html_path,
        title=title,
        sample_rows=sample_rows,
        checkpoint_columns=checkpoint_columns,
    )
    print(f"updated {md_path}")
    print(f"added {len(missing_paths)} checkpoints")


if __name__ == "__main__":
    main()
