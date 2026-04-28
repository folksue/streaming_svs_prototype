from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.sample_checkpoint_compare import (  # type: ignore
    checkpoint_label,
    format_link_target,
    run_checkpoint_inference,
    write_html,
    write_markdown,
)


CELL_RE = re.compile(r"\[wav\]\(([^)]+)\)<br>`acc=([0-9.]+)`")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append exactly one checkpoint column into an existing comparison page.")
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--comparison-dir", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--link-style", choices=["path", "fileuri", "relative"], default="relative")
    return p.parse_args()


def parse_existing_md(md_path: Path) -> tuple[str, list[str], list[dict]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    title_line = next((line for line in lines if line.startswith("# ")), "# Checkpoint Comparison")
    title = title_line[2:].strip()
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
                continue
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
    return title, checkpoint_columns, rows


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir).resolve()
    md_path = comparison_dir / "comparison.md"
    html_path = comparison_dir / "comparison.html"
    cache_path = Path(args.cache).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    label = checkpoint_label(checkpoint_path)

    existing_title, checkpoint_labels, sample_rows = parse_existing_md(md_path)
    title = args.title or existing_title
    added = False
    if label not in checkpoint_labels:
        added = True
        indices = [row["idx"] for row in sample_rows]
        results = run_checkpoint_inference(
            checkpoint_path=checkpoint_path,
            cache_path=cache_path,
            indices=indices,
            out_root=comparison_dir,
        )
        for row in sample_rows:
            result = dict(results[row["idx"]])
            result["wav_path"] = format_link_target(Path(result["wav_path"]), args.link_style, comparison_dir)
            row["checkpoint_results"][label] = result
        checkpoint_labels.append(label)

    checkpoint_columns = [{"label": c} for c in checkpoint_labels]
    write_markdown(md_path, title, sample_rows, checkpoint_columns)
    write_html(html_path, title, sample_rows, checkpoint_columns)
    print(f"updated {md_path}")
    print(f"{'appended' if added else 'already_present'} {label}")


if __name__ == "__main__":
    main()
