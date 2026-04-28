from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from comparison_md_to_ipynb import build_html, parse_md


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build root-level checkpoint_qc bundle (md/html/ipynb) from per-split comparison files.")
    p.add_argument("--qc-root", required=True, help="checkpoint_qc root directory")
    p.add_argument("--splits", nargs="+", default=["valid", "test"], help="Ordered split list to include")
    p.add_argument("--title", default="Checkpoint QC Comparison")
    return p.parse_args()


def build_notebook_bundle(title: str, sections: list[dict]) -> dict:
    cells: list[dict] = []
    for sec in sections:
        split = sec["split"]
        if sec["exists"]:
            html = sec["html"]
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/html": [html],
                                "text/plain": [f"<rendered audio table: {split}>"],
                            },
                            "metadata": {},
                        }
                    ],
                    "source": [f"# {split} comparison\n"],
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## {split}\n", "\n", "comparison.md 不存在，当前尚未生成该 split 的 QC 结果。\n"],
                }
            )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
            "checkpoint_qc": {
                "title": title,
                "splits": [sec["split"] for sec in sections],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _with_split_prefix(path: str, split: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme:
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    if path.startswith(f"{split}/"):
        return path
    return f"{split}/{path}"


def _prefix_row_paths(rows: list[dict], split: str) -> list[dict]:
    patched: list[dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["gt_path"] = _with_split_prefix(str(row["gt_path"]), split)
        new_row["codec_path"] = _with_split_prefix(str(row["codec_path"]), split)
        new_results = []
        for item in row["checkpoint_results"]:
            new_item = dict(item)
            new_item["wav_path"] = _with_split_prefix(str(item["wav_path"]), split)
            new_results.append(new_item)
        new_row["checkpoint_results"] = new_results
        patched.append(new_row)
    return patched


def main() -> None:
    args = parse_args()
    qc_root = Path(args.qc_root).resolve()
    sections: list[dict] = []
    md_lines = [f"# {args.title}", ""]
    html_sections: list[str] = []

    for split in args.splits:
        split_dir = qc_root / split
        md_path = split_dir / "comparison.md"
        html_path = split_dir / "comparison.html"
        ipynb_path = split_dir / "comparison.ipynb"
        if not md_path.exists():
            sections.append({"split": split, "exists": False})
            md_lines.extend(
                [
                    f"## {split}",
                    "",
                    f"- comparison: 缺失 (`{split}/comparison.md`)",
                    "",
                ]
            )
            html_sections.append(
                f"<section><h2>{split}</h2><p>comparison.md 不存在，当前尚未生成该 split 的 QC 结果。</p></section>"
            )
            continue

        split_title, ckpt_labels, rows = parse_md(md_path)
        prefixed_rows = _prefix_row_paths(rows, split)
        html_table = build_html(f"{split_title} [{split}]", ckpt_labels, prefixed_rows)
        sections.append({"split": split, "exists": True, "html": html_table})
        md_lines.extend(
            [
                f"## {split}",
                "",
                f"- [comparison.md]({split}/comparison.md)",
                f"- [comparison.html]({split}/comparison.html)" if html_path.exists() else f"- comparison.html: 缺失 (`{split}/comparison.html`)",
                f"- [comparison.ipynb]({split}/comparison.ipynb)" if ipynb_path.exists() else f"- comparison.ipynb: 缺失 (`{split}/comparison.ipynb`)",
                "",
            ]
        )
        html_sections.append(f"<section><h2>{split}</h2>{html_table}</section>")

    (qc_root / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    html_doc = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{args.title}</title></head><body>"
        f"<h1>{args.title}</h1>{''.join(html_sections)}</body></html>"
    )
    (qc_root / "comparison.html").write_text(html_doc, encoding="utf-8")
    nb = build_notebook_bundle(args.title, sections)
    (qc_root / "comparison.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(qc_root / "comparison.ipynb")


if __name__ == "__main__":
    main()
