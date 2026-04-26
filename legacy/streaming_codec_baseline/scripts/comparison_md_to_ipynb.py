from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote


GT_RE = re.compile(r"\[gt\]\(([^)]+)\)")
CODEC_RE = re.compile(r"\[codec_q2\]\(([^)]+)\)")
WAV_RE = re.compile(r"\[wav\]\(([^)]+)\)<br>`acc=([0-9.]+)`")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert comparison.md into a VS Code-friendly notebook with audio players.")
    parser.add_argument("--md", required=True, help="Path to comparison.md")
    parser.add_argument("--out", default=None, help="Output ipynb path; defaults to comparison.ipynb beside the markdown file.")
    return parser.parse_args()


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def encode_uri(path: str) -> str:
    return quote(path, safe="/:._-()")


def parse_md(md_path: Path) -> tuple[str, list[str], list[dict]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    header = next(line for line in lines if line.startswith("| idx |"))
    columns = [c.strip() for c in header.strip("|").split("|")]
    ckpt_labels = columns[4:]

    rows: list[dict] = []
    for line in lines:
        if not line.startswith("| ") or line.startswith("|---") or line == header:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        gt_path = GT_RE.search(cells[2]).group(1)
        codec_path = CODEC_RE.search(cells[3]).group(1)
        checkpoint_results = []
        for label, cell in zip(ckpt_labels, cells[4:]):
            m = WAV_RE.search(cell)
            checkpoint_results.append(
                {
                    "label": label,
                    "wav_path": m.group(1),
                    "acc": float(m.group(2)),
                }
            )
        rows.append(
            {
                "idx": cells[0],
                "utt_id": cells[1].strip("`"),
                "gt_path": gt_path,
                "codec_path": codec_path,
                "checkpoint_results": checkpoint_results,
            }
        )
    return title, ckpt_labels, rows


def build_html(title: str, ckpt_labels: list[str], rows: list[dict]) -> str:
    header_cols = "".join(f"<th>{html_escape(label)}</th>" for label in ckpt_labels)
    body_rows = []
    for row in rows:
        cells = [
            f"<td class='meta'>{html_escape(row['idx'])}</td>",
            f"<td class='meta'><code>{html_escape(row['utt_id'])}</code></td>",
            (
                "<td><div class='cell'>"
                f"<a href='{html_escape(encode_uri(row['gt_path']))}'>{html_escape(Path(row['gt_path']).name)}</a>"
                f"<audio controls preload='none' src='{html_escape(encode_uri(row['gt_path']))}'></audio>"
                "</div></td>"
            ),
            (
                "<td><div class='cell'>"
                f"<a href='{html_escape(encode_uri(row['codec_path']))}'>codec_recon_q2.wav</a>"
                f"<audio controls preload='none' src='{html_escape(encode_uri(row['codec_path']))}'></audio>"
                "</div></td>"
            ),
        ]
        for item in row["checkpoint_results"]:
            uri = encode_uri(item["wav_path"])
            cells.append(
                "<td><div class='cell'>"
                f"<a href='{html_escape(uri)}'>{html_escape(item['label'])}.wav</a>"
                f"<audio controls preload='none' src='{html_escape(uri)}'></audio>"
                f"<div class='score'>acc={item['acc']:.6f}</div>"
                "</div></td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<style>
.wrap {{
  padding: 12px;
  font-family: ui-serif, Georgia, serif;
  color: #1d1b16;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 28px;
}}
p.note {{
  margin: 0 0 16px;
  color: #5d5648;
}}
.table-wrap {{
  overflow: auto;
  border: 1px solid #d8cfba;
  background: #fffdf8;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  min-width: 1800px;
}}
th, td {{
  border: 1px solid #d8cfba;
  padding: 10px;
  vertical-align: top;
}}
th {{
  position: sticky;
  top: 0;
  background: #efe5cf;
  text-align: left;
}}
.meta {{
  white-space: nowrap;
}}
.cell {{
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 170px;
}}
.score {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #8e3b2e;
  font-size: 12px;
}}
audio {{
  width: 100%;
  min-width: 160px;
  height: 32px;
}}
a {{
  color: #8e3b2e;
  text-decoration: none;
}}
a:hover {{
  text-decoration: underline;
}}
</style>
<div class="wrap">
  <h1>{html_escape(title)}</h1>
  <p class="note">VS Code notebook 预览版。每个单元格都带音频播放器和原始 wav 链接。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>idx</th>
          <th>utt_id</th>
          <th>GT</th>
          <th>codec_recon_q2</th>
          {header_cols}
        </tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
  </div>
</div>"""


def build_notebook(title: str, html: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "这个 notebook 已经预渲染，不需要先执行单元格。\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {
                            "text/html": [html],
                            "text/plain": [f"<rendered audio table: {title}>"],
                        },
                        "metadata": {},
                    }
                ],
                "source": ["from IPython.display import HTML\n", "HTML('<comparison preview rendered in output>')\n"],
            },
        ],
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    args = parse_args()
    md_path = Path(args.md).resolve()
    out_path = Path(args.out).resolve() if args.out else md_path.with_suffix(".ipynb")
    title, ckpt_labels, rows = parse_md(md_path)
    html = build_html(title, ckpt_labels, rows)
    notebook = build_notebook(title, html)
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
