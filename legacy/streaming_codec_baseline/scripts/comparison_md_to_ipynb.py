from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote


GT_RE = re.compile(r"\[gt\]\(([^)]+)\)")
CODEC_RE = re.compile(r"\[(?:codec|codec_q2)\]\(([^)]+)\)")
WAV_RE = re.compile(r"\[wav\]\(([^)]+)\)<br>`acc=([0-9.eE+-]+|nan)`")
LOGITS_RE = re.compile(r"\[logits\]\(([^)]+)\)")


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
            m_logits = LOGITS_RE.search(cell)
            checkpoint_results.append(
                {
                    "label": label,
                    "wav_path": m.group(1),
                    "acc": float(m.group(2)),
                    "logits_path": m_logits.group(1) if m_logits else "",
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
    regen_cell_source = [
        "from pathlib import Path\n",
        "import sys\n",
        "\n",
        "repo = Path.cwd().resolve()\n",
        "while repo != repo.parent and not (repo / '.git').exists():\n",
        "    repo = repo.parent\n",
        "if not (repo / '.git').exists():\n",
        "    raise RuntimeError('Could not locate repo root (.git) from current working directory')\n",
        "sys.path.insert(0, str(repo / 'legacy/streaming_codec_baseline'))\n",
        "from scripts.qc_notebook_tools import run_split_compare\n",
        "out_dir = run_split_compare(\n",
        "    run_name='mixed_4ds_ace_100m_8cq',\n",
        "    split='valid',\n",
        "    sample_count=5,\n",
        "    seed=42,\n",
        "    dump_logits=True,\n",
        "    logits_topk=10,\n",
        "    checkpoints=[\n",
        "        repo / 'legacy/streaming_codec_baseline/artifacts/checkpoints/mixed_4ds_ace_100m_8cq/best.pt',\n",
        "        repo / 'legacy/streaming_codec_baseline/artifacts/checkpoints/mixed_4ds_ace_100m_8cq/last.pt',\n",
        "    ],\n",
        "    out_subdir='valid',\n",
        "    force_same_samples=True,\n",
        ")\n",
        "print('Done. Output:', out_dir / 'comparison.ipynb')\n",
    ]

    debug_cell_source = [
        "import json\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import display\n",
        "\n",
        "comparison_md = Path('comparison.md').resolve()\n",
        "sample_index = 0\n",
        "checkpoint_index = 0\n",
        "\n",
        "def _parse_rows(md_path: Path):\n",
        "    rows = []\n",
        "    for line in md_path.read_text(encoding='utf-8').splitlines():\n",
        "        if not line.startswith('| ') or line.startswith('|---') or line.startswith('| idx |'):\n",
        "            continue\n",
        "        cells = [c.strip() for c in line.strip('|').split('|')]\n",
        "        if len(cells) < 5:\n",
        "            continue\n",
        "        ckpt_cells = cells[4:]\n",
        "        parsed_ckpts = []\n",
        "        for cell in ckpt_cells:\n",
        "            m_wav = re.search(r\"\\[wav\\]\\(([^)]+)\\)\", cell)\n",
        "            m_logits = re.search(r\"\\[logits\\]\\(([^)]+)\\)\", cell)\n",
        "            parsed_ckpts.append({'wav_path': m_wav.group(1) if m_wav else '', 'logits_path': m_logits.group(1) if m_logits else ''})\n",
        "        rows.append({'idx': int(cells[0]), 'utt_id': cells[1].strip('`'), 'ckpts': parsed_ckpts})\n",
        "    return rows\n",
        "\n",
        "import re\n",
        "rows = _parse_rows(comparison_md)\n",
        "row = rows[sample_index]\n",
        "ckpt_pos = int(checkpoint_index)\n",
        "logits_path = Path(row['ckpts'][ckpt_pos]['logits_path']).resolve()\n",
        "if not logits_path.exists():\n",
        "    raise FileNotFoundError(f'logits debug not found: {logits_path}')\n",
        "d = np.load(logits_path, allow_pickle=True)\n",
        "pred = d['pred_tokens']\n",
        "gt = d['gt_tokens'] if 'gt_tokens' in d else None\n",
        "gt_rank = d['gt_rank'] if 'gt_rank' in d else None\n",
        "print('utt_id=', row['utt_id'])\n",
        "print('logits_path=', logits_path)\n",
        "print('shape pred=', pred.shape)\n",
        "if gt is None:\n",
        "    print('No gt tensors in npz.')\n",
        "else:\n",
        "    mismatch = (pred != gt)\n",
        "    err_cnt = int(mismatch.sum())\n",
        "    total = int(mismatch.size)\n",
        "    print(f'mismatch tokens: {err_cnt}/{total} ({err_cnt/total:.4%})')\n",
        "    step_err = mismatch.mean(axis=(1,2))\n",
        "    summary_df = pd.DataFrame({'t': np.arange(len(step_err)), 'error_rate': step_err})\n",
        "    display(summary_df.sort_values('error_rate', ascending=False).head(50))\n",
        "    rows_out = []\n",
        "    T,S,K = mismatch.shape\n",
        "    for t in range(T):\n",
        "        for s in range(S):\n",
        "            for k in range(K):\n",
        "                if mismatch[t,s,k]:\n",
        "                    rows_out.append({'t': t, 's': s, 'k': k, 'pred': int(pred[t,s,k]), 'gt': int(gt[t,s,k]), 'gt_rank': int(gt_rank[t,s,k]) if gt_rank is not None else -1})\n",
        "    df = pd.DataFrame(rows_out)\n",
        "    if len(df) == 0:\n",
        "        print('No mismatches.')\n",
        "    else:\n",
        "        display(df.sort_values(['gt_rank','t','s','k'], ascending=[False,True,True,True]).head(200))\n",
        "    heat = mismatch.mean(axis=2)\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.imshow(heat.T, aspect='auto', origin='lower')\n",
        "    plt.colorbar(label='error rate over codebooks')\n",
        "    plt.xlabel('time step t')\n",
        "    plt.ylabel('slot s')\n",
        "    plt.title('Token Error Heatmap (slot x time)')\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
    ]

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
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": regen_cell_source,
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": debug_cell_source,
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
