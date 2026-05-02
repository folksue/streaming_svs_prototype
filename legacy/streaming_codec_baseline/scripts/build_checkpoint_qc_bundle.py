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


def build_notebook_bundle(title: str, sections: list[dict], teacher_forced_html: str | None = None) -> dict:
    cells: list[dict] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                "\n",
                "直接使用下面的 `train comparison` / `valid comparison` cell 即可查看或重建结果；最后一个 cell 用于按样本和 checkpoint 查看 logits 细节。\n",
            ],
        },
    ]
    for sec in sections:
        split = sec["split"]
        outputs = []
        if sec.get("exists") and sec.get("html"):
            outputs = [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/html": [sec["html"]],
                        "text/plain": [f"<rendered audio table: {split}>"],
                    },
                    "metadata": {},
                }
            ]
        cells.append(
            {
                "cell_type": "code",
                "execution_count": 1 if outputs else None,
                "metadata": {},
                "outputs": outputs,
                "source": [
                    f"# {split} comparison: regenerate + load html\n",
                    "from pathlib import Path\n",
                    "import sys\n",
                    "import importlib\n",
                    "\n",
                    "repo = Path.cwd().resolve()\n",
                    "while repo != repo.parent and not (repo / '.git').exists():\n",
                    "    repo = repo.parent\n",
                    "if not (repo / '.git').exists():\n",
                    "    raise RuntimeError('Could not locate repo root (.git) from current working directory')\n",
                    "sys.path.insert(0, str(repo / 'legacy/streaming_codec_baseline'))\n",
                    "import scripts.qc_notebook_tools as qc_notebook_tools\n",
                    "importlib.reload(qc_notebook_tools)\n",
                    "run_split_compare = qc_notebook_tools.run_split_compare\n",
                    "display_split_html = qc_notebook_tools.display_split_html\n",
                    "\n",
                    "run_name = 'mixed_4ds_ace_100m_8cq'\n",
                    f"split = '{split}'\n",
                    "DO_REGENERATE = False\n",
                    "html_path = repo / f'legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split}/comparison.html'\n",
                    "if DO_REGENERATE or not html_path.exists():\n",
                    "    run_split_compare(run_name=run_name, split=split, sample_count=10, seed=42, dump_logits=True, logits_topk=10, out_subdir=split, force_same_samples=True)\n",
                    "display_split_html(run_name=run_name, split_or_subdir=split)\n",
                ],
            }
        )

    tf_outputs = []
    if teacher_forced_html:
        tf_outputs = [
            {
                "output_type": "display_data",
                "data": {
                    "text/html": [teacher_forced_html],
                    "text/plain": ["<rendered audio table: valid_teacher_forced>"],
                },
                "metadata": {},
            }
        ]
    cells.append(
        {
            "cell_type": "code",
            "execution_count": 1 if tf_outputs else None,
            "metadata": {},
            "outputs": tf_outputs,
            "source": [
                "# teacher-forced recon comparison: use GT history, only predict current step\n",
                "from pathlib import Path\n",
                "import sys\n",
                "import importlib\n",
                "\n",
                "repo = Path.cwd().resolve()\n",
                "while repo != repo.parent and not (repo / '.git').exists():\n",
                "    repo = repo.parent\n",
                "if not (repo / '.git').exists():\n",
                "    raise RuntimeError('Could not locate repo root (.git) from current working directory')\n",
                "sys.path.insert(0, str(repo / 'legacy/streaming_codec_baseline'))\n",
                "import scripts.qc_notebook_tools as qc_notebook_tools\n",
                "importlib.reload(qc_notebook_tools)\n",
                "run_teacher_forced_compare = qc_notebook_tools.run_teacher_forced_compare\n",
                "display_teacher_forced_html = qc_notebook_tools.display_teacher_forced_html\n",
                "\n",
                "run_name = 'mixed_4ds_ace_100m_8cq'\n",
                "split = 'valid'  # or train\n",
                "subdir = f'{split}_teacher_forced'\n",
                "DO_REGENERATE = False\n",
                "html_path = repo / f'legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{subdir}/comparison.html'\n",
                "if DO_REGENERATE or not html_path.exists():\n",
                "    ckpt_dir = repo / f'legacy/streaming_codec_baseline/artifacts/checkpoints/{run_name}'\n",
                "    checkpoints = [ckpt_dir / 'best.pt', ckpt_dir / 'last.pt']\n",
                "    run_teacher_forced_compare(\n",
                "        run_name=run_name,\n",
                "        split=split,\n",
                "        sample_count=10,\n",
                "        seed=42,\n",
                "        checkpoints=checkpoints,\n",
                "        out_subdir=subdir,\n",
                "        force_same_samples=True,\n",
                "    )\n",
                "display_teacher_forced_html(run_name=run_name, split_or_subdir=subdir)\n",
            ],
        }
    )

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# logits debug: select sample + ckpts, then inspect audio and per-step logits details\n",
                "from pathlib import Path\n",
                "import sys\n",
                "import importlib\n",
                "\n",
                "repo = Path.cwd().resolve()\n",
                "while repo != repo.parent and not (repo / '.git').exists():\n",
                "    repo = repo.parent\n",
                "if not (repo / '.git').exists():\n",
                "    raise RuntimeError('Could not locate repo root (.git) from current working directory')\n",
                "sys.path.insert(0, str(repo / 'legacy/streaming_codec_baseline'))\n",
                "import scripts.qc_notebook_tools as qc_notebook_tools\n",
                "importlib.reload(qc_notebook_tools)\n",
                "render_logits_debug = qc_notebook_tools.render_logits_debug\n",
                "\n",
                "run_name = 'mixed_4ds_ace_100m_8cq'\n",
                "subdir = 'valid'  # or train\n",
                "sample_id = 'sample_0051'  # 改这里，先看上面打印出的 available samples\n",
                "selected_ckpt_tags = ['best_ep010', 'last_ep080']  # [] 表示该 sample 下全部 ckpt\n",
                "step_t = None  # 仅在 step_range 不为 None 时会被忽略；仅保留兼容\n",
                "step_range = None  # None: 打印全部 step；例如 (100, 140): 只展示这段区间内每一个 step\n",
                "layer_indices = [0]  # 例如 [0] 看 coarse；[0,1,2] 看多层；None 表示全部层\n",
                "render_logits_debug(\n",
                "    run_name=run_name,\n",
                "    split_or_subdir=subdir,\n",
                "    sample_id=sample_id,\n",
                "    ckpt_tags=selected_ckpt_tags,\n",
                "    step_t=step_t,\n",
                "    step_range=step_range,\n",
                "    layer_indices=layer_indices,\n",
                ")\n",
            ],
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
            md_lines.extend([f"## {split}", "", f"- comparison: 缺失 (`{split}/comparison.md`)", ""])
            html_sections.append(f"<section><h2>{split}</h2><p>comparison.md 不存在，当前尚未生成该 split 的 QC 结果。</p></section>")
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
    teacher_forced_html = None
    tf_md_path = qc_root / "valid_teacher_forced" / "comparison.md"
    if tf_md_path.exists():
        tf_title, tf_ckpt_labels, tf_rows = parse_md(tf_md_path)
        tf_prefixed_rows = _prefix_row_paths(tf_rows, "valid_teacher_forced")
        teacher_forced_html = build_html(f"{tf_title} [valid_teacher_forced]", tf_ckpt_labels, tf_prefixed_rows)

    nb = build_notebook_bundle(args.title, sections, teacher_forced_html=teacher_forced_html)
    (qc_root / "comparison.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(qc_root / "comparison.ipynb")


if __name__ == "__main__":
    main()
