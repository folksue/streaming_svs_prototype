from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, HTML, display


def find_repo_root(start: Path | None = None) -> Path:
    repo = (start or Path.cwd()).resolve()
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    if not (repo / ".git").exists():
        raise RuntimeError("Could not locate repo root (.git) from current working directory")
    return repo


def run_split_compare(
    run_name: str,
    split: str,
    sample_count: int = 10,
    seed: int = 42,
    dump_logits: bool = False,
    logits_topk: int = 10,
    checkpoints: list[Path] | None = None,
    out_subdir: str | None = None,
    force_same_samples: bool = True,
) -> Path:
    repo = find_repo_root()
    cache = repo / f"legacy/streaming_codec_baseline/artifacts/data/mixed_4ds_ace_8cq_{split}_codes_sharded"
    manifest = repo / f"legacy/streaming_codec_baseline/artifacts/data/mixed_4ds_ace_8cq_{split}_manifest.json"
    ckpt_dir = repo / f"legacy/streaming_codec_baseline/artifacts/checkpoints/{run_name}"
    ckpts = checkpoints or sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    target_subdir = out_subdir or split
    out_dir = repo / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{target_subdir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    indices_file = out_dir / "sample_indices.json"
    base_split_dir = repo / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split}"
    base_indices_file = base_split_dir / "sample_indices.json"

    if force_same_samples and not indices_file.exists():
        if base_indices_file.exists():
            indices_file.write_text(base_indices_file.read_text(encoding="utf-8"), encoding="utf-8")
        elif base_split_dir.exists():
            sample_ids = []
            for p in sorted(base_split_dir.glob("sample_*")):
                m = re.match(r"^sample_(\d+)$", p.name)
                if m:
                    sample_ids.append(int(m.group(1)))
            if sample_ids:
                payload = {"indices": sample_ids, "source": str(base_split_dir)}
                indices_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "legacy/streaming_codec_baseline/scripts/sample_checkpoint_compare.py"),
        "--cache",
        str(cache),
        "--manifest",
        str(manifest),
        "--checkpoints",
        *[str(x) for x in ckpts],
        "--indices-file",
        str(indices_file),
        "--random-sample",
        "--sample-count",
        str(int(sample_count)),
        "--seed",
        str(int(seed)),
        "--out_dir",
        str(out_dir),
        "--title",
        f"{run_name} checkpoint qc [{split}]",
    ]
    if dump_logits:
        cmd.extend(["--dump-logits", "--logits-topk", str(int(logits_topk))])
    subprocess.run(cmd, check=True, cwd=repo)

    md_path = out_dir / "comparison.md"
    subprocess.run(
        [
            sys.executable,
            str(repo / "legacy/streaming_codec_baseline/scripts/comparison_md_to_ipynb.py"),
            "--md",
            str(md_path),
        ],
        check=True,
        cwd=repo,
    )
    return out_dir


def display_split_html(run_name: str, split_or_subdir: str = "valid") -> None:
    repo = find_repo_root()
    html_path = (
        repo
        / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split_or_subdir}/comparison.html"
    )
    if not html_path.exists():
        raise FileNotFoundError(f"comparison.html not found: {html_path}")
    display(HTML(html_path.read_text(encoding="utf-8")))


def run_teacher_forced_compare(
    run_name: str,
    split: str,
    sample_count: int = 10,
    seed: int = 42,
    checkpoints: list[Path] | None = None,
    out_subdir: str | None = None,
    force_same_samples: bool = True,
) -> Path:
    repo = find_repo_root()
    cache = repo / f"legacy/streaming_codec_baseline/artifacts/data/mixed_4ds_ace_8cq_{split}_codes_sharded"
    manifest = repo / f"legacy/streaming_codec_baseline/artifacts/data/mixed_4ds_ace_8cq_{split}_manifest.json"
    ckpt_dir = repo / f"legacy/streaming_codec_baseline/artifacts/checkpoints/{run_name}"
    ckpts = checkpoints or sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    target_subdir = out_subdir or f"{split}_teacher_forced"
    out_dir = repo / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{target_subdir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    indices_file = out_dir / "sample_indices.json"
    base_split_dir = repo / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split}"
    base_indices_file = base_split_dir / "sample_indices.json"

    if force_same_samples and not indices_file.exists():
        if base_indices_file.exists():
            indices_file.write_text(base_indices_file.read_text(encoding="utf-8"), encoding="utf-8")
        elif base_split_dir.exists():
            sample_ids = []
            for p in sorted(base_split_dir.glob("sample_*")):
                m = re.match(r"^sample_(\d+)$", p.name)
                if m:
                    sample_ids.append(int(m.group(1)))
            if sample_ids:
                payload = {"indices": sample_ids, "source": str(base_split_dir)}
                indices_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "legacy/streaming_codec_baseline/scripts/teacher_forced_recon_compare.py"),
        "--cache",
        str(cache),
        "--manifest",
        str(manifest),
        "--checkpoints",
        *[str(x) for x in ckpts],
        "--indices-file",
        str(indices_file),
        "--random-sample",
        "--sample-count",
        str(int(sample_count)),
        "--seed",
        str(int(seed)),
        "--out_dir",
        str(out_dir),
        "--title",
        f"{run_name} teacher-forced qc [{split}]",
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    return out_dir


def display_teacher_forced_html(run_name: str, split_or_subdir: str = "valid_teacher_forced") -> None:
    repo = find_repo_root()
    html_path = (
        repo
        / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split_or_subdir}/comparison.html"
    )
    if not html_path.exists():
        raise FileNotFoundError(f"teacher-forced comparison.html not found: {html_path}")
    display(HTML(html_path.read_text(encoding="utf-8")))


def _logits_root(repo: Path, run_name: str, split_or_subdir: str) -> Path:
    return repo / f"legacy/streaming_codec_baseline/artifacts/outputs/{run_name}/checkpoint_qc/{split_or_subdir}"


def list_logits_choices(run_name: str, split_or_subdir: str = "valid") -> tuple[list[str], dict[str, list[str]]]:
    repo = find_repo_root()
    root = _logits_root(repo, run_name, split_or_subdir)
    sample_ids: list[str] = []
    ckpts_by_sample: dict[str, list[str]] = {}
    for sample_dir in sorted(root.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        tags = sorted(p.name[: -len("_logits_debug.npz")] for p in sample_dir.glob("*_logits_debug.npz"))
        if not tags:
            continue
        sample_ids.append(sample_dir.name)
        ckpts_by_sample[sample_dir.name] = tags
    return sample_ids, ckpts_by_sample


def render_logits_debug(
    run_name: str,
    split_or_subdir: str = "valid",
    sample_id: str | None = None,
    ckpt_tags: list[str] | None = None,
    step_t: int | None = None,
    step_range: tuple[int, int] | None = None,
    layer_indices: list[int] | None = None,
) -> None:
    repo = find_repo_root()
    root = _logits_root(repo, run_name, split_or_subdir)
    sample_ids, ckpts_by_sample = list_logits_choices(run_name, split_or_subdir)
    if not sample_ids:
        raise FileNotFoundError(f"No logits debug artifacts found under {root}")

    sample_id = sample_id or sample_ids[0]
    if sample_id not in ckpts_by_sample:
        raise FileNotFoundError(f"sample_id not found or has no logits artifacts: {sample_id}")

    available_ckpts = ckpts_by_sample[sample_id]
    selected_ckpts = ckpt_tags or available_ckpts
    selected_ckpts = [x for x in selected_ckpts if x in available_ckpts]
    if not selected_ckpts:
        raise ValueError(f"No selected ckpts available for {sample_id}. available={available_ckpts}")

    sample_dir = root / sample_id
    summary_rows: list[dict] = []
    print(f"split: {split_or_subdir}")
    print(f"sample_id: {sample_id}")
    print("available samples:")
    print(", ".join(sample_ids))
    print("available ckpts:")
    print(", ".join(available_ckpts))
    print("selected ckpts:")
    print(", ".join(selected_ckpts))
    sample_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "available_ckpts": [", ".join(ckpts_by_sample[sid]) for sid in sample_ids],
        }
    )
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(sample_df.to_string(index=False))

    audio_items = []
    for label, fname in [("gt", "gt.wav"), ("codec", "codec_recon_k8.wav")]:
        path = sample_dir / fname
        if path.exists():
            audio_items.append((label, path))
    for ckpt in selected_ckpts:
        path = sample_dir / f"{ckpt}.wav"
        if path.exists():
            audio_items.append((ckpt, path))

    for label, path in audio_items:
        print(label, path.name)
        display(Audio(filename=str(path)))

    for ckpt in selected_ckpts:
        npz_path = sample_dir / f"{ckpt}_logits_debug.npz"
        d = np.load(npz_path, allow_pickle=True)
        pred = d["pred_tokens"]
        gt = d["gt_tokens"]
        gt_rank = d["gt_rank"]
        gt_logit = d["gt_logit"]
        topk_idx = d["topk_idx"]
        topk_val = d["topk_val"]

        mismatch = pred != gt
        step_err = mismatch.mean(axis=(1, 2))
        step_rank = gt_rank.mean(axis=(1, 2))
        T, S, K = mismatch.shape
        selected_layers = layer_indices or list(range(K))
        selected_layers = [int(k) for k in selected_layers if 0 <= int(k) < K]
        if not selected_layers:
            raise ValueError(f"No valid layer_indices selected. requested={layer_indices}, available=0..{K-1}")
        step_df = pd.DataFrame(
            {
                "t": np.arange(len(step_err)),
                "error_rate": step_err,
                "mean_gt_rank": step_rank,
            }
        ).sort_values(["error_rate", "mean_gt_rank"], ascending=[False, False])

        chosen_t = int(step_df.iloc[0]["t"] if step_t is None else step_t)
        summary_rows.append(
            {
                "ckpt": ckpt,
                "mismatch_tokens": int(mismatch.sum()),
                "total_tokens": int(mismatch.size),
                "mismatch_rate": float(mismatch.mean()),
                "worst_t": chosen_t,
                "worst_t_error_rate": float(step_err[chosen_t]),
                "worst_t_mean_gt_rank": float(step_rank[chosen_t]),
            }
        )

        print(f"\n[{ckpt}] total steps: {T}")
        print(f"[{ckpt}] total slots per step: {S}")
        print(f"[{ckpt}] total layers/codebooks: {K}")
        print(f"[{ckpt}] selected layers: {selected_layers}")

        full_step_df = pd.DataFrame(
            {
                "t": np.arange(len(step_err)),
                "error_rate": step_err,
                "mean_gt_rank": step_rank,
            }
        )
        if step_range is None:
            shown_step_df = full_step_df
            print(f"\n[{ckpt}] all steps")
        else:
            start_t, end_t = step_range
            shown_step_df = full_step_df[(full_step_df["t"] >= int(start_t)) & (full_step_df["t"] <= int(end_t))]
            print(f"\n[{ckpt}] steps {int(start_t)}..{int(end_t)}")
        with pd.option_context("display.max_colwidth", None, "display.width", 200):
            print(shown_step_df.to_string(index=False))

        if step_range is None:
            detail_steps = list(range(T))
            print(f"[{ckpt}] detail at all steps")
        else:
            start_t, end_t = step_range
            if start_t > end_t:
                raise ValueError(f"invalid step_range: {step_range}")
            detail_steps = [t for t in range(int(start_t), int(end_t) + 1) if 0 <= t < T]
            print(f"[{ckpt}] detail at steps {int(start_t)}..{int(end_t)}")
        if not detail_steps:
            raise ValueError(f"no valid steps selected from step_range={step_range}, T={T}")
        rows = []
        for t in detail_steps:
            for s in range(S):
                for k in selected_layers:
                    rows.append(
                        {
                            "t": t,
                            "s": s,
                            "k": k,
                            "pred": int(pred[t, s, k]),
                            "gt": int(gt[t, s, k]),
                            "is_correct": bool(pred[t, s, k] == gt[t, s, k]),
                            "gt_rank": int(gt_rank[t, s, k]),
                            "gt_logit": float(gt_logit[t, s, k]),
                            "topk_tokens": [int(x) for x in topk_idx[t, s, k].tolist()],
                            "topk_logits": [float(x) for x in topk_val[t, s, k].tolist()],
                        }
                    )
        detail_df = pd.DataFrame(rows).sort_values(
            ["t", "is_correct", "gt_rank", "s", "k"], ascending=[True, True, False, True, True]
        )
        with pd.option_context("display.max_colwidth", None, "display.width", 400):
            print(detail_df.to_string(index=False))

        plot_steps = np.array(detail_steps, dtype=int)
        gt_rank_mean_by_tk = gt_rank.mean(axis=1)
        gt_rank_slot_by_tk = gt_rank.transpose(2, 1, 0)
        fig, axes = plt.subplots(len(selected_layers), 2, figsize=(14, max(4, 4 * len(selected_layers))), squeeze=False)
        for row_idx, k in enumerate(selected_layers):
            ax_line = axes[row_idx][0]
            ax_heat = axes[row_idx][1]
            ax_line.plot(plot_steps, gt_rank_mean_by_tk[plot_steps, k], marker="o", linewidth=1.2)
            ax_line.set_title(f"{ckpt} layer={k} mean GT rank by step")
            ax_line.set_xlabel("step t")
            ax_line.set_ylabel("mean gt_rank over slots")
            ax_line.grid(True, alpha=0.3)

            heat = gt_rank_slot_by_tk[k, :, :][:, plot_steps]
            im = ax_heat.imshow(heat, aspect="auto", origin="lower")
            ax_heat.set_title(f"{ckpt} layer={k} GT rank heatmap")
            ax_heat.set_xlabel("selected step index")
            ax_heat.set_ylabel("slot s")
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()

    print("\n[summary across selected ckpts]")
    summary_df = pd.DataFrame(summary_rows).sort_values("mismatch_rate", ascending=False)
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(summary_df.to_string(index=False))
