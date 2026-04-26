from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import SVSSequenceDataset
from infer import EncodecDecoder, build_model
from utils import get_device, masked_code_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect autoregressive logits vs ground truth for one sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--decode_wav", action="store_true")
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


def build_html(summary: dict, rows: list[dict], pred_wav: str | None, gt_wav: str | None) -> str:
    audio_html = ""
    if pred_wav:
        audio_html += (
            "<div class='audio-block'><div class='audio-title'>Predicted wav</div>"
            f"<audio controls preload='none' src='{html_escape(encode_uri(pred_wav))}'></audio></div>"
        )
    if gt_wav:
        audio_html += (
            "<div class='audio-block'><div class='audio-title'>Ground truth wav</div>"
            f"<audio controls preload='none' src='{html_escape(encode_uri(gt_wav))}'></audio></div>"
        )

    row_html = []
    for row in rows:
        topk_html = "<br>".join(
            f"{int(item['token'])}: {item['prob']:.4f}" for item in row["topk"]
        )
        row_html.append(
            "<tr>"
            f"<td>{row['step']}</td>"
            f"<td>{row['slot']}</td>"
            f"<td>{row['codebook']}</td>"
            f"<td>{row['pred_token']}</td>"
            f"<td>{row['gt_token']}</td>"
            f"<td class='match'>{'yes' if row['match'] else 'no'}</td>"
            f"<td>{row['pred_prob']:.6f}</td>"
            f"<td>{row['true_prob']:.6f}</td>"
            f"<td>{row['true_rank']}</td>"
            f"<td>{row['entropy']:.4f}</td>"
            f"<td>{row['top1_margin']:.4f}</td>"
            f"<td class='topk'>{topk_html}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Logits diagnostic - {html_escape(summary['utt_id'])}</title>
  <style>
    body {{ margin: 0; background: #f4efe6; color: #211d16; font-family: ui-serif, Georgia, serif; }}
    .wrap {{ padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta {{ color: #675f52; margin-bottom: 14px; }}
    .summary {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .card {{ background: #fffdf8; border: 1px solid #d9cfb8; padding: 10px 12px; }}
    .label {{ font-size: 12px; color: #7d7361; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 18px; margin-top: 4px; }}
    .audio-row {{ display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }}
    .audio-block {{ background: #fffdf8; border: 1px solid #d9cfb8; padding: 10px 12px; min-width: 320px; }}
    .audio-title {{ margin-bottom: 8px; color: #5c5245; }}
    audio {{ width: 100%; }}
    .table-wrap {{ overflow: auto; border: 1px solid #d9cfb8; background: #fffdf8; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1400px; }}
    th, td {{ border: 1px solid #d9cfb8; padding: 8px 10px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #eee3ca; text-align: left; }}
    td.match {{ font-weight: 700; color: #8b3a2d; }}
    td.topk {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AR logits diagnostic</h1>
    <div class="meta">{html_escape(summary['utt_id'])} | sample index {summary['index']} | checkpoint {html_escape(summary['checkpoint_label'])}</div>
    <div class="summary">
      <div class="card"><div class="label">AR Acc</div><div class="value">{summary['ar_acc']:.6f}</div></div>
      <div class="card"><div class="label">Coarse AR Acc</div><div class="value">{summary['coarse_ar_acc']:.6f}</div></div>
      <div class="card"><div class="label">Fine AR Acc</div><div class="value">{summary['fine_ar_acc']:.6f}</div></div>
      <div class="card"><div class="label">Steps</div><div class="value">{summary['num_steps']}</div></div>
      <div class="card"><div class="label">Codebooks</div><div class="value">{summary['num_codebooks']}</div></div>
      <div class="card"><div class="label">Rows</div><div class="value">{len(rows)}</div></div>
    </div>
    <div class="audio-row">{audio_html}</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>step</th>
            <th>slot</th>
            <th>codebook</th>
            <th>pred</th>
            <th>gt</th>
            <th>match</th>
            <th>pred_prob</th>
            <th>true_prob</th>
            <th>true_rank</th>
            <th>entropy</th>
            <th>top1_margin</th>
            <th>topk probs</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>"""


def build_notebook(title: str, html: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "这个 notebook 已经预渲染，不需要手动运行。\n"],
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
                            "text/plain": [f"<rendered logits diagnostic: {title}>"],
                        },
                        "metadata": {},
                    }
                ],
                "source": ["from IPython.display import HTML\n", "HTML('<logits diagnostic rendered in output>')\n"],
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


@torch.no_grad()
def collect_rows(model, sample, device: torch.device, topk: int) -> tuple[list[dict], torch.Tensor]:
    note_id = sample.note_id.unsqueeze(0).to(device)
    phoneme_id = sample.phoneme_id.unsqueeze(0).to(device)
    slur = sample.slur.unsqueeze(0).to(device)
    phone_progress = sample.phone_progress.unsqueeze(0).to(device)
    codes_gt = sample.codes.unsqueeze(0).to(device)

    b, t = note_id.shape
    cond = model.cond_enc(note_id, phoneme_id, slur, phone_progress)
    kv_cache = [(None, None) for _ in range(len(model.blocks))]
    pos = 0
    prev_frame_codes = torch.full(
        (b, model.num_codebooks),
        fill_value=model.audio_bos_id,
        dtype=torch.long,
        device=device,
    )

    if model.prefix_pad_steps > 0:
        pad_control = model.prefix_control_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
        pad_frame = model.prefix_frame_pad.to(cond.dtype).view(1, 1, -1).expand(b, 1, -1)
        for _ in range(model.prefix_pad_steps):
            control_in = pad_control + model.token_type_emb.weight[0].view(1, 1, -1) + model.within_step_pos_emb[0].to(cond.dtype).view(1, 1, -1)
            _, kv_cache = model._forward_one_token_incremental(control_in, pos, kv_cache)
            pos += 1
            for frame_slot in range(model.audio_slots_per_step):
                frame_in = pad_frame + model.token_type_emb.weight[1].view(1, 1, -1) + model.within_step_pos_emb[1 + frame_slot].to(cond.dtype).view(1, 1, -1)
                _, kv_cache = model._forward_one_token_incremental(frame_in, pos, kv_cache)
                pos += 1

    pred_steps = []
    rows: list[dict] = []

    for step_idx in range(t):
        step_tokens = []
        control_tok = model._encode_control(cond[:, step_idx : step_idx + 1, :])
        control_in = control_tok + model.token_type_emb.weight[0].view(1, 1, -1) + model.within_step_pos_emb[0].to(cond.dtype).view(1, 1, -1)
        _, kv_cache = model._forward_one_token_incremental(control_in, pos, kv_cache)
        pos += 1

        for frame_idx in range(model.audio_slots_per_step):
            frame_codes_in = prev_frame_codes.view(b, 1, 1, model.num_codebooks)
            frame_tok = model.audio_in(model._embed_frame_tokens(frame_codes_in)).view(b, 1, -1)
            frame_in = frame_tok + model.token_type_emb.weight[1].view(1, 1, -1) + model.within_step_pos_emb[1 + frame_idx].to(cond.dtype).view(1, 1, -1)
            frame_hidden, kv_cache = model._forward_one_token_incremental(frame_in, pos, kv_cache)
            pos += 1

            frame_hidden_flat = frame_hidden[:, 0, :]
            logits_per_codebook = [head(frame_hidden_flat) for head in model.audio_heads]
            next_logits = torch.stack(logits_per_codebook, dim=1)  # [B,K,V]
            next_tokens = model._sample_next(next_logits, temperature=0.0).long()
            gt_tokens = codes_gt[:, step_idx, frame_idx, :]

            probs = torch.softmax(next_logits[0], dim=-1)  # [K,V]
            top_vals, top_idx = torch.topk(probs, k=min(topk, probs.size(-1)), dim=-1)
            entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)

            for codebook_idx in range(model.num_codebooks):
                gt_token = int(gt_tokens[0, codebook_idx].item())
                pred_token = int(next_tokens[0, codebook_idx].item())
                prob_vec = probs[codebook_idx]
                true_prob = float(prob_vec[gt_token].item())
                pred_prob = float(prob_vec[pred_token].item())
                true_rank = int((prob_vec > prob_vec[gt_token]).sum().item()) + 1
                top1_margin = float(top_vals[codebook_idx, 0].item() - top_vals[codebook_idx, 1].item()) if top_vals.size(1) > 1 else 0.0
                rows.append(
                    {
                        "step": step_idx,
                        "slot": frame_idx,
                        "codebook": codebook_idx,
                        "pred_token": pred_token,
                        "gt_token": gt_token,
                        "match": pred_token == gt_token,
                        "pred_prob": pred_prob,
                        "true_prob": true_prob,
                        "true_rank": true_rank,
                        "entropy": float(entropy[codebook_idx].item()),
                        "top1_margin": top1_margin,
                        "topk": [
                            {"token": int(tok.item()), "prob": float(val.item())}
                            for tok, val in zip(top_idx[codebook_idx], top_vals[codebook_idx])
                        ],
                    }
                )

            prev_frame_codes = next_tokens
            step_tokens.append(next_tokens)

        pred_steps.append(torch.stack(step_tokens, dim=1))

    codes_pred = torch.cat(pred_steps, dim=1)
    return rows, codes_pred


def save_json(path: Path, summary: dict, rows: list[dict]) -> None:
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt, device)
    dataset = SVSSequenceDataset(args.cache)
    sample = dataset[args.index]

    rows, codes_pred = collect_rows(model, sample, device, topk=args.topk)

    codes_gt = sample.codes.unsqueeze(0).to(device)
    mask = torch.ones(1, codes_gt.size(1), device=device)
    ar_acc = float(masked_code_accuracy(codes_pred, codes_gt, mask).item())
    coarse_ar_acc = float(masked_code_accuracy(codes_pred[..., 0], codes_gt[..., 0], mask).item())
    if codes_gt.size(-1) > 1:
        fine_vals = [
            masked_code_accuracy(codes_pred[..., k], codes_gt[..., k], mask)
            for k in range(1, codes_gt.size(-1))
        ]
        fine_ar_acc = float(torch.stack(fine_vals).mean().item())
    else:
        fine_ar_acc = coarse_ar_acc

    checkpoint_label = Path(args.checkpoint).stem
    summary = {
        "utt_id": sample.utt_id,
        "index": args.index,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_label": checkpoint_label,
        "num_steps": int(codes_gt.size(1)),
        "num_codebooks": int(codes_gt.size(-1)),
        "ar_acc": ar_acc,
        "coarse_ar_acc": coarse_ar_acc,
        "fine_ar_acc": fine_ar_acc,
    }

    pred_wav = None
    gt_wav = None
    if args.decode_wav:
        decoder = EncodecDecoder(
            model_name=ckpt["config"]["audio"]["encodec_model_name"],
            device=device,
        )
        pred_flat = codes_pred.squeeze(0).reshape(-1, codes_pred.size(-1))
        pred_audio = decoder.decode_from_codes(pred_flat)
        pred_wav_path = out_dir / "pred.wav"
        sf.write(pred_wav_path, pred_audio.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))
        pred_wav = str(pred_wav_path.relative_to(out_dir))

        gt_flat = codes_gt.squeeze(0).reshape(-1, codes_gt.size(-1))
        gt_audio = decoder.decode_from_codes(gt_flat)
        gt_wav_path = out_dir / "gt.wav"
        sf.write(gt_wav_path, gt_audio.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))
        gt_wav = str(gt_wav_path.relative_to(out_dir))

    html = build_html(summary, rows, pred_wav=pred_wav, gt_wav=gt_wav)
    html_path = out_dir / "diagnostic.html"
    html_path.write_text(html, encoding="utf-8")

    nb = build_notebook(f"logits diagnostic - {sample.utt_id}", html)
    ipynb_path = out_dir / "diagnostic.ipynb"
    ipynb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")

    save_json(out_dir / "diagnostic.json", summary, rows)
    np.savez(
        out_dir / "diagnostic.npz",
        codes_pred=codes_pred.squeeze(0).cpu().numpy().astype(np.int64),
        codes_gt=codes_gt.squeeze(0).cpu().numpy().astype(np.int64),
    )
    print(html_path)
    print(ipynb_path)


if __name__ == "__main__":
    main()
