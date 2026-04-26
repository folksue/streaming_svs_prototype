from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass
class PairItem:
    utt_id: str
    ref_wav: str
    pred_wav: str


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=True)
    x = torch.from_numpy(wav).float().transpose(0, 1)
    if x.size(0) > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != target_sr:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=target_sr)
    return x


def _align_pair(ref: torch.Tensor, pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(ref.size(-1), pred.size(-1))
    return ref[..., :n], pred[..., :n]


def _extract_f0(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    f0_min: float,
    f0_max: float,
) -> torch.Tensor:
    frame_time = float(hop_length) / float(sample_rate)
    win_length = 3
    f0 = torchaudio.functional.detect_pitch_frequency(
        wav,
        sample_rate=sample_rate,
        frame_time=frame_time,
        win_length=win_length,
        freq_low=f0_min,
        freq_high=f0_max,
    )  # [C, T]
    return f0[0]


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    xm = x - x.mean()
    ym = y - y.mean()
    denom = torch.sqrt((xm.pow(2).sum() + 1e-8) * (ym.pow(2).sum() + 1e-8))
    return float((xm * ym).sum().item() / float(denom.item()))


def _f0_metrics(
    ref_wav: torch.Tensor,
    pred_wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    f0_min: float,
    f0_max: float,
) -> Dict[str, float]:
    ref_f0 = _extract_f0(ref_wav, sample_rate, hop_length, f0_min, f0_max)
    pred_f0 = _extract_f0(pred_wav, sample_rate, hop_length, f0_min, f0_max)
    t = min(ref_f0.numel(), pred_f0.numel())
    if t == 0:
        return {
            "f0_rmse_hz": float("nan"),
            "f0_cents_rmse": float("nan"),
            "f0_corr": float("nan"),
            "vuv_acc": float("nan"),
            "ffe_proxy": float("nan"),
            "voiced_ref_ratio": float("nan"),
            "voiced_pred_ratio": float("nan"),
        }

    ref_f0 = ref_f0[:t]
    pred_f0 = pred_f0[:t]
    ref_v = ref_f0 > 1.0
    pred_v = pred_f0 > 1.0
    uv_match = (ref_v == pred_v).float().mean().item()

    both_voiced = ref_v & pred_v
    if both_voiced.any():
        rf = ref_f0[both_voiced]
        pf = pred_f0[both_voiced]
        rmse_hz = torch.sqrt(torch.mean((rf - pf).pow(2))).item()
        cents = 1200.0 * torch.log2((pf + 1e-6) / (rf + 1e-6))
        cents_rmse = torch.sqrt(torch.mean(cents.pow(2))).item()
        corr = _safe_pearson(torch.log(rf + 1e-6), torch.log(pf + 1e-6))
        gross_pitch = (torch.abs(cents) > 50.0).float()
        ffe = ((~both_voiced).float().mean() + gross_pitch.mean()).item() * 0.5
    else:
        rmse_hz = float("nan")
        cents_rmse = float("nan")
        corr = float("nan")
        ffe = float("nan")

    return {
        "f0_rmse_hz": float(rmse_hz),
        "f0_cents_rmse": float(cents_rmse),
        "f0_corr": float(corr),
        "vuv_acc": float(uv_match),
        "ffe_proxy": float(ffe),
        "voiced_ref_ratio": float(ref_v.float().mean().item()),
        "voiced_pred_ratio": float(pred_v.float().mean().item()),
    }


def _spectral_metrics(
    ref_wav: torch.Tensor,
    pred_wav: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_mfcc: int,
) -> Dict[str, float]:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=40.0,
        f_max=sample_rate / 2.0,
        power=2.0,
    )
    ref_mel = mel(ref_wav).clamp_min(1e-8).log10()
    pred_mel = mel(pred_wav).clamp_min(1e-8).log10()
    t = min(ref_mel.size(-1), pred_mel.size(-1))
    ref_mel = ref_mel[..., :t]
    pred_mel = pred_mel[..., :t]

    l1_logmel = torch.mean(torch.abs(ref_mel - pred_mel)).item()
    l2_logmel = torch.sqrt(torch.mean((ref_mel - pred_mel).pow(2))).item()

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": max(40, n_mels),
            "f_min": 40.0,
            "f_max": sample_rate / 2.0,
            "power": 2.0,
        },
    )
    ref_mfcc = mfcc(ref_wav)[..., :t]
    pred_mfcc = mfcc(pred_wav)[..., :t]
    # MCD-like proxy (without DTW).
    mcd_like = torch.mean(torch.sqrt(torch.sum((ref_mfcc - pred_mfcc).pow(2), dim=1))).item()

    return {
        "logmel_l1": float(l1_logmel),
        "logmel_l2": float(l2_logmel),
        "mcd_like": float(mcd_like),
    }


def compute_pair_metrics(
    ref_path: str,
    pred_path: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_mfcc: int,
    f0_min: float,
    f0_max: float,
) -> Dict[str, float]:
    ref_wav = _load_wav(ref_path, target_sr=sample_rate)
    pred_wav = _load_wav(pred_path, target_sr=sample_rate)
    ref_wav, pred_wav = _align_pair(ref_wav, pred_wav)

    out = {}
    out.update(
        _f0_metrics(
            ref_wav=ref_wav,
            pred_wav=pred_wav,
            sample_rate=sample_rate,
            hop_length=hop_length,
            f0_min=f0_min,
            f0_max=f0_max,
        )
    )
    out.update(
        _spectral_metrics(
            ref_wav=ref_wav,
            pred_wav=pred_wav,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )
    )
    return out


def load_pairs_from_json(path: str) -> List[PairItem]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("pairs json must be a list")
    pairs: List[PairItem] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"pairs[{i}] must be dict")
        utt_id = str(item.get("utt_id", f"item_{i:06d}"))
        ref_wav = str(item["ref_wav"])
        pred_wav = str(item["pred_wav"])
        pairs.append(PairItem(utt_id=utt_id, ref_wav=ref_wav, pred_wav=pred_wav))
    return pairs


def load_pairs_from_dirs(ref_dir: str, pred_dir: str) -> List[PairItem]:
    ref_map = {
        os.path.splitext(x)[0]: os.path.join(ref_dir, x)
        for x in os.listdir(ref_dir)
        if x.lower().endswith(".wav")
    }
    pred_map = {
        os.path.splitext(x)[0]: os.path.join(pred_dir, x)
        for x in os.listdir(pred_dir)
        if x.lower().endswith(".wav")
    }
    common = sorted(set(ref_map.keys()) & set(pred_map.keys()))
    if not common:
        raise RuntimeError("No matched wav basename between ref_dir and pred_dir")
    return [PairItem(utt_id=k, ref_wav=ref_map[k], pred_wav=pred_map[k]) for k in common]


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for row in rows for k in row.keys() if k not in {"utt_id"}})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and not math.isnan(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_json", type=str, default=None, help="List of {utt_id, ref_wav, pred_wav}")
    p.add_argument("--ref_dir", type=str, default=None, help="Reference wav dir (basename match)")
    p.add_argument("--pred_dir", type=str, default=None, help="Prediction wav dir (basename match)")
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=240)
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--n_mfcc", type=int, default=13)
    p.add_argument("--f0_min", type=float, default=50.0)
    p.add_argument("--f0_max", type=float, default=1100.0)
    p.add_argument("--out_json", type=str, default="artifacts/eval/audio_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs_json:
        pairs = load_pairs_from_json(args.pairs_json)
    else:
        if not args.ref_dir or not args.pred_dir:
            raise ValueError("Either --pairs_json or both --ref_dir/--pred_dir must be provided.")
        pairs = load_pairs_from_dirs(args.ref_dir, args.pred_dir)

    rows: List[Dict[str, float]] = []
    for item in pairs:
        metrics = compute_pair_metrics(
            ref_path=item.ref_wav,
            pred_path=item.pred_wav,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
            f0_min=args.f0_min,
            f0_max=args.f0_max,
        )
        rows.append({"utt_id": item.utt_id, **metrics})

    summary = aggregate_metrics(rows)
    payload = {
        "num_pairs": len(rows),
        "summary": summary,
        "items": rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {args.out_json}")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}" if not math.isnan(v) else f"{k}: nan")


if __name__ == "__main__":
    main()

