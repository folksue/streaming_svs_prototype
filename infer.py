from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import soundfile as sf
import torch

from dataset import SVSSequenceDataset
from model import StreamingSVSModel
from utils import get_device, masked_code_accuracy


class EncodecDecoder:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import EncodecModel  # type: ignore

        self.device = device
        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        audio_codes = codes.transpose(0, 1).unsqueeze(0).to(self.device)
        audio_scales = [None]

        try:
            decoded = self.model.decode(audio_codes=audio_codes, audio_scales=audio_scales)
            wav = decoded.audio_values
        except Exception:
            wav = self.model.decode(audio_codes=audio_codes, audio_scales=audio_scales)[0]
        return wav.squeeze().detach().cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--decode_wav", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def build_model(ckpt: Dict, dev: torch.device) -> StreamingSVSModel:
    cfg = ckpt["config"]["model"]
    meta = ckpt["meta"]

    model = StreamingSVSModel(
        num_codebooks=int(meta["num_codebooks"]),
        frames_per_chunk=int(meta["frames_per_chunk"]),
        codebook_size=int(meta["codebook_size"]),
        phoneme_vocab_size=cfg["phoneme_vocab_size"],
        phoneme_emb_dim=cfg["phoneme_emb_dim"],
        pitch_proj_dim=cfg["pitch_proj_dim"],
        time_proj_dim=cfg["time_proj_dim"],
        cond_dim=cfg["cond_dim"],
        token_emb_dim=cfg.get("token_emb_dim", 128),
        prev_chunk_dim=cfg.get("prev_chunk_dim", 256),
        model_dim=cfg["model_dim"],
        num_blocks=cfg["num_blocks"],
    ).to(dev)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    dev = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=dev)
    model = build_model(ckpt, dev)

    ds = SVSSequenceDataset(args.cache)
    sample = ds[args.index]

    codes_gt = sample.codes.unsqueeze(0).to(dev)
    phoneme = sample.phoneme_id.unsqueeze(0).to(dev)
    cond = sample.cond_num.unsqueeze(0).to(dev)
    on_b = sample.note_on_boundary.unsqueeze(0).to(dev)
    off_b = sample.note_off_boundary.unsqueeze(0).to(dev)
    mask = torch.ones(1, codes_gt.size(1), device=dev)

    with torch.no_grad():
        codes_pred = model.generate(phoneme, cond, temperature=args.temperature)

    acc = masked_code_accuracy(codes_pred, codes_gt, mask)
    on_acc = masked_code_accuracy(codes_pred, codes_gt, on_b)
    off_acc = masked_code_accuracy(codes_pred, codes_gt, off_b)

    print(
        f"utt={sample.utt_id} token_acc={acc.item():.6f} "
        f"on_boundary_acc={on_acc.item():.6f} off_boundary_acc={off_acc.item():.6f}"
    )

    pred_np = codes_pred.squeeze(0).cpu().numpy().astype(np.int64)
    gt_np = codes_gt.squeeze(0).cpu().numpy().astype(np.int64)

    np.savez(
        os.path.join(args.out_dir, f"sample_{args.index:04d}.npz"),
        utt_id=sample.utt_id,
        codes_pred=pred_np,
        codes_gt=gt_np,
    )

    if args.decode_wav:
        decoder = EncodecDecoder(
            model_name=ckpt["config"]["audio"]["encodec_model_name"],
            device=dev,
        )
        code_frames = codes_pred.squeeze(0).reshape(-1, codes_pred.size(-1))
        wav = decoder.decode_from_codes(code_frames)
        out_wav = os.path.join(args.out_dir, f"sample_{args.index:04d}.wav")
        sf.write(out_wav, wav.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))
        print(f"saved wav: {out_wav}")


if __name__ == "__main__":
    main()
