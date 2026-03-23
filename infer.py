from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import soundfile as sf
import torch

from dataset import SVSSequenceDataset
from model import StreamingSVSModel
from utils import boundary_error, continuity_metric, get_device, masked_l1


class EncodecDecoder:
    def __init__(self, model_name: str, device: torch.device, codebook_size: int = 1024):
        from transformers import EncodecModel  # type: ignore

        self.device = device
        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.codebook_size = codebook_size

    @torch.inference_mode()
    def decode_from_continuous_codes(self, latent_frames: torch.Tensor) -> torch.Tensor:
        """
        latent_frames: [T, D] in [-1,1], where D is treated as num_codebooks.

        We quantize back to integer EnCodec codes for waveform decoding.
        """
        x = ((latent_frames + 1.0) * 0.5 * (self.codebook_size - 1)).round().long()
        x = x.clamp(0, self.codebook_size - 1)

        # -> [num_codebooks, batch, frames]
        audio_codes = x.transpose(0, 1).unsqueeze(1).to(self.device)

        try:
            decoded = self.model.decode(audio_codes=audio_codes, audio_scales=None)
            wav = decoded.audio_values
        except Exception:
            # API fallback for different transformers versions
            wav = self.model.decode(audio_codes)
        return wav.squeeze().detach().cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--decode_wav", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def build_model(ckpt: Dict, dev: torch.device) -> StreamingSVSModel:
    cfg = ckpt["config"]["model"]
    meta = ckpt["meta"]

    model = StreamingSVSModel(
        latent_dim=int(meta["latent_dim"]),
        frames_per_chunk=int(meta["frames_per_chunk"]),
        phoneme_vocab_size=cfg["phoneme_vocab_size"],
        phoneme_emb_dim=cfg["phoneme_emb_dim"],
        pitch_proj_dim=cfg["pitch_proj_dim"],
        time_proj_dim=cfg["time_proj_dim"],
        cond_dim=cfg["cond_dim"],
        ctx_hidden=cfg["ctx_hidden"],
        ctx_layers=cfg["ctx_layers"],
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

    z_gt = sample.z.unsqueeze(0).to(dev)
    phoneme = sample.phoneme_id.unsqueeze(0).to(dev)
    cond = sample.cond_num.unsqueeze(0).to(dev)
    on_b = sample.note_on_boundary.unsqueeze(0).to(dev)
    off_b = sample.note_off_boundary.unsqueeze(0).to(dev)
    mask = torch.ones(1, z_gt.size(1), device=dev)

    with torch.no_grad():
        z_pred = model.generate(phoneme, cond, temperature=args.temperature)

    l1 = masked_l1(z_pred, z_gt, mask)
    cont = continuity_metric(z_pred, mask)
    on_err = boundary_error(z_pred, z_gt, on_b)
    off_err = boundary_error(z_pred, z_gt, off_b)

    print(
        f"utt={sample.utt_id} latent_l1={l1.item():.6f} continuity={cont.item():.6f} "
        f"on_boundary={on_err.item():.6f} off_boundary={off_err.item():.6f}"
    )

    pred_np = z_pred.squeeze(0).cpu().numpy().astype(np.float32)
    gt_np = z_gt.squeeze(0).cpu().numpy().astype(np.float32)

    np.savez(
        os.path.join(args.out_dir, f"sample_{args.index:04d}.npz"),
        utt_id=sample.utt_id,
        z_pred=pred_np,
        z_gt=gt_np,
    )

    if args.decode_wav:
        decoder = EncodecDecoder(
            model_name=ckpt["config"]["audio"]["encodec_model_name"],
            device=dev,
        )
        z_frames = z_pred.squeeze(0).reshape(-1, z_pred.size(-1))
        wav = decoder.decode_from_continuous_codes(z_frames)
        out_wav = os.path.join(args.out_dir, f"sample_{args.index:04d}.wav")
        sf.write(out_wav, wav.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))
        print(f"saved wav: {out_wav}")


if __name__ == "__main__":
    main()
