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

    @torch.inference_mode()
    def decode_chunk(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode one generated chunk [F, K] to waveform.

        Note:
        This uses chunk-wise decode calls on top of the pretrained EnCodec decoder.
        It is streaming-style at the inference script level, but does not keep an
        explicit persistent decoder state between chunks.
        """
        return self.decode_from_codes(chunk_codes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--decode_wav", action="store_true")
    p.add_argument("--save_chunk_wavs", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def build_model(ckpt: Dict, dev: torch.device) -> StreamingSVSModel:
    cfg = ckpt["config"]["model"]
    meta = ckpt["meta"]
    note_vocab_size = max(int(cfg["note_vocab_size"]), int(meta.get("max_note_id", 0)) + 1)

    model = StreamingSVSModel(
        num_codebooks=int(meta["num_codebooks"]),
        frames_per_chunk=int(meta["frames_per_chunk"]),
        codebook_size=int(meta["codebook_size"]),
        note_vocab_size=note_vocab_size,
        phoneme_vocab_size=cfg["phoneme_vocab_size"],
        note_emb_dim=cfg["note_emb_dim"],
        phoneme_emb_dim=cfg["phoneme_emb_dim"],
        slur_emb_dim=cfg["slur_emb_dim"],
        phone_progress_bins=cfg["phone_progress_bins"],
        phone_progress_emb_dim=cfg["phone_progress_emb_dim"],
        cond_dim=cfg["cond_dim"],
        token_emb_dim=cfg.get("token_emb_dim", 128),
        prev_chunk_dim=cfg.get("prev_chunk_dim", 256),
        model_dim=cfg["model_dim"],
        attn_heads=cfg.get("attn_heads", 8),
        attn_layers=cfg.get("attn_layers", 4),
        history_window=cfg.get("history_window", 8),
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
    note_id = sample.note_id.unsqueeze(0).to(dev)
    phoneme = sample.phoneme_id.unsqueeze(0).to(dev)
    slur = sample.slur.unsqueeze(0).to(dev)
    phone_progress = sample.phone_progress.unsqueeze(0).to(dev)
    on_b = sample.note_on_boundary.unsqueeze(0).to(dev)
    off_b = sample.note_off_boundary.unsqueeze(0).to(dev)
    mask = torch.ones(1, codes_gt.size(1), device=dev)

    decoder = None
    stream_wav_chunks = []
    pred_chunks = []
    if args.decode_wav:
        decoder = EncodecDecoder(
            model_name=ckpt["config"]["audio"]["encodec_model_name"],
            device=dev,
        )

    with torch.no_grad():
        for chunk_idx, chunk_codes in enumerate(
            model.generate_stream(
                note_id=note_id,
                phoneme_id=phoneme,
                slur=slur,
                phone_progress=phone_progress,
                temperature=args.temperature,
            )
        ):
            pred_chunks.append(chunk_codes)
            if decoder is not None:
                chunk_wav = decoder.decode_chunk(chunk_codes.squeeze(0))
                stream_wav_chunks.append(chunk_wav)
                if args.save_chunk_wavs:
                    chunk_path = os.path.join(args.out_dir, f"sample_{args.index:04d}_chunk_{chunk_idx:04d}.wav")
                    sf.write(chunk_path, chunk_wav.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))

    codes_pred = torch.stack(pred_chunks, dim=1)

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
        wav = torch.cat(stream_wav_chunks, dim=-1) if stream_wav_chunks else torch.empty(0)
        out_wav = os.path.join(args.out_dir, f"sample_{args.index:04d}.wav")
        sf.write(out_wav, wav.numpy(), int(ckpt["config"]["audio"]["sample_rate"]))
        print(f"saved wav: {out_wav}")
        print("note: wav was produced with chunk-wise decode calls for streaming-style inference.")


if __name__ == "__main__":
    main()
