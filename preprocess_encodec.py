from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm

from utils import ensure_dir, get_device, load_json, load_yaml, set_seed


@dataclass
class ChunkedItem:
    utt_id: str
    z: torch.Tensor               # [T, F, D]
    phoneme_id: torch.Tensor      # [T]
    cond_num: torch.Tensor        # [T, 7]
    note_on_boundary: torch.Tensor
    note_off_boundary: torch.Tensor


class FrozenEncodec:
    """
    Wrapper around pretrained EnCodec.
    Priority:
    1) transformers.EncodecModel (recommended)
    2) fail fast with clear error if unavailable

    NOTE:
    We keep EnCodec frozen and only use it for preprocessing/decoding.
    """

    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name

        try:
            from transformers import EncodecModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers with EncodecModel is required. Install requirements.txt first."
            ) from e

        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Some versions expose codebook size differently.
        codebook_size = getattr(self.model.config, "codebook_size", 1024)
        self.codebook_size = int(codebook_size)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Returns continuous latent frames [T, D].

        Implementation detail:
        - We use EnCodec discrete codes and normalize them to continuous features.
        - This keeps preprocessing robust across API variants.
        """
        if wav.dim() == 2:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        x = wav.unsqueeze(0).to(self.device)  # [1, 1, samples]
        encoded = self.model.encode(x)

        # Expected shape in transformers: [num_codebooks, batch, frames]
        codes = encoded.audio_codes
        if codes.dim() == 4:
            # Some versions use [num_codebooks, batch, frames, 1]
            codes = codes.squeeze(-1)

        if codes.dim() != 3:
            raise RuntimeError(f"Unexpected EnCodec code shape: {tuple(codes.shape)}")

        # -> [frames, num_codebooks]
        codes = codes[:, 0, :].transpose(0, 1).contiguous().float()
        z = (codes / max(self.codebook_size - 1, 1)) * 2.0 - 1.0
        return z.cpu()


def find_active_interval(t: float, intervals: List[List[float]]) -> int:
    for i, (s, e) in enumerate(intervals):
        if s <= t < e:
            return i
    return -1


def note_features(t: float, note_intervals: List[List[float]], note_pitch_midi: List[float]) -> Tuple[float, float, float, float, float, bool, bool]:
    idx = find_active_interval(t, note_intervals)
    if idx < 0:
        return 0.0, t, t, 0.0, 0.0, False, False

    s, e = note_intervals[idx]
    dur = max(e - s, 1e-4)
    pitch = float(note_pitch_midi[idx])
    t_since = max(t - s, 0.0)
    t_to_end = max(e - t, 0.0)
    prog = min(max((t - s) / dur, 0.0), 1.0)

    boundary_eps = 0.5 * 0.1  # half chunk at 100ms
    on_boundary = abs(t - s) <= boundary_eps
    off_boundary = abs(t - e) <= boundary_eps
    return pitch, s, e, t_since, t_to_end, prog, on_boundary, off_boundary


def phoneme_features(t: float, phoneme_intervals: List[List[float]], phoneme_ids: List[int]) -> Tuple[int, float]:
    idx = find_active_interval(t, phoneme_intervals)
    if idx < 0:
        return 0, 0.0
    s, e = phoneme_intervals[idx]
    dur = max(e - s, 1e-4)
    prog = min(max((t - s) / dur, 0.0), 1.0)
    return int(phoneme_ids[idx]), prog


def chunk_sequence(
    utt: Dict[str, Any],
    z_frames: torch.Tensor,
    chunk_ms: int,
    frames_per_chunk: Optional[int] = None,
) -> Optional[ChunkedItem]:
    duration_sec = float(utt.get("duration", 0.0))
    if duration_sec <= 0:
        # derive from alignment max end if absent
        max_note = max((x[1] for x in utt["note_intervals"]), default=0.0)
        max_ph = max((x[1] for x in utt["phoneme_intervals"]), default=0.0)
        duration_sec = max(max_note, max_ph)

    total_frames = z_frames.size(0)
    if duration_sec <= 1e-6:
        return None

    frame_rate = total_frames / duration_sec
    if frames_per_chunk is None:
        frames_per_chunk = max(1, int(round(frame_rate * (chunk_ms / 1000.0))))

    n_chunks = total_frames // frames_per_chunk
    if n_chunks < 2:
        return None

    z = z_frames[: n_chunks * frames_per_chunk].view(n_chunks, frames_per_chunk, -1)

    phoneme_ids = []
    cond_nums = []
    note_on_b = []
    note_off_b = []

    for t_idx in range(n_chunks):
        center_t = (t_idx + 0.5) * (chunk_ms / 1000.0)

        ph_id, ph_prog = phoneme_features(center_t, utt["phoneme_intervals"], utt["phoneme_ids"])
        pitch, note_on, note_off, t_since, t_to_end, note_prog, on_b, off_b = note_features(
            center_t, utt["note_intervals"], utt["note_pitch_midi"]
        )

        phoneme_ids.append(ph_id)
        cond_nums.append([
            pitch,
            note_on,
            note_off,
            t_since,
            t_to_end,
            note_prog,
            ph_prog,
        ])
        note_on_b.append(float(on_b))
        note_off_b.append(float(off_b))

    return ChunkedItem(
        utt_id=str(utt["utt_id"]),
        z=z.float(),
        phoneme_id=torch.tensor(phoneme_ids, dtype=torch.long),
        cond_num=torch.tensor(cond_nums, dtype=torch.float32),
        note_on_boundary=torch.tensor(note_on_b, dtype=torch.float32),
        note_off_boundary=torch.tensor(note_off_b, dtype=torch.float32),
    )


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def build_cache(cfg: Dict[str, Any], manifest_path: str, out_path: str) -> None:
    dev = get_device()
    encodec = FrozenEncodec(cfg["audio"]["encodec_model_name"], dev)

    entries = load_json(manifest_path)
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a list of utterance dicts")

    items: List[ChunkedItem] = []
    resolved_frames_per_chunk: Optional[int] = None

    for utt in tqdm(entries, desc=f"preprocess {manifest_path}"):
        wav = load_audio(utt["audio_path"], cfg["audio"]["sample_rate"])
        z_frames = encodec.encode(wav, cfg["audio"]["sample_rate"])
        item = chunk_sequence(
            utt=utt,
            z_frames=z_frames,
            chunk_ms=cfg["audio"]["chunk_ms"],
            frames_per_chunk=resolved_frames_per_chunk,
        )
        if item is None:
            continue

        if resolved_frames_per_chunk is None:
            resolved_frames_per_chunk = int(item.z.size(1))
        items.append(item)

    if not items:
        raise RuntimeError(f"No usable items parsed from {manifest_path}")

    latent_dim = int(items[0].z.size(-1))
    frames_per_chunk = int(items[0].z.size(1))

    payload = {
        "items": [
            {
                "utt_id": it.utt_id,
                "z": it.z,
                "phoneme_id": it.phoneme_id,
                "cond_num": it.cond_num,
                "note_on_boundary": it.note_on_boundary,
                "note_off_boundary": it.note_off_boundary,
            }
            for it in items
        ],
        "meta": {
            "latent_dim": latent_dim,
            "frames_per_chunk": frames_per_chunk,
            "chunk_ms": cfg["audio"]["chunk_ms"],
            "encodec_model": cfg["audio"]["encodec_model_name"],
            "sample_rate": cfg["audio"]["sample_rate"],
        },
    }

    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)
    print(f"Saved cache: {out_path} | items={len(items)} | latent_dim={latent_dim} | F={frames_per_chunk}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--split", type=str, choices=["train", "valid", "both"], default="both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    if args.split in ("train", "both"):
        build_cache(cfg, cfg["data"]["train_manifest"], cfg["data"]["train_cache"])
    if args.split in ("valid", "both"):
        build_cache(cfg, cfg["data"]["valid_manifest"], cfg["data"]["valid_cache"])


if __name__ == "__main__":
    main()
