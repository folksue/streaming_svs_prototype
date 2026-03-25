from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm
import soundfile as sf

from metadata_adapters import (
    adapter_needs_phoneme_vocab,
    attach_phoneme_ids,
    build_or_load_phoneme_vocab,
    load_manifest_entries,
)
from utils import ensure_dir, get_device, load_yaml, normalize_config_paths, set_seed

CACHE_FORMAT = "svs_chunk_codes"
CACHE_FORMAT_VERSION = 1


@dataclass
class ChunkedItem:
    utt_id: str
    codes: torch.Tensor           # [T, F, K]
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

        codebook_size = getattr(self.model.config, "codebook_size", 1024)
        self.codebook_size = int(codebook_size)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Returns discrete EnCodec codes [T, K]."""
        if wav.dim() == 2:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        x = wav.unsqueeze(0).to(self.device)  # [1, 1, samples]
        encoded = self.model.encode(x)

        codes = encoded.audio_codes
        if codes.dim() == 4:
            if codes.size(-1) == 1:
                codes = codes.squeeze(-1)
            elif codes.size(0) == 1 and codes.size(1) == 1:
                codes = codes.squeeze(0).squeeze(0)
            elif codes.size(0) == 1:
                codes = codes.squeeze(0)

        if codes.dim() == 3:
            if codes.size(0) == 1:
                codes = codes.squeeze(0)
            elif codes.size(1) == 1:
                codes = codes[:, 0, :]
            elif codes.size(2) == 1:
                codes = codes[:, :, 0]

        if codes.dim() != 2:
            raise RuntimeError(f"Unexpected EnCodec code shape: {tuple(codes.shape)}")

        if codes.size(0) <= codes.size(1):
            codes = codes.transpose(0, 1)
        return codes.contiguous().long().cpu()


def find_active_interval(t: float, intervals: List[List[float]]) -> int:
    for i, (s, e) in enumerate(intervals):
        if s <= t < e:
            return i
    return -1


def note_features(
    t: float,
    note_intervals: List[List[float]],
    note_pitch_midi: List[float],
) -> Tuple[float, float, float, float, float, float, bool, bool]:
    idx = find_active_interval(t, note_intervals)
    if idx < 0:
        return 0.0, t, t, 0.0, 0.0, 0.0, False, False

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
    code_frames: torch.Tensor,
    chunk_ms: int,
    frames_per_chunk: Optional[int] = None,
) -> Optional[ChunkedItem]:
    duration_sec = float(utt.get("duration", 0.0))
    if duration_sec <= 0:
        max_note = max((x[1] for x in utt["note_intervals"]), default=0.0)
        max_ph = max((x[1] for x in utt["phoneme_intervals"]), default=0.0)
        duration_sec = max(max_note, max_ph)

    total_frames = code_frames.size(0)
    if duration_sec <= 1e-6:
        return None

    frame_rate = total_frames / duration_sec
    if frames_per_chunk is None:
        frames_per_chunk = max(1, int(round(frame_rate * (chunk_ms / 1000.0))))

    n_chunks = total_frames // frames_per_chunk
    if n_chunks < 2:
        return None

    codes = code_frames[: n_chunks * frames_per_chunk].view(n_chunks, frames_per_chunk, -1)

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
        codes=codes.long(),
        phoneme_id=torch.tensor(phoneme_ids, dtype=torch.long),
        cond_num=torch.tensor(cond_nums, dtype=torch.float32),
        note_on_boundary=torch.tensor(note_on_b, dtype=torch.float32),
        note_off_boundary=torch.tensor(note_off_b, dtype=torch.float32),
    )


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav_np, sr = sf.read(path, always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1).float()

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def build_cache(
    cfg: Dict[str, Any],
    entries: List[Dict[str, Any]],
    out_path: str,
    split_name: str,
    frames_per_chunk_override: Optional[int] = None,
) -> Dict[str, int]:
    dev = get_device()
    encodec = FrozenEncodec(cfg["audio"]["encodec_model_name"], dev)

    items: List[ChunkedItem] = []
    resolved_frames_per_chunk: Optional[int] = frames_per_chunk_override

    for utt in tqdm(entries, desc=f"preprocess {split_name}"):
        wav = load_audio(utt["audio_path"], cfg["audio"]["sample_rate"])
        code_frames = encodec.encode(wav, cfg["audio"]["sample_rate"])
        item = chunk_sequence(
            utt=utt,
            code_frames=code_frames,
            chunk_ms=cfg["audio"]["chunk_ms"],
            frames_per_chunk=resolved_frames_per_chunk,
        )
        if item is None:
            continue

        if resolved_frames_per_chunk is None:
            resolved_frames_per_chunk = int(item.codes.size(1))
        items.append(item)

    if not items:
        raise RuntimeError(f"No usable items parsed for split={split_name}")

    num_codebooks = int(items[0].codes.size(-1))
    frames_per_chunk = int(items[0].codes.size(1))

    payload = {
        "items": [
            {
                "utt_id": it.utt_id,
                "codes": it.codes,
                "phoneme_id": it.phoneme_id,
                "cond_num": it.cond_num,
                "note_on_boundary": it.note_on_boundary,
                "note_off_boundary": it.note_off_boundary,
            }
            for it in items
        ],
        "meta": {
            "cache_format": CACHE_FORMAT,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "num_codebooks": num_codebooks,
            "frames_per_chunk": frames_per_chunk,
            "chunk_ms": cfg["audio"]["chunk_ms"],
            "encodec_model": cfg["audio"]["encodec_model_name"],
            "sample_rate": cfg["audio"]["sample_rate"],
            "codebook_size": encodec.codebook_size,
            "cond_dim": 7,
        },
    }

    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)
    print(f"Saved cache: {out_path} | items={len(items)} | codebooks={num_codebooks} | F={frames_per_chunk}")
    return {
        "num_codebooks": num_codebooks,
        "frames_per_chunk": frames_per_chunk,
    }


def get_split_paths(cfg: Dict[str, Any], split_name: str) -> tuple[str, str, str, str | None]:
    data_cfg = cfg["data"]
    manifest_path = data_cfg[f"{split_name}_manifest"]
    cache_path = data_cfg[f"{split_name}_cache"]
    adapter_name = data_cfg.get(f"{split_name}_manifest_adapter", data_cfg.get("manifest_adapter", "manifest"))
    audio_root = data_cfg.get(f"{split_name}_audio_root")
    return manifest_path, cache_path, adapter_name, audio_root


def maybe_auto_split_entries(
    cfg: Dict[str, Any],
    split_entries: Dict[str, List[Dict[str, Any]]],
    adapter_names: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    data_cfg = cfg["data"]
    if "train" not in split_entries or "valid" not in split_entries:
        return split_entries
    if split_entries["valid"]:
        return split_entries
    if not data_cfg.get("auto_split_valid", False):
        return split_entries

    train_entries = split_entries["train"]
    if not train_entries:
        raise ValueError("Cannot auto-split validation set from empty training entries")

    valid_ratio = float(data_cfg.get("valid_ratio", 0.05))
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"data.valid_ratio must be in (0, 1), got {valid_ratio}")

    min_valid = int(data_cfg.get("min_valid_items", 1))
    buckets = []
    for entry in train_entries:
        score = stable_hash_score(str(entry["utt_id"]), seed=int(cfg.get("seed", 42)))
        buckets.append((score, entry))
    buckets.sort(key=lambda x: x[0])

    valid_count = max(min_valid, int(round(len(buckets) * valid_ratio)))
    valid_count = min(valid_count, max(len(buckets) - 1, 1))
    valid_entries = [entry for _, entry in buckets[:valid_count]]
    train_after_split = [entry for _, entry in buckets[valid_count:]]
    if not train_after_split:
        raise ValueError("Auto split left no training data; reduce data.valid_ratio")

    split_entries["train"] = train_after_split
    split_entries["valid"] = valid_entries
    adapter_names["valid"] = adapter_names.get("train", adapter_names.get("valid", "manifest"))
    print(
        f"Auto-split validation set from training entries: "
        f"train={len(train_after_split)} valid={len(valid_entries)} ratio={valid_ratio:.4f}"
    )
    return split_entries


def stable_hash_score(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16)


def prepare_entries(cfg: Dict[str, Any], split_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    split_entries: Dict[str, List[Dict[str, Any]]] = {}
    adapter_names: Dict[str, str] = {}

    for split_name in split_names:
        manifest_path, _, adapter_name, audio_root = get_split_paths(cfg, split_name)
        entries = load_manifest_entries(
            manifest_path=manifest_path,
            adapter_name=adapter_name,
            audio_root=audio_root,
        )
        split_entries[split_name] = entries
        adapter_names[split_name] = adapter_name

    split_entries = maybe_auto_split_entries(cfg, split_entries, adapter_names)

    needs_vocab = any(adapter_needs_phoneme_vocab(name) or any("phoneme_symbols" in x for x in split_entries[split]) for split, name in adapter_names.items())
    if needs_vocab:
        vocab_path = cfg["data"].get("phoneme_vocab_path", "data/phoneme_vocab.json")
        allow_create = "train" in split_names
        vocab = build_or_load_phoneme_vocab(vocab_path, list(split_entries.values()), allow_create=allow_create)
        split_entries = {
            split_name: attach_phoneme_ids(entries, vocab)
            for split_name, entries in split_entries.items()
        }

    return split_entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--split", type=str, choices=["train", "valid", "both"], default="both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = normalize_config_paths(load_yaml(args.config))
    set_seed(cfg["seed"])

    split_names: List[str] = []
    if args.split in ("train", "both"):
        split_names.append("train")
    if args.split in ("valid", "both"):
        split_names.append("valid")

    split_entries = prepare_entries(cfg, split_names)

    train_meta: Optional[Dict[str, int]] = None
    if args.split in ("train", "both"):
        _, cache_path, _, _ = get_split_paths(cfg, "train")
        train_meta = build_cache(cfg, split_entries["train"], cache_path, split_name="train")
    if args.split in ("valid", "both"):
        _, cache_path, _, _ = get_split_paths(cfg, "valid")
        frames_per_chunk_override = None
        if train_meta is not None:
            frames_per_chunk_override = int(train_meta["frames_per_chunk"])
        build_cache(
            cfg,
            split_entries["valid"],
            cache_path,
            split_name="valid",
            frames_per_chunk_override=frames_per_chunk_override,
        )


if __name__ == "__main__":
    main()
