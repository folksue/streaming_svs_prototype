from __future__ import annotations

import argparse
import io
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import torch
import torchaudio
from tqdm import tqdm
import soundfile as sf

from metadata_adapters import (
    adapter_needs_phoneme_vocab,
    attach_phoneme_ids,
    attach_singer_ids,
    build_or_load_phoneme_vocab,
    build_or_load_singer_vocab,
    load_manifest_entries,
)
from utils import ensure_dir, get_device, load_merged_yaml, normalize_config_paths, save_json, set_seed

CACHE_FORMAT = "svs_step_codes"
CACHE_FORMAT_VERSION = 3
SHARDED_CACHE_FORMAT_VERSION = 4
RAW_CACHE_FORMAT = "encodec_codes_only"
RAW_CACHE_FORMAT_VERSION = 1


@dataclass
class StepItem:
    utt_id: str
    codes: torch.Tensor           # [T, S, K]
    note_id: torch.Tensor         # [T]
    phoneme_id: torch.Tensor      # [T]
    slur: torch.Tensor            # [T]
    phone_progress: torch.Tensor  # [T]
    note_on_boundary: torch.Tensor
    note_off_boundary: torch.Tensor
    singer_id: torch.Tensor


@dataclass
class RawCodeItem:
    utt_id: str
    codes: torch.Tensor  # [T_raw, K]


class FrozenEncodec:
    """
    Wrapper around pretrained EnCodec.
    Priority:
    1) transformers.EncodecModel (recommended)
    2) fail fast with clear error if unavailable

    NOTE:
    We keep EnCodec frozen and only use it for preprocessing/decoding.
    """

    def __init__(self, model_name: str, device: torch.device, bandwidth: float | None = None):
        self.device = device
        self.model_name = model_name
        self.bandwidth = None if bandwidth is None else float(bandwidth)

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
        upsampling_ratios = list(getattr(self.model.config, "upsampling_ratios", [1]))
        frame_stride = 1
        for ratio in upsampling_ratios:
            frame_stride *= int(ratio)
        self.frame_stride = max(frame_stride, 1)

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Returns discrete EnCodec codes [T, K]."""
        if wav.dim() == 2:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        x = wav.unsqueeze(0).to(self.device)  # [1, 1, samples]
        if self.bandwidth is None:
            encoded = self.model.encode(x)
        else:
            try:
                encoded = self.model.encode(x, bandwidth=self.bandwidth)
            except TypeError:
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

    @torch.inference_mode()
    def encode_batch(self, wavs: List[torch.Tensor], sr: int) -> List[torch.Tensor]:
        """Returns a list of discrete EnCodec codes, each shaped [T, K]."""
        if not wavs:
            return []

        lengths = [int(wav.size(-1)) for wav in wavs]
        max_len = max(lengths)
        batch = torch.zeros(len(wavs), 1, max_len, dtype=wavs[0].dtype, device=self.device)
        padding_mask = torch.zeros(len(wavs), max_len, dtype=torch.bool, device=self.device)

        for idx, wav in enumerate(wavs):
            mono = wav
            if mono.dim() == 2:
                mono = mono.mean(dim=0, keepdim=True)
            if mono.dim() == 1:
                mono = mono.unsqueeze(0)
            n = int(mono.size(-1))
            batch[idx, :, :n] = mono.to(self.device)
            padding_mask[idx, :n] = True

        encoded = self.model.encode(batch, padding_mask=padding_mask, bandwidth=self.bandwidth)
        codes = encoded.audio_codes

        if codes.dim() != 4:
            raise RuntimeError(f"Unexpected batched EnCodec code shape: {tuple(codes.shape)}")
        if codes.size(0) != 1:
            raise RuntimeError(f"Unexpected leading EnCodec batch shape: {tuple(codes.shape)}")

        codes = codes.squeeze(0).permute(0, 2, 1).contiguous()  # [B, T, K]
        outputs: List[torch.Tensor] = []
        for idx, num_samples in enumerate(lengths):
            num_frames = max(1, (num_samples + self.frame_stride - 1) // self.frame_stride)
            outputs.append(codes[idx, :num_frames, :].long().cpu())
        return outputs


def find_active_interval(t: float, intervals: List[List[float]]) -> int:
    for i, (s, e) in enumerate(intervals):
        if s <= t < e:
            return i
    return -1


def note_features(
    t: float,
    note_intervals: List[List[float]],
    note_pitch_midi: List[float],
    note_slur: List[int],
    boundary_eps: float,
) -> Tuple[int, float, float, bool, bool, int]:
    idx = find_active_interval(t, note_intervals)
    if idx < 0:
        return 0, t, t, False, False, 0

    s, e = note_intervals[idx]
    pitch = int(round(float(note_pitch_midi[idx])))
    slur = int(note_slur[idx]) if idx < len(note_slur) else 0

    on_boundary = abs(t - s) <= boundary_eps
    off_boundary = abs(t - e) <= boundary_eps
    return pitch, s, e, on_boundary, off_boundary, slur


def bucketize_progress(progress: float, num_bins: int) -> int:
    if num_bins <= 0:
        raise ValueError(f"phone_progress_bins must be positive, got {num_bins}")
    clamped = min(max(float(progress), 0.0), 1.0)
    return min(int(clamped * num_bins), num_bins - 1)


def phoneme_features(
    t: float,
    phoneme_intervals: List[List[float]],
    phoneme_ids: List[int],
    phone_progress_bins: int,
) -> Tuple[int, int]:
    idx = find_active_interval(t, phoneme_intervals)
    if idx < 0:
        return 0, 0
    s, e = phoneme_intervals[idx]
    dur = max(e - s, 1e-4)
    prog = min(max((t - s) / dur, 0.0), 1.0)
    return int(phoneme_ids[idx]), bucketize_progress(prog, phone_progress_bins)


def step_sequence(
    utt: Dict[str, Any],
    code_frames: torch.Tensor,
    tokens_per_step: int,
    phone_progress_bins: int,
    tokens_per_step_override: Optional[int] = None,
) -> Optional[StepItem]:
    duration_sec = float(utt.get("duration", 0.0))
    if duration_sec <= 0:
        max_note = max((x[1] for x in utt["note_intervals"]), default=0.0)
        max_ph = max((x[1] for x in utt["phoneme_intervals"]), default=0.0)
        duration_sec = max(max_note, max_ph)

    total_frames = code_frames.size(0)
    if duration_sec <= 1e-6:
        return None

    frame_rate = total_frames / duration_sec
    if tokens_per_step_override is None:
        tokens_per_step_override = max(1, int(tokens_per_step))

    n_steps = total_frames // tokens_per_step_override
    if n_steps < 2:
        return None

    codes = code_frames[: n_steps * tokens_per_step_override].view(n_steps, tokens_per_step_override, -1)
    step_duration_sec = tokens_per_step_override / frame_rate
    boundary_eps = 0.5 * step_duration_sec

    note_ids = []
    phoneme_ids = []
    slurs = []
    phone_progress = []
    note_on_b = []
    note_off_b = []
    singer_ids = []
    utterance_singer_id = int(utt.get("singer_id", 0))

    for t_idx in range(n_steps):
        start_frame = t_idx * tokens_per_step_override
        center_t = (start_frame + 0.5 * tokens_per_step_override) / frame_rate

        ph_id, ph_prog = phoneme_features(
            center_t,
            utt["phoneme_intervals"],
            utt["phoneme_ids"],
            phone_progress_bins=phone_progress_bins,
        )
        pitch, note_on, note_off, on_b, off_b, slur = note_features(
            center_t,
            utt["note_intervals"],
            utt["note_pitch_midi"],
            utt.get("note_slur", []),
            boundary_eps=boundary_eps,
        )

        # note_id is the integerized MIDI pitch. 0 remains the shared rest/inactive note id.
        note_ids.append(pitch)
        phoneme_ids.append(ph_id)
        slurs.append(slur)
        phone_progress.append(ph_prog)
        note_on_b.append(float(on_b))
        note_off_b.append(float(off_b))
        singer_ids.append(utterance_singer_id)

    return StepItem(
        utt_id=str(utt["utt_id"]),
        codes=codes.long(),
        note_id=torch.tensor(note_ids, dtype=torch.long),
        phoneme_id=torch.tensor(phoneme_ids, dtype=torch.long),
        slur=torch.tensor(slurs, dtype=torch.long),
        phone_progress=torch.tensor(phone_progress, dtype=torch.long),
        note_on_boundary=torch.tensor(note_on_b, dtype=torch.float32),
        note_off_boundary=torch.tensor(note_off_b, dtype=torch.float32),
        singer_id=torch.tensor(singer_ids, dtype=torch.long),
    )


_PARQUET_AUDIO_CACHE: dict[str, list[dict]] = {}


def parse_parquet_audio_uri(path: str) -> tuple[str, int]:
    if not path.startswith("parquet://"):
        raise ValueError(f"Not a parquet audio URI: {path}")
    payload = path[len("parquet://") :]
    parquet_path, row_idx = payload.rsplit("::", 1)
    return unquote(parquet_path), int(row_idx)


def load_parquet_audio_bytes(path: str) -> bytes:
    parquet_path, row_idx = parse_parquet_audio_uri(path)
    rows = _PARQUET_AUDIO_CACHE.get(parquet_path)
    if rows is None:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path, columns=["audio"])
        rows = table.column("audio").to_pylist()
        _PARQUET_AUDIO_CACHE.clear()
        _PARQUET_AUDIO_CACHE[parquet_path] = rows
    item = rows[row_idx]
    if not isinstance(item, dict) or not isinstance(item.get("bytes"), (bytes, bytearray)):
        raise RuntimeError(f"Invalid parquet audio payload at {parquet_path} row {row_idx}")
    return bytes(item["bytes"])


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    if str(path).startswith("parquet://"):
        audio_bytes = load_parquet_audio_bytes(str(path))
        wav_np, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1).float()
        if wav.numel() == 0 or wav.size(-1) == 0:
            raise RuntimeError(f"Decoded empty audio from parquet source: {path}")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav_np, sr = sf.read(path, always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1).float()

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.numel() == 0 or wav.size(-1) == 0:
        raise RuntimeError(f"Decoded empty audio from source: {path}")
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def get_split_raw_cache_path(cfg: Dict[str, Any], split_name: str) -> str | None:
    data_cfg = cfg["data"]
    raw_key = f"{split_name}_raw_codes_cache"
    raw_path = data_cfg.get(raw_key)
    if raw_path is None:
        return None
    raw_path_str = str(raw_path).strip()
    return raw_path_str or None


def save_raw_cache(
    *,
    items: List[RawCodeItem],
    out_path: str,
    cfg: Dict[str, Any],
    codebook_size: int,
) -> None:
    payload = {
        "items": [{"utt_id": it.utt_id, "codes": it.codes} for it in items],
        "meta": {
            "cache_format": RAW_CACHE_FORMAT,
            "cache_format_version": RAW_CACHE_FORMAT_VERSION,
            "encodec_model": cfg["audio"]["encodec_model_name"],
            "encodec_bandwidth": cfg["audio"].get("encodec_bandwidth"),
            "sample_rate": int(cfg["audio"]["sample_rate"]),
            "codebook_size": int(codebook_size),
            "num_codebooks": int(items[0].codes.size(-1)) if items else 0,
        },
    }
    ensure_dir(os.path.dirname(out_path) or ".")
    torch.save(payload, out_path)


def load_raw_cache(raw_path: str) -> Dict[str, Any]:
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid raw cache '{raw_path}': payload must be a dict.")
    items = payload.get("items")
    meta = payload.get("meta")
    if not isinstance(items, list) or not isinstance(meta, dict):
        raise ValueError(f"Invalid raw cache '{raw_path}': missing items/meta.")
    if str(meta.get("cache_format")) != RAW_CACHE_FORMAT:
        raise ValueError(
            f"Unsupported raw cache format in '{raw_path}': "
            f"expected '{RAW_CACHE_FORMAT}', got '{meta.get('cache_format')}'."
        )
    if int(meta.get("cache_format_version", -1)) != RAW_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported raw cache format version in '{raw_path}': "
            f"expected {RAW_CACHE_FORMAT_VERSION}, got {meta.get('cache_format_version')}."
        )
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid raw cache '{raw_path}': items[{idx}] must be a dict.")
        if "utt_id" not in item or "codes" not in item:
            raise ValueError(f"Invalid raw cache '{raw_path}': items[{idx}] missing utt_id/codes.")
        if not isinstance(item["codes"], torch.Tensor) or item["codes"].dim() != 2:
            raise ValueError(f"Invalid raw cache '{raw_path}': items[{idx}]['codes'] must be [T,K] tensor.")
    return payload


def validate_raw_cache_meta(raw_payload: Dict[str, Any], cfg: Dict[str, Any], raw_path: str) -> None:
    meta = raw_payload["meta"]
    expected_model = str(cfg["audio"]["encodec_model_name"])
    expected_bandwidth = cfg["audio"].get("encodec_bandwidth")
    expected_sample_rate = int(cfg["audio"]["sample_rate"])
    expected_num_codebooks = cfg["audio"].get("expected_num_codebooks")

    if str(meta.get("encodec_model")) != expected_model:
        raise ValueError(
            f"Raw cache '{raw_path}' encodec_model mismatch: "
            f"expected {expected_model}, got {meta.get('encodec_model')}"
        )
    if meta.get("encodec_bandwidth") != expected_bandwidth:
        raise ValueError(
            f"Raw cache '{raw_path}' encodec_bandwidth mismatch: "
            f"expected {expected_bandwidth}, got {meta.get('encodec_bandwidth')}"
        )
    if int(meta.get("sample_rate", -1)) != expected_sample_rate:
        raise ValueError(
            f"Raw cache '{raw_path}' sample_rate mismatch: "
            f"expected {expected_sample_rate}, got {meta.get('sample_rate')}"
        )
    if expected_num_codebooks is not None and int(meta.get("num_codebooks", -1)) != int(expected_num_codebooks):
        raise ValueError(
            f"Raw cache '{raw_path}' num_codebooks mismatch: "
            f"expected {expected_num_codebooks}, got {meta.get('num_codebooks')}"
        )


def build_raw_cache(
    cfg: Dict[str, Any],
    entries: List[Dict[str, Any]],
    split_name: str,
) -> tuple[List[RawCodeItem], Dict[str, int]]:
    dev = get_device()
    encodec = FrozenEncodec(
        cfg["audio"]["encodec_model_name"],
        dev,
        bandwidth=cfg["audio"].get("encodec_bandwidth"),
    )
    expected_num_codebooks = cfg["audio"].get("expected_num_codebooks")
    expected_num_codebooks = None if expected_num_codebooks is None else int(expected_num_codebooks)
    preprocess_batch_size = max(1, int(cfg["audio"].get("preprocess_batch_size", 1)))
    skipped_empty_audio = 0
    raw_items: List[RawCodeItem] = []
    pending_utts: List[Dict[str, Any]] = []
    pending_wavs: List[torch.Tensor] = []

    def flush_pending() -> None:
        if not pending_wavs:
            return
        code_batches = (
            encodec.encode_batch(pending_wavs, cfg["audio"]["sample_rate"])
            if preprocess_batch_size > 1
            else [encodec.encode(pending_wavs[0], cfg["audio"]["sample_rate"])]
        )
        for batch_utt, code_frames in zip(pending_utts, code_batches):
            if expected_num_codebooks is not None and int(code_frames.size(-1)) != expected_num_codebooks:
                raise ValueError(
                    f"Expected {expected_num_codebooks} codebooks from EnCodec, "
                    f"but got {int(code_frames.size(-1))}. "
                    f"Check audio.encodec_bandwidth / model."
                )
            raw_items.append(RawCodeItem(utt_id=str(batch_utt["utt_id"]), codes=code_frames.long().cpu()))
        pending_utts.clear()
        pending_wavs.clear()

    for utt in tqdm(entries, desc=f"encodec {split_name}"):
        try:
            wav = load_audio(utt["audio_path"], cfg["audio"]["sample_rate"])
        except RuntimeError as exc:
            if "empty audio" in str(exc).lower():
                skipped_empty_audio += 1
                print(
                    f"skip empty audio: split={split_name} utt={utt.get('utt_id', '<unknown>')} "
                    f"path={utt.get('audio_path', '<missing>')}"
                )
                continue
            raise
        pending_utts.append(utt)
        pending_wavs.append(wav)
        if len(pending_wavs) >= preprocess_batch_size:
            flush_pending()

    flush_pending()
    if not raw_items:
        raise RuntimeError(f"No usable raw EnCodec items parsed for split={split_name}")

    print(
        f"Built raw cache items: split={split_name} items={len(raw_items)} "
        f"codebooks={int(raw_items[0].codes.size(-1))} skipped_empty_audio={skipped_empty_audio}"
    )
    return raw_items, {
        "codebook_size": int(encodec.codebook_size),
        "num_codebooks": int(raw_items[0].codes.size(-1)),
    }


def build_step_cache_from_raw(
    cfg: Dict[str, Any],
    entries: List[Dict[str, Any]],
    raw_items: List[RawCodeItem],
    out_path: str,
    split_name: str,
    codebook_size: int,
    tokens_per_step_override: Optional[int] = None,
) -> Dict[str, int]:
    entry_by_utt_id = {str(entry["utt_id"]): entry for entry in entries}
    items: List[StepItem] = []
    resolved_tokens_per_step: Optional[int] = tokens_per_step_override
    skipped_step_sequence = 0

    for raw_item in tqdm(raw_items, desc=f"step-cache {split_name}"):
        utt = entry_by_utt_id.get(raw_item.utt_id)
        if utt is None:
            raise KeyError(f"Raw cache utt_id '{raw_item.utt_id}' not found in split metadata for {split_name}.")
        item = step_sequence(
            utt=utt,
            code_frames=raw_item.codes,
            tokens_per_step=cfg["audio"]["tokens_per_step"],
            phone_progress_bins=cfg["model"]["phone_progress_bins"],
            tokens_per_step_override=resolved_tokens_per_step,
        )
        if item is None:
            skipped_step_sequence += 1
            continue
        if resolved_tokens_per_step is None:
            resolved_tokens_per_step = int(item.codes.size(1))
        items.append(item)

    if not items:
        raise RuntimeError(f"No usable step-cache items parsed for split={split_name}")

    num_codebooks = int(items[0].codes.size(-1))
    tokens_per_step = int(items[0].codes.size(1))

    payload = {
        "items": [
            {
                "utt_id": it.utt_id,
                "codes": it.codes,
                "note_id": it.note_id,
                "phoneme_id": it.phoneme_id,
                "slur": it.slur,
                "phone_progress": it.phone_progress,
                "note_on_boundary": it.note_on_boundary,
                "note_off_boundary": it.note_off_boundary,
                "singer_id": it.singer_id,
            }
            for it in items
        ],
        "meta": {
            "cache_format": CACHE_FORMAT,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "num_codebooks": num_codebooks,
            "tokens_per_step": tokens_per_step,
            "encodec_model": cfg["audio"]["encodec_model_name"],
            "encodec_bandwidth": cfg["audio"].get("encodec_bandwidth"),
            "sample_rate": cfg["audio"]["sample_rate"],
            "codebook_size": int(codebook_size),
            "phone_progress_bins": int(cfg["model"]["phone_progress_bins"]),
            "max_note_id": max(int(it.note_id.max().item()) for it in items),
            "max_singer_id": max(int(it.singer_id.max().item()) for it in items),
        },
    }

    ensure_dir(os.path.dirname(out_path) or ".")
    shard_items = int(cfg["data"].get("cache_shard_items", 0) or 0)
    if shard_items > 0:
        save_sharded_cache(payload, out_path, items_per_shard=shard_items)
    else:
        torch.save(payload, out_path)
    print(
        f"Saved cache: {out_path} | items={len(items)} | codebooks={num_codebooks} | S={tokens_per_step} "
        f"| skipped_step_sequence={skipped_step_sequence}"
    )
    return {
        "num_codebooks": num_codebooks,
        "tokens_per_step": tokens_per_step,
    }


def build_cache(
    cfg: Dict[str, Any],
    entries: List[Dict[str, Any]],
    out_path: str,
    split_name: str,
    tokens_per_step_override: Optional[int] = None,
) -> Dict[str, int]:
    raw_cache_path = get_split_raw_cache_path(cfg, split_name)
    raw_payload: Dict[str, Any] | None = None
    if raw_cache_path and os.path.exists(raw_cache_path):
        raw_payload = load_raw_cache(raw_cache_path)
        validate_raw_cache_meta(raw_payload, cfg, raw_cache_path)
        print(f"Reusing raw cache: {raw_cache_path} | items={len(raw_payload['items'])}")
    else:
        raw_items, raw_meta = build_raw_cache(cfg, entries, split_name)
        if raw_cache_path:
            save_raw_cache(
                items=raw_items,
                out_path=raw_cache_path,
                cfg=cfg,
                codebook_size=int(raw_meta["codebook_size"]),
            )
            print(f"Saved raw cache: {raw_cache_path} | items={len(raw_items)}")
        raw_payload = {
            "items": [{"utt_id": it.utt_id, "codes": it.codes} for it in raw_items],
            "meta": {
                "cache_format": RAW_CACHE_FORMAT,
                "cache_format_version": RAW_CACHE_FORMAT_VERSION,
                "encodec_model": cfg["audio"]["encodec_model_name"],
                "encodec_bandwidth": cfg["audio"].get("encodec_bandwidth"),
                "sample_rate": int(cfg["audio"]["sample_rate"]),
                "codebook_size": int(raw_meta["codebook_size"]),
                "num_codebooks": int(raw_meta["num_codebooks"]),
            },
        }

    raw_items_for_step = [
        RawCodeItem(utt_id=str(item["utt_id"]), codes=item["codes"].long().cpu())
        for item in raw_payload["items"]
    ]
    return build_step_cache_from_raw(
        cfg=cfg,
        entries=entries,
        raw_items=raw_items_for_step,
        out_path=out_path,
        split_name=split_name,
        codebook_size=int(raw_payload["meta"]["codebook_size"]),
        tokens_per_step_override=tokens_per_step_override,
    )


def save_sharded_cache(payload: Dict[str, Any], out_path: str, items_per_shard: int) -> None:
    output_dir = out_path
    if output_dir.endswith(".pt"):
        output_dir = output_dir[: -len(".pt")] + "_sharded"
    ensure_dir(output_dir)
    shard_dir = os.path.join(output_dir, "shards")
    ensure_dir(shard_dir)

    items = payload["items"]
    meta = dict(payload["meta"])
    meta["cache_format_version"] = SHARDED_CACHE_FORMAT_VERSION

    index: List[Dict[str, int]] = []
    shards: List[Dict[str, Any]] = []
    for shard_idx, start in enumerate(range(0, len(items), items_per_shard)):
        shard_items_payload = items[start : start + items_per_shard]
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pt")
        torch.save({"items": shard_items_payload}, shard_path)
        shards.append(
            {
                "path": shard_path,
                "start_idx": start,
                "length": len(shard_items_payload),
            }
        )
        for item_idx in range(len(shard_items_payload)):
            index.append({"shard_idx": shard_idx, "item_idx": item_idx})

    torch.save(
        {
            "items": [],
            "meta": meta,
            "storage": "sharded",
            "num_items": len(items),
            "index": index,
            "shards": shards,
        },
        os.path.join(output_dir, "index.pt"),
    )


def get_split_cache_path(cfg: Dict[str, Any], split_name: str) -> str:
    data_cfg = cfg["data"]
    return str(data_cfg[f"{split_name}_cache"])


def get_split_sources(cfg: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    data_cfg = cfg["data"]
    list_key = f"{split_name}_sources"
    source_list = data_cfg.get(list_key)

    if isinstance(source_list, list) and source_list:
        normalized_sources: List[Dict[str, Any]] = []
        for idx, source in enumerate(source_list):
            if not isinstance(source, dict):
                raise ValueError(f"data.{list_key}[{idx}] must be a mapping")
            manifest_path = source.get("manifest")
            if not manifest_path:
                raise ValueError(f"data.{list_key}[{idx}].manifest is required")
            normalized_sources.append(
                {
                    "manifest_path": str(manifest_path),
                    "adapter_name": str(source.get("adapter", data_cfg.get("manifest_adapter", "manifest"))),
                    "audio_root": source.get("audio_root"),
                    "source_name": str(source.get("name", f"{split_name}_src{idx}")),
                }
            )
        return normalized_sources

    manifest_path = data_cfg[f"{split_name}_manifest"]
    adapter_name = data_cfg.get(f"{split_name}_manifest_adapter", data_cfg.get("manifest_adapter", "manifest"))
    audio_root = data_cfg.get(f"{split_name}_audio_root")
    return [
        {
            "manifest_path": manifest_path,
            "adapter_name": adapter_name,
            "audio_root": audio_root,
            "source_name": split_name,
        }
    ]


def maybe_auto_split_entries(
    cfg: Dict[str, Any],
    split_entries: Dict[str, List[Dict[str, Any]]],
    adapter_names: Dict[str, List[str]],
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
    adapter_names["valid"] = adapter_names.get("train", adapter_names.get("valid", ["manifest"]))
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
    adapter_names: Dict[str, List[str]] = {}

    for split_name in split_names:
        merged_entries: List[Dict[str, Any]] = []
        source_adapters: List[str] = []
        seen_utt_ids: set[str] = set()
        for source in get_split_sources(cfg, split_name):
            manifest_path = source["manifest_path"]
            adapter_name = source["adapter_name"]
            audio_root = source.get("audio_root")
            source_name = source["source_name"]
            source_entries = load_manifest_entries(
                manifest_path=manifest_path,
                adapter_name=adapter_name,
                audio_root=audio_root,
            )
            for entry in source_entries:
                uid = str(entry.get("utt_id", ""))
                if uid in seen_utt_ids:
                    item = dict(entry)
                    item["utt_id"] = f"{source_name}:{uid}"
                    merged_entries.append(item)
                    seen_utt_ids.add(item["utt_id"])
                else:
                    merged_entries.append(entry)
                    seen_utt_ids.add(uid)
            source_adapters.append(adapter_name)
        split_entries[split_name] = merged_entries
        adapter_names[split_name] = source_adapters

    split_entries = maybe_auto_split_entries(cfg, split_entries, adapter_names)

    needs_vocab = False
    for split_name, names in adapter_names.items():
        if any(adapter_needs_phoneme_vocab(name) for name in names):
            needs_vocab = True
            break
        if any("phoneme_symbols" in x for x in split_entries[split_name]):
            needs_vocab = True
            break
    if needs_vocab:
        vocab_path = cfg["data"].get("phoneme_vocab_path", "data/phoneme_vocab.json")
        allow_create = "train" in split_names
        vocab = build_or_load_phoneme_vocab(vocab_path, list(split_entries.values()), allow_create=allow_create)
        split_entries = {
            split_name: attach_phoneme_ids(entries, vocab)
            for split_name, entries in split_entries.items()
        }

    singer_vocab_path = cfg["data"].get("singer_vocab_path", "data/singer_vocab.json")
    allow_create_singer = "train" in split_names
    singer_vocab = build_or_load_singer_vocab(
        singer_vocab_path,
        list(split_entries.values()),
        allow_create=allow_create_singer,
    )
    split_entries = {
        split_name: attach_singer_ids(entries, singer_vocab)
        for split_name, entries in split_entries.items()
    }

    return split_entries


def save_split_manifests(cfg: Dict[str, Any], split_entries: Dict[str, List[Dict[str, Any]]]) -> None:
    data_cfg = cfg["data"]
    for split_name, entries in split_entries.items():
        manifest_path = data_cfg.get(f"{split_name}_manifest")
        if not manifest_path:
            continue
        ensure_dir(os.path.dirname(str(manifest_path)) or ".")
        save_json(str(manifest_path), entries)
        print(f"Saved merged manifest: {manifest_path} | entries={len(entries)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", default=None, help="Can be passed multiple times; later files override earlier ones.")
    p.add_argument("--data-config", type=str, default=None)
    p.add_argument("--model-config", type=str, default=None)
    p.add_argument("--split", type=str, choices=["train", "valid", "both"], default="both")
    return p.parse_args()


def resolve_config_paths(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    if args.config:
        paths.extend(args.config)
    if args.data_config:
        paths.append(args.data_config)
    if args.model_config:
        paths.append(args.model_config)
    if not paths:
        paths = ["config.yaml"]
    return paths


def main() -> None:
    args = parse_args()
    cfg = normalize_config_paths(load_merged_yaml(resolve_config_paths(args)))
    set_seed(cfg["seed"])

    split_names: List[str] = []
    if args.split in ("train", "both"):
        split_names.append("train")
    if args.split in ("valid", "both"):
        split_names.append("valid")

    split_entries = prepare_entries(cfg, split_names)
    save_split_manifests(cfg, split_entries)

    train_meta: Optional[Dict[str, int]] = None
    if args.split in ("train", "both"):
        cache_path = get_split_cache_path(cfg, "train")
        train_meta = build_cache(cfg, split_entries["train"], cache_path, split_name="train")
    if args.split in ("valid", "both"):
        cache_path = get_split_cache_path(cfg, "valid")
        tokens_per_step_override = None
        if train_meta is not None:
            tokens_per_step_override = int(train_meta["tokens_per_step"])
        build_cache(
            cfg,
            split_entries["valid"],
            cache_path,
            split_name="valid",
            tokens_per_step_override=tokens_per_step_override,
        )


if __name__ == "__main__":
    main()
