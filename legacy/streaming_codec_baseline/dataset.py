from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import Dataset, Sampler

CACHE_FORMAT = "svs_step_codes"
SUPPORTED_CACHE_FORMAT_VERSION = 4
LEGACY_CACHE_FORMAT_VERSION = 3


@dataclass
class SequenceItem:
    utt_id: str
    codes: torch.Tensor
    note_id: torch.Tensor
    phoneme_id: torch.Tensor
    slur: torch.Tensor
    phone_progress: torch.Tensor
    note_on_boundary: torch.Tensor
    note_off_boundary: torch.Tensor
    singer_id: torch.Tensor


def is_sharded_cache_path(cache_path: str) -> bool:
    path = Path(cache_path)
    return path.is_dir() and (path / "index.pt").exists()


def load_cache_payload(cache_path: str) -> Dict[str, Any]:
    if is_sharded_cache_path(cache_path):
        index_path = Path(cache_path) / "index.pt"
        payload = torch.load(index_path, map_location="cpu", weights_only=False)
        validate_cache_payload(payload, cache_path)
        return payload
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    validate_cache_payload(payload, cache_path)
    return payload


class SVSSequenceDataset(Dataset):
    def __init__(self, cache_path: str, max_seq_len: int | None = None):
        self.cache_path = cache_path
        self.max_seq_len = max_seq_len
        self._shard_cache_idx: int | None = None
        self._shard_cache_items: List[Dict[str, Any]] | None = None

        payload = load_cache_payload(cache_path)
        self.meta = payload["meta"]
        self.storage = str(payload.get("storage", "single"))
        if self.storage == "sharded":
            self.index = payload["index"]
            self.shards = payload["shards"]
            self.num_items = int(payload["num_items"])
            self.items_raw = None
        else:
            self.items_raw = payload["items"]
            self.index = None
            self.shards = None
            self.num_items = len(self.items_raw)

    def shard_ranges(self) -> List[range]:
        if self.storage != "sharded":
            return [range(0, self.num_items)]
        assert self.shards is not None
        return [
            range(int(shard["start_idx"]), int(shard["start_idx"]) + int(shard["length"]))
            for shard in self.shards
        ]

    def __len__(self) -> int:
        return self.num_items

    def _load_shard_items(self, shard_idx: int) -> List[Dict[str, Any]]:
        if self._shard_cache_idx == shard_idx and self._shard_cache_items is not None:
            return self._shard_cache_items
        assert self.shards is not None
        shard_path = self.shards[shard_idx]["path"]
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        items = payload["items"]
        self._shard_cache_idx = shard_idx
        self._shard_cache_items = items
        return items

    def _raw_item(self, idx: int) -> Dict[str, Any]:
        if self.storage == "sharded":
            assert self.index is not None
            shard_idx = int(self.index[idx]["shard_idx"])
            item_idx = int(self.index[idx]["item_idx"])
            items = self._load_shard_items(shard_idx)
            return items[item_idx]
        assert self.items_raw is not None
        return self.items_raw[idx]

    def __getitem__(self, idx: int) -> SequenceItem:
        it = self._raw_item(idx)
        codes = it["codes"].long()
        note_id = it["note_id"].long()
        phoneme = it["phoneme_id"].long()
        slur = it["slur"].long()
        phone_progress = it["phone_progress"].long()
        on_b = it["note_on_boundary"].float()
        off_b = it["note_off_boundary"].float()
        singer = it["singer_id"].long() if "singer_id" in it else torch.zeros_like(note_id, dtype=torch.long)

        if self.max_seq_len is not None and codes.size(0) > self.max_seq_len:
            codes = codes[: self.max_seq_len]
            note_id = note_id[: self.max_seq_len]
            phoneme = phoneme[: self.max_seq_len]
            slur = slur[: self.max_seq_len]
            phone_progress = phone_progress[: self.max_seq_len]
            on_b = on_b[: self.max_seq_len]
            off_b = off_b[: self.max_seq_len]
            singer = singer[: self.max_seq_len]

        return SequenceItem(
            utt_id=it["utt_id"],
            codes=codes,
            note_id=note_id,
            phoneme_id=phoneme,
            slur=slur,
            phone_progress=phone_progress,
            note_on_boundary=on_b,
            note_off_boundary=off_b,
            singer_id=singer,
        )


def collate_sequences(batch: List[SequenceItem]) -> Dict[str, torch.Tensor | List[str]]:
    bsz = len(batch)
    t_max = max(x.codes.size(0) for x in batch)
    f = batch[0].codes.size(1)
    k = batch[0].codes.size(2)

    codes = torch.zeros(bsz, t_max, f, k, dtype=torch.long)
    note_id = torch.zeros(bsz, t_max, dtype=torch.long)
    phoneme_id = torch.zeros(bsz, t_max, dtype=torch.long)
    slur = torch.zeros(bsz, t_max, dtype=torch.long)
    phone_progress = torch.zeros(bsz, t_max, dtype=torch.long)
    note_on_boundary = torch.zeros(bsz, t_max)
    note_off_boundary = torch.zeros(bsz, t_max)
    singer_id = torch.zeros(bsz, t_max, dtype=torch.long)
    mask = torch.zeros(bsz, t_max)

    utt_ids: List[str] = []
    for i, it in enumerate(batch):
        t = it.codes.size(0)
        codes[i, :t] = it.codes
        note_id[i, :t] = it.note_id
        phoneme_id[i, :t] = it.phoneme_id
        slur[i, :t] = it.slur
        phone_progress[i, :t] = it.phone_progress
        note_on_boundary[i, :t] = it.note_on_boundary
        note_off_boundary[i, :t] = it.note_off_boundary
        singer_id[i, :t] = it.singer_id
        mask[i, :t] = 1.0
        utt_ids.append(it.utt_id)

    return {
        "utt_id": utt_ids,
        "codes": codes,
        "note_id": note_id,
        "phoneme_id": phoneme_id,
        "slur": slur,
        "phone_progress": phone_progress,
        "note_on_boundary": note_on_boundary,
        "note_off_boundary": note_off_boundary,
        "singer_id": singer_id,
        "mask": mask,
    }


def _require_key(container: Dict, key: str, where: str, cache_path: str):
    if key not in container:
        raise ValueError(f"Invalid cache '{cache_path}': missing key '{key}' in {where}.")


def _validate_item_tensor(item: Dict, key: str, expected_dim: int, cache_path: str):
    value = item[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Invalid cache '{cache_path}': item['{key}'] must be a torch.Tensor.")
    if value.dim() != expected_dim:
        raise ValueError(
            f"Invalid cache '{cache_path}': item['{key}'] must have dim={expected_dim}, got dim={value.dim()}."
        )


def validate_cache_payload(payload: Dict, cache_path: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache '{cache_path}': payload must be a dict.")
    _require_key(payload, "items", "payload", cache_path)
    _require_key(payload, "meta", "payload", cache_path)

    items = payload["items"]
    meta = payload["meta"]
    storage = str(payload.get("storage", "single"))
    if not isinstance(items, list):
        if storage != "sharded":
            raise ValueError(f"Invalid cache '{cache_path}': 'items' must be a list.")
        items = []
    if storage == "single" and not items:
        raise ValueError(f"Invalid cache '{cache_path}': 'items' is empty.")
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid cache '{cache_path}': 'meta' must be a dict.")

    required_meta = {
        "cache_format": str,
        "cache_format_version": int,
        "num_codebooks": int,
        "tokens_per_step": int,
        "codebook_size": int,
        "sample_rate": int,
        "phone_progress_bins": int,
        "max_note_id": int,
    }
    for key, expected_type in required_meta.items():
        _require_key(meta, key, "meta", cache_path)
        if not isinstance(meta[key], expected_type):
            raise ValueError(
                f"Invalid cache '{cache_path}': meta['{key}'] must be {expected_type.__name__}, got {type(meta[key]).__name__}."
            )

    if meta["cache_format"] != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported cache format in '{cache_path}': expected '{CACHE_FORMAT}', got '{meta['cache_format']}'."
        )
    if meta["cache_format_version"] not in (LEGACY_CACHE_FORMAT_VERSION, SUPPORTED_CACHE_FORMAT_VERSION):
        raise ValueError(
            f"Unsupported cache format version in '{cache_path}': expected one of "
            f"({LEGACY_CACHE_FORMAT_VERSION}, {SUPPORTED_CACHE_FORMAT_VERSION}), got {meta['cache_format_version']}."
        )
    if meta["num_codebooks"] <= 0 or meta["tokens_per_step"] <= 0 or meta["codebook_size"] <= 0:
        raise ValueError(
            f"Invalid cache '{cache_path}': num_codebooks/tokens_per_step/codebook_size must be positive."
        )
    if meta["phone_progress_bins"] <= 0:
        raise ValueError(
            f"Invalid cache '{cache_path}': phone_progress_bins must be positive."
        )

    expected_s = int(meta["tokens_per_step"])
    expected_k = int(meta["num_codebooks"])

    if storage == "sharded":
        _require_key(payload, "index", "payload", cache_path)
        _require_key(payload, "shards", "payload", cache_path)
        _require_key(payload, "num_items", "payload", cache_path)
        if not isinstance(payload["index"], list):
            raise ValueError(f"Invalid cache '{cache_path}': 'index' must be a list for sharded cache.")
        if not isinstance(payload["shards"], list) or not payload["shards"]:
            raise ValueError(f"Invalid cache '{cache_path}': 'shards' must be a non-empty list for sharded cache.")
        if int(payload["num_items"]) <= 0:
            raise ValueError(f"Invalid cache '{cache_path}': 'num_items' must be positive for sharded cache.")
        items_to_validate = []
    else:
        items_to_validate = items

    for idx, item in enumerate(items_to_validate):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid cache '{cache_path}': items[{idx}] must be a dict.")
        for key in (
            "utt_id",
            "codes",
            "note_id",
            "phoneme_id",
            "slur",
            "phone_progress",
            "note_on_boundary",
            "note_off_boundary",
        ):
            _require_key(item, key, f"items[{idx}]", cache_path)
        has_singer_id = "singer_id" in item

        _validate_item_tensor(item, "codes", expected_dim=3, cache_path=cache_path)
        _validate_item_tensor(item, "note_id", expected_dim=1, cache_path=cache_path)
        _validate_item_tensor(item, "phoneme_id", expected_dim=1, cache_path=cache_path)
        _validate_item_tensor(item, "slur", expected_dim=1, cache_path=cache_path)
        _validate_item_tensor(item, "phone_progress", expected_dim=1, cache_path=cache_path)
        _validate_item_tensor(item, "note_on_boundary", expected_dim=1, cache_path=cache_path)
        _validate_item_tensor(item, "note_off_boundary", expected_dim=1, cache_path=cache_path)
        if has_singer_id:
            _validate_item_tensor(item, "singer_id", expected_dim=1, cache_path=cache_path)

        codes = item["codes"]
        note_id = item["note_id"]
        phoneme_id = item["phoneme_id"]
        slur = item["slur"]
        phone_progress = item["phone_progress"]
        note_on_boundary = item["note_on_boundary"]
        note_off_boundary = item["note_off_boundary"]
        singer_id = item["singer_id"] if has_singer_id else None

        t = codes.size(0)
        if t <= 0:
            raise ValueError(f"Invalid cache '{cache_path}': items[{idx}] has empty time dimension.")
        if codes.size(1) != expected_s or codes.size(2) != expected_k:
            raise ValueError(
                f"Invalid cache '{cache_path}': items[{idx}]['codes'] shape mismatch, expected [T,{expected_s},{expected_k}], got {tuple(codes.shape)}."
            )
        if (
            note_id.size(0) != t
            or phoneme_id.size(0) != t
            or slur.size(0) != t
            or phone_progress.size(0) != t
            or note_on_boundary.size(0) != t
            or note_off_boundary.size(0) != t
            or (singer_id is not None and singer_id.size(0) != t)
        ):
            raise ValueError(
                f"Invalid cache '{cache_path}': items[{idx}] time length mismatch across tensors."
            )
        if int(phone_progress.max().item()) >= int(meta["phone_progress_bins"]):
            raise ValueError(
                f"Invalid cache '{cache_path}': items[{idx}]['phone_progress'] contains out-of-range bucket ids."
            )


def validate_cache_meta_compatibility(train_meta: Dict, valid_meta: Dict, train_path: str, valid_path: str) -> None:
    keys = (
        "cache_format",
        "cache_format_version",
        "num_codebooks",
        "tokens_per_step",
        "codebook_size",
        "phone_progress_bins",
    )
    for key in keys:
        train_v = train_meta.get(key)
        valid_v = valid_meta.get(key)
        if train_v != valid_v:
            raise ValueError(
                "Train/valid cache mismatch: "
                f"{key} differs ({train_path}: {train_v}, {valid_path}: {valid_v}). "
                "Please regenerate both caches with the same preprocess config."
            )


class ShardAwareRandomSampler(Sampler[int]):
    def __init__(self, dataset: SVSSequenceDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = int(seed)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        shard_ranges = list(self.dataset.shard_ranges())
        rng.shuffle(shard_ranges)
        for shard_range in shard_ranges:
            indices = list(shard_range)
            rng.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self) -> int:
        return len(self.dataset)
