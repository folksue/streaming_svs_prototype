from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_merged_yaml(paths: Iterable[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in paths:
        cfg = load_yaml(path)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a YAML object: {path}")
        merged = deep_merge_dict(merged, cfg)
    return merged


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_path(path: str | os.PathLike[str] | None, repo_root: Path | None = None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    base_dir = repo_root or get_repo_root()
    return str((base_dir / path_obj).resolve())


def normalize_config_paths(cfg: Dict[str, Any], repo_root: Path | None = None) -> Dict[str, Any]:
    root = repo_root or get_repo_root()
    resolved = deepcopy(cfg)

    data_cfg = resolved.get("data", {})
    for key in (
        "train_manifest",
        "valid_manifest",
        "train_audio_root",
        "valid_audio_root",
        "phoneme_vocab_path",
        "train_cache",
        "valid_cache",
    ):
        if key in data_cfg and data_cfg[key] is not None:
            data_cfg[key] = resolve_repo_path(data_cfg[key], root)

    for split_key in ("train_sources", "valid_sources"):
        source_list = data_cfg.get(split_key)
        if not isinstance(source_list, list):
            continue
        normalized_sources = []
        for source in source_list:
            if not isinstance(source, dict):
                normalized_sources.append(source)
                continue
            source_item = deepcopy(source)
            for source_path_key in ("manifest", "audio_root"):
                if source_path_key in source_item and source_item[source_path_key] is not None:
                    source_item[source_path_key] = resolve_repo_path(source_item[source_path_key], root)
            normalized_sources.append(source_item)
        data_cfg[split_key] = normalized_sources

    logging_cfg = resolved.get("logging", {})
    if "tensorboard_dir" in logging_cfg and logging_cfg["tensorboard_dir"] is not None:
        logging_cfg["tensorboard_dir"] = resolve_repo_path(logging_cfg["tensorboard_dir"], root)

    paths_cfg = resolved.get("paths", {})
    for key in ("checkpoints", "outputs"):
        if key in paths_cfg and paths_cfg[key] is not None:
            paths_cfg[key] = resolve_repo_path(paths_cfg[key], root)

    return resolved


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.min_lr_scale = min_lr_scale
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_id = 0

    def step(self) -> None:
        self.step_id += 1
        if self.step_id <= self.warmup_steps:
            scale = self.step_id / self.warmup_steps
        else:
            p = (self.step_id - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(p, 0.0), 1.0)))
            scale = self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine

        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * scale


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: [B, T, ...], mask: [B, T]
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_code_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    correct = (pred == target).float()
    while mask.dim() < correct.dim():
        mask = mask.unsqueeze(-1)
    expanded_mask = mask.expand_as(correct)
    correct = correct * expanded_mask
    return correct.sum() / expanded_mask.sum().clamp_min(1.0)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    while mask.dim() < target_log_probs.dim():
        mask = mask.unsqueeze(-1)
    expanded_mask = mask.expand_as(target_log_probs)
    loss = -target_log_probs * expanded_mask
    return loss.sum() / expanded_mask.sum().clamp_min(1.0)


def continuity_metric(gen_z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # gen_z: [B, T, F, D], mask: [B, T]
    if gen_z.size(1) < 2:
        return torch.tensor(0.0, device=gen_z.device)
    d = (gen_z[:, 1:] - gen_z[:, :-1]).abs().mean(dim=(2, 3))
    m = (mask[:, 1:] * mask[:, :-1]).float()
    return (d * m).sum() / m.sum().clamp_min(1.0)


def boundary_error(pred: torch.Tensor, target: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    # pred/target: [B, T, F, D], boundary_mask: [B, T]
    err = (pred - target).abs().mean(dim=(2, 3))
    m = boundary_mask.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
