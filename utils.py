from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict

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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
