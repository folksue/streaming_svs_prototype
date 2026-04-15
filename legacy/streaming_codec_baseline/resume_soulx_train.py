from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SVSSequenceDataset, collate_sequences
from logging_utils import build_metric_logger
from train import build_model, run_epoch, run_validation
from utils import WarmupCosine, get_device, load_yaml, normalize_config_paths, save_checkpoint, set_seed


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    config_path = repo_root / "soulx_train_config.yaml"
    ckpt_path = repo_root / "artifacts" / "checkpoints" / "soulx_train" / "last.pt"
    best_ckpt_path = repo_root / "artifacts" / "checkpoints" / "soulx_train" / "best.pt"

    cfg = normalize_config_paths(load_yaml(config_path), repo_root=repo_root)
    set_seed(cfg["seed"])
    dev = get_device()

    train_set = SVSSequenceDataset(cfg["data"]["train_cache"], max_seq_len=cfg["data"]["max_seq_len"])
    valid_set = SVSSequenceDataset(cfg["data"]["valid_cache"], max_seq_len=cfg["data"]["max_seq_len"])

    num_codebooks = int(train_set.meta["num_codebooks"])
    frames_per_chunk = int(train_set.meta["frames_per_chunk"])
    codebook_size = int(train_set.meta["codebook_size"])
    model = build_model(cfg, num_codebooks, frames_per_chunk, codebook_size).to(dev)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_sequences,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_sequences,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = cfg["train"]["epochs"] * max(len(train_loader), 1)
    scheduler = WarmupCosine(
        optimizer,
        warmup_steps=cfg["train"]["warmup_steps"],
        total_steps=total_steps,
    )

    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = int(ckpt["epoch"]) + 1
    global_step = int(ckpt["epoch"]) * max(len(train_loader), 1)
    scheduler.step_id = global_step

    best_val = math.inf
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        best_val = float(best_ckpt.get("val", {}).get("ce", math.inf))
    else:
        best_val = float(ckpt.get("val", {}).get("ce", math.inf))

    print(
        f"Resuming from epoch={ckpt['epoch']} -> start_epoch={start_epoch} "
        f"global_step={global_step} best_val={best_val:.5f}"
    )

    logger = build_metric_logger(cfg)
    try:
        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            tr = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                dev=dev,
                train=True,
                grad_clip=cfg["train"]["grad_clip"],
                use_amp=cfg["train"]["use_amp"],
                log_interval=cfg["train"]["log_interval"],
                logger=logger,
                epoch=epoch,
                global_step=global_step,
            )
            global_step = int(tr.pop("global_step"))
            logger.log_metrics(
                {
                    "train/ce": float(tr["ce"]),
                    "train/acc": float(tr["acc"]),
                    "train/epoch": float(epoch),
                },
                step=global_step,
            )

            if epoch % cfg["train"]["val_interval"] == 0:
                va = run_validation(model, valid_loader, dev)
                print(
                    f"epoch={epoch:03d} "
                    f"train/ce={tr['ce']:.5f} train/acc={tr['acc']:.5f} "
                    f"val/ce={va['ce']:.5f} val/acc={va['acc']:.5f} "
                    f"val/on_acc={va['on_boundary_acc']:.5f} val/off_acc={va['off_boundary_acc']:.5f}"
                )
                logger.log_metrics(
                    {
                        "val/ce": float(va["ce"]),
                        "val/acc": float(va["acc"]),
                        "val/on_boundary_acc": float(va["on_boundary_acc"]),
                        "val/off_boundary_acc": float(va["off_boundary_acc"]),
                        "val/epoch": float(epoch),
                    },
                    step=global_step,
                )

                payload = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "meta": train_set.meta,
                    "val": va,
                }
                save_checkpoint(os.path.join(cfg["paths"]["checkpoints"], "last.pt"), payload)
                if va["ce"] < best_val:
                    best_val = va["ce"]
                    save_checkpoint(os.path.join(cfg["paths"]["checkpoints"], "best.pt"), payload)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
