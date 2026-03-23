from __future__ import annotations

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader

from dataset import SVSSequenceDataset, collate_sequences
from model import StreamingSVSModel
from utils import (
    WarmupCosine,
    boundary_error,
    continuity_metric,
    get_device,
    load_yaml,
    masked_l1,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def build_model(cfg: dict, latent_dim: int, frames_per_chunk: int) -> StreamingSVSModel:
    m = cfg["model"]
    return StreamingSVSModel(
        latent_dim=latent_dim,
        frames_per_chunk=frames_per_chunk,
        phoneme_vocab_size=m["phoneme_vocab_size"],
        phoneme_emb_dim=m["phoneme_emb_dim"],
        pitch_proj_dim=m["pitch_proj_dim"],
        time_proj_dim=m["time_proj_dim"],
        cond_dim=m["cond_dim"],
        ctx_hidden=m["ctx_hidden"],
        ctx_layers=m["ctx_layers"],
        model_dim=m["model_dim"],
        num_blocks=m["num_blocks"],
    )


def run_epoch(model, loader, optimizer, scheduler, dev, train: bool, grad_clip: float, use_amp: bool, log_interval: int):
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and train))

    total_loss = 0.0
    total_cont = 0.0
    total_n = 0

    for step, batch in enumerate(loader, start=1):
        z = batch["z"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        cond_num = batch["cond_num"].to(dev)
        mask = batch["mask"].to(dev)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model.forward_train(z_gt=z, phoneme_id=phoneme_id, cond_num=cond_num)
            loss = masked_l1(pred, z, mask)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        cont = continuity_metric(pred.detach(), mask)

        total_loss += float(loss.item())
        total_cont += float(cont.item())
        total_n += 1

        if train and step % log_interval == 0:
            print(f"step={step} train/l1={loss.item():.5f} train/cont={cont.item():.5f}")

    return {
        "l1": total_loss / max(total_n, 1),
        "continuity": total_cont / max(total_n, 1),
    }


@torch.no_grad()
def run_validation(model, loader, dev):
    model.eval()
    total_l1 = 0.0
    total_cont = 0.0
    total_on = 0.0
    total_off = 0.0
    n = 0

    for batch in loader:
        z = batch["z"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        cond_num = batch["cond_num"].to(dev)
        mask = batch["mask"].to(dev)
        on_b = batch["note_on_boundary"].to(dev)
        off_b = batch["note_off_boundary"].to(dev)

        pred = model.forward_train(z_gt=z, phoneme_id=phoneme_id, cond_num=cond_num)

        l1 = masked_l1(pred, z, mask)
        cont = continuity_metric(pred, mask)
        on_err = boundary_error(pred, z, on_b * mask)
        off_err = boundary_error(pred, z, off_b * mask)

        total_l1 += float(l1.item())
        total_cont += float(cont.item())
        total_on += float(on_err.item())
        total_off += float(off_err.item())
        n += 1

    return {
        "l1": total_l1 / max(n, 1),
        "continuity": total_cont / max(n, 1),
        "on_boundary_l1": total_on / max(n, 1),
        "off_boundary_l1": total_off / max(n, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    dev = get_device()

    train_set = SVSSequenceDataset(cfg["data"]["train_cache"], max_seq_len=cfg["data"]["max_seq_len"])
    valid_set = SVSSequenceDataset(cfg["data"]["valid_cache"], max_seq_len=cfg["data"]["max_seq_len"])

    latent_dim = int(train_set.meta["latent_dim"])
    frames_per_chunk = int(train_set.meta["frames_per_chunk"])

    model = build_model(cfg, latent_dim, frames_per_chunk).to(dev)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params / 1e6:.2f}M")

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

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = math.inf

    for epoch in range(1, cfg["train"]["epochs"] + 1):
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
        )

        if epoch % cfg["train"]["val_interval"] == 0:
            va = run_validation(model, valid_loader, dev)
            print(
                f"epoch={epoch:03d} "
                f"train/l1={tr['l1']:.5f} train/cont={tr['continuity']:.5f} "
                f"val/l1={va['l1']:.5f} val/cont={va['continuity']:.5f} "
                f"val/on={va['on_boundary_l1']:.5f} val/off={va['off_boundary_l1']:.5f}"
            )

            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "meta": train_set.meta,
                "val": va,
            }
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), payload)

            if va["l1"] < best_val:
                best_val = va["l1"]
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), payload)


if __name__ == "__main__":
    main()
