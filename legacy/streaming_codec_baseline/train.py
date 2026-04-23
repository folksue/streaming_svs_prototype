from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SVSSequenceDataset, collate_sequences, validate_cache_meta_compatibility
from logging_utils import build_metric_logger
from model import StreamingSVSModel
from utils import (
    WarmupCosine,
    get_device,
    load_merged_yaml,
    masked_cross_entropy,
    masked_code_accuracy,
    normalize_config_paths,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", default=None, help="Can be passed multiple times; later files override earlier ones.")
    p.add_argument("--data-config", type=str, default=None)
    p.add_argument("--model-config", type=str, default=None)
    return p.parse_args()


def resolve_config_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.config:
        paths.extend(args.config)
    if args.data_config:
        paths.append(args.data_config)
    if args.model_config:
        paths.append(args.model_config)
    if not paths:
        paths = ["config.yaml"]
    return paths


def build_model(cfg: dict, num_codebooks: int, tokens_per_step: int, codebook_size: int) -> StreamingSVSModel:
    m = cfg["model"]
    note_vocab_size = max(int(m["note_vocab_size"]), int(cfg["_runtime_max_note_id"]) + 1)
    return StreamingSVSModel(
        num_codebooks=num_codebooks,
        tokens_per_step=tokens_per_step,
        codebook_size=codebook_size,
        note_vocab_size=note_vocab_size,
        phoneme_vocab_size=m["phoneme_vocab_size"],
        note_emb_dim=m["note_emb_dim"],
        phoneme_emb_dim=m["phoneme_emb_dim"],
        slur_emb_dim=m["slur_emb_dim"],
        phone_progress_bins=m["phone_progress_bins"],
        phone_progress_emb_dim=m["phone_progress_emb_dim"],
        cond_dim=m["cond_dim"],
        token_emb_dim=m.get("token_emb_dim", 128),
        prev_step_dim=m.get("prev_step_dim", 256),
        model_dim=m["model_dim"],
        attn_heads=m.get("attn_heads", 8),
        attn_layers=m.get("attn_layers", 4),
        history_window=m.get("history_window", 8),
        num_blocks=m["num_blocks"],
        audio_history_window=m.get("audio_history_window", 64),
        control_encoder_type=m.get("control_encoder_type", "mlp"),
        control_conv_layers=m.get("control_conv_layers", 0),
        control_conv_hidden_mult=m.get("control_conv_hidden_mult", 2.0),
        control_conv_kernel_size=m.get("control_conv_kernel_size", 7),
        control_conv_dilation=m.get("control_conv_dilation", 1),
        prefix_pad_steps=m.get("ablation_prefix_pad_steps", 0),
        rope_base=m.get("rope_base", 10000.0),
        ffn_mult=m.get("ffn_mult", 8.0 / 3.0),
        attn_bias=m.get("attn_bias", False),
        ffn_bias=m.get("ffn_bias", False),
        norm_eps=m.get("norm_eps", 1e-6),
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    dev,
    train: bool,
    grad_clip: float,
    use_amp: bool,
    log_interval: int,
    logger,
    epoch: int,
    global_step: int,
):
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and train))

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for step, batch in enumerate(loader, start=1):
        codes = batch["codes"].to(dev)
        note_id = batch["note_id"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        slur = batch["slur"].to(dev)
        phone_progress = batch["phone_progress"].to(dev)
        mask = batch["mask"].to(dev)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model.forward_train(
                codes=codes,
                note_id=note_id,
                phoneme_id=phoneme_id,
                slur=slur,
                phone_progress=phone_progress,
                mask=mask,
            )
            loss = masked_cross_entropy(logits, codes, mask)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        acc = masked_code_accuracy(logits.detach().argmax(dim=-1), codes, mask)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        total_n += 1

        if train and step % log_interval == 0:
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(f"step={step} train/ce={loss.item():.5f} train/acc={acc.item():.5f}")
            logger.log_metrics(
                {
                    "train/step_ce": float(loss.item()),
                    "train/step_acc": float(acc.item()),
                    "train/lr": current_lr,
                    "train/epoch": float(epoch),
                },
                step=global_step + step,
            )

    return {
        "ce": total_loss / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
        "global_step": global_step + total_n,
    }


@torch.no_grad()
def run_validation(model, loader, dev):
    model.eval()
    total_ce = 0.0
    total_acc = 0.0
    total_on = 0.0
    total_off = 0.0
    n = 0

    for batch in loader:
        codes = batch["codes"].to(dev)
        note_id = batch["note_id"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        slur = batch["slur"].to(dev)
        phone_progress = batch["phone_progress"].to(dev)
        mask = batch["mask"].to(dev)
        on_b = batch["note_on_boundary"].to(dev)
        off_b = batch["note_off_boundary"].to(dev)

        logits = model.forward_train(
            codes=codes,
            note_id=note_id,
            phoneme_id=phoneme_id,
            slur=slur,
            phone_progress=phone_progress,
            mask=mask,
        )
        pred_codes = logits.argmax(dim=-1)

        ce = masked_cross_entropy(logits, codes, mask)
        acc = masked_code_accuracy(pred_codes, codes, mask)
        on_acc = masked_code_accuracy(pred_codes, codes, on_b * mask)
        off_acc = masked_code_accuracy(pred_codes, codes, off_b * mask)

        total_ce += float(ce.item())
        total_acc += float(acc.item())
        total_on += float(on_acc.item())
        total_off += float(off_acc.item())
        n += 1

    return {
        "ce": total_ce / max(n, 1),
        "acc": total_acc / max(n, 1),
        "on_boundary_acc": total_on / max(n, 1),
        "off_boundary_acc": total_off / max(n, 1),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    cfg = normalize_config_paths(load_merged_yaml(resolve_config_paths(args)), repo_root=repo_root)
    set_seed(cfg["seed"])

    dev = get_device()

    train_set = SVSSequenceDataset(cfg["data"]["train_cache"], max_seq_len=cfg["data"]["max_seq_len"])
    valid_set = SVSSequenceDataset(cfg["data"]["valid_cache"], max_seq_len=cfg["data"]["max_seq_len"])
    validate_cache_meta_compatibility(
        train_meta=train_set.meta,
        valid_meta=valid_set.meta,
        train_path=cfg["data"]["train_cache"],
        valid_path=cfg["data"]["valid_cache"],
    )

    num_codebooks = int(train_set.meta["num_codebooks"])
    tokens_per_step = int(train_set.meta["tokens_per_step"])
    codebook_size = int(train_set.meta["codebook_size"])
    cfg["_runtime_max_note_id"] = int(train_set.meta.get("max_note_id", 0))

    model = build_model(cfg, num_codebooks, tokens_per_step, codebook_size).to(dev)
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
    logger = build_metric_logger(cfg)
    logger.log_config(cfg)
    logger.log_metrics({"model/num_params": float(num_params)}, step=0)
    best_val = math.inf
    global_step = 0

    try:
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
                save_checkpoint(os.path.join(ckpt_dir, "last.pt"), payload)

                if va["ce"] < best_val:
                    best_val = va["ce"]
                    save_checkpoint(os.path.join(ckpt_dir, "best.pt"), payload)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
