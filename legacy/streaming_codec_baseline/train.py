from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ShardAwareRandomSampler, SVSSequenceDataset, collate_sequences, validate_cache_meta_compatibility
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


def run_periodic_checkpoint_qc(cfg: dict, checkpoint_path: Path, epoch: int) -> None:
    qc_cfg = cfg.get("train", {}).get("qc", {})
    if not qc_cfg.get("enabled", False):
        return
    interval_epochs = int(qc_cfg.get("interval_epochs", 0) or 0)
    if interval_epochs <= 0 or (epoch % interval_epochs) != 0:
        return

    splits = qc_cfg.get("splits", ["valid"])
    if isinstance(splits, str):
        splits = [splits]
    sample_count = int(qc_cfg.get("sample_count", 5) or 5)
    seed = int(qc_cfg.get("seed", cfg.get("seed", 42)))
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "scripts" / "sample_checkpoint_compare.py"
    qc_root = Path(cfg["paths"]["outputs"]).resolve() / "checkpoint_qc"
    run_name = cfg.get("logging", {}).get("run_name") or Path(cfg["paths"]["checkpoints"]).name

    for split in splits:
        manifest_path = cfg.get("data", {}).get(f"{split}_manifest")
        cache_path = cfg.get("data", {}).get(f"{split}_cache")
        if not manifest_path or not cache_path:
            print(f"[qc] skip split={split}: missing manifest/cache")
            continue
        out_dir = qc_root / str(split) / f"epoch_{epoch:03d}"
        title = f"{run_name} {split} epoch {epoch:03d}"
        cmd = [
            sys.executable,
            str(script_path),
            "--cache",
            str(cache_path),
            "--manifest",
            str(manifest_path),
            "--checkpoints",
            str(checkpoint_path),
            "--random-sample",
            "--sample-count",
            str(sample_count),
            "--seed",
            str(seed),
            "--out_dir",
            str(out_dir),
            "--title",
            title,
            "--link-style",
            "relative",
        ]
        print(f"[qc] running split={split} epoch={epoch:03d} -> {out_dir}")
        subprocess.run(cmd, cwd=repo_root, check=True)


def build_model(cfg: dict, num_codebooks: int, tokens_per_step: int, codebook_size: int) -> StreamingSVSModel:
    m = cfg["model"]
    note_vocab_size = max(int(m["note_vocab_size"]), int(cfg["_runtime_max_note_id"]) + 1)
    runtime_max_singer_id = int(cfg.get("_runtime_max_singer_id", 0))
    configured_singer_vocab_size = int(m.get("singer_vocab_size", 0))
    singer_vocab_size = (
        max(configured_singer_vocab_size, runtime_max_singer_id + 1)
        if configured_singer_vocab_size > 0 or runtime_max_singer_id > 0
        else 0
    )
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
        singer_vocab_size=singer_vocab_size,
        singer_emb_dim=int(m.get("singer_emb_dim", 0)),
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
    total_coarse_acc = 0.0
    total_fine_acc = 0.0
    total_per_k_acc = None
    total_n = 0

    for step, batch in enumerate(loader, start=1):
        codes = batch["codes"].to(dev)
        note_id = batch["note_id"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        slur = batch["slur"].to(dev)
        phone_progress = batch["phone_progress"].to(dev)
        singer_id = batch["singer_id"].to(dev)
        mask = batch["mask"].to(dev)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model.forward_train(
                codes=codes,
                note_id=note_id,
                phoneme_id=phoneme_id,
                slur=slur,
                phone_progress=phone_progress,
                singer_id=singer_id,
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
        coarse_acc, fine_acc = compute_coarse_fine_accuracy(logits.detach().argmax(dim=-1), codes, mask)
        per_k_acc = compute_per_codebook_accuracy(logits.detach().argmax(dim=-1), codes, mask)

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        total_coarse_acc += float(coarse_acc.item())
        total_fine_acc += float(fine_acc.item())
        if total_per_k_acc is None:
            total_per_k_acc = [0.0 for _ in range(len(per_k_acc))]
        for k in range(len(per_k_acc)):
            total_per_k_acc[k] += float(per_k_acc[k].item())
        total_n += 1

        if train and step % log_interval == 0:
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"step={step} train/ce={loss.item():.5f} train/acc={acc.item():.5f} "
                f"train/coarse_acc={coarse_acc.item():.5f} train/fine_acc={fine_acc.item():.5f}"
            )
            logger.log_metrics(
                {
                    "train/step_ce": float(loss.item()),
                    "train/step_acc": float(acc.item()),
                    "train/step_coarse_acc": float(coarse_acc.item()),
                    "train/step_fine_acc": float(fine_acc.item()),
                    **{f"train/step_acc_k{k}": float(v.item()) for k, v in enumerate(per_k_acc)},
                    "train/lr": current_lr,
                    "train/epoch": float(epoch),
                },
                step=global_step + step,
            )

    metrics = {
        "ce": total_loss / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
        "coarse_acc": total_coarse_acc / max(total_n, 1),
        "fine_acc": total_fine_acc / max(total_n, 1),
        "global_step": global_step + total_n,
    }
    if total_per_k_acc is not None:
        for k in range(len(total_per_k_acc)):
            metrics[f"acc_k{k}"] = total_per_k_acc[k] / max(total_n, 1)
    return metrics


def compute_coarse_fine_accuracy(
    pred_codes: torch.Tensor,  # [B,T,S,K]
    target_codes: torch.Tensor,  # [B,T,S,K]
    mask: torch.Tensor,  # [B,T]
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_acc = masked_code_accuracy(pred_codes[..., 0], target_codes[..., 0], mask)
    if target_codes.size(-1) <= 1:
        return coarse_acc, coarse_acc
    fine_accs = [
        masked_code_accuracy(pred_codes[..., k], target_codes[..., k], mask)
        for k in range(1, target_codes.size(-1))
    ]
    fine_acc = torch.stack(fine_accs).mean()
    return coarse_acc, fine_acc


def compute_per_codebook_accuracy(
    pred_codes: torch.Tensor,  # [B,T,S,K]
    target_codes: torch.Tensor,  # [B,T,S,K]
    mask: torch.Tensor,  # [B,T]
) -> list[torch.Tensor]:
    return [
        masked_code_accuracy(pred_codes[..., k], target_codes[..., k], mask)
        for k in range(target_codes.size(-1))
    ]


@torch.no_grad()
def run_validation(model, loader, dev):
    model.eval()
    total_ce = 0.0
    total_acc = 0.0
    total_coarse = 0.0
    total_fine = 0.0
    total_on = 0.0
    total_off = 0.0
    total_ar_acc = 0.0
    total_ar_coarse = 0.0
    total_ar_fine = 0.0
    total_ar_on = 0.0
    total_ar_off = 0.0
    total_per_k = None
    total_ar_per_k = None
    n = 0

    for batch in loader:
        codes = batch["codes"].to(dev)
        note_id = batch["note_id"].to(dev)
        phoneme_id = batch["phoneme_id"].to(dev)
        slur = batch["slur"].to(dev)
        phone_progress = batch["phone_progress"].to(dev)
        singer_id = batch["singer_id"].to(dev)
        mask = batch["mask"].to(dev)
        on_b = batch["note_on_boundary"].to(dev)
        off_b = batch["note_off_boundary"].to(dev)

        logits = model.forward_train(
            codes=codes,
            note_id=note_id,
            phoneme_id=phoneme_id,
            slur=slur,
            phone_progress=phone_progress,
            singer_id=singer_id,
            mask=mask,
        )
        pred_codes = logits.argmax(dim=-1)

        ce = masked_cross_entropy(logits, codes, mask)
        acc = masked_code_accuracy(pred_codes, codes, mask)
        coarse_acc, fine_acc = compute_coarse_fine_accuracy(pred_codes, codes, mask)
        per_k_acc = compute_per_codebook_accuracy(pred_codes, codes, mask)
        on_acc = masked_code_accuracy(pred_codes, codes, on_b * mask)
        off_acc = masked_code_accuracy(pred_codes, codes, off_b * mask)

        # Pure autoregressive validation (non-teacher-forcing): generate full sequence from control tokens.
        ar_codes = model.generate(
            note_id=note_id,
            phoneme_id=phoneme_id,
            slur=slur,
            phone_progress=phone_progress,
            singer_id=singer_id,
            temperature=0.0,
        )
        ar_acc = masked_code_accuracy(ar_codes, codes, mask)
        ar_coarse_acc, ar_fine_acc = compute_coarse_fine_accuracy(ar_codes, codes, mask)
        ar_per_k_acc = compute_per_codebook_accuracy(ar_codes, codes, mask)
        ar_on_acc = masked_code_accuracy(ar_codes, codes, on_b * mask)
        ar_off_acc = masked_code_accuracy(ar_codes, codes, off_b * mask)

        total_ce += float(ce.item())
        total_acc += float(acc.item())
        total_coarse += float(coarse_acc.item())
        total_fine += float(fine_acc.item())
        total_on += float(on_acc.item())
        total_off += float(off_acc.item())
        total_ar_acc += float(ar_acc.item())
        total_ar_coarse += float(ar_coarse_acc.item())
        total_ar_fine += float(ar_fine_acc.item())
        total_ar_on += float(ar_on_acc.item())
        total_ar_off += float(ar_off_acc.item())
        if total_per_k is None:
            total_per_k = [0.0 for _ in range(len(per_k_acc))]
            total_ar_per_k = [0.0 for _ in range(len(ar_per_k_acc))]
        for k in range(len(per_k_acc)):
            total_per_k[k] += float(per_k_acc[k].item())
            total_ar_per_k[k] += float(ar_per_k_acc[k].item())
        n += 1

    metrics = {
        "ce": total_ce / max(n, 1),
        "acc": total_acc / max(n, 1),
        "coarse_acc": total_coarse / max(n, 1),
        "fine_acc": total_fine / max(n, 1),
        "on_boundary_acc": total_on / max(n, 1),
        "off_boundary_acc": total_off / max(n, 1),
        "ar_acc": total_ar_acc / max(n, 1),
        "ar_coarse_acc": total_ar_coarse / max(n, 1),
        "ar_fine_acc": total_ar_fine / max(n, 1),
        "ar_on_boundary_acc": total_ar_on / max(n, 1),
        "ar_off_boundary_acc": total_ar_off / max(n, 1),
    }
    if total_per_k is not None and total_ar_per_k is not None:
        for k in range(len(total_per_k)):
            metrics[f"acc_k{k}"] = total_per_k[k] / max(n, 1)
            metrics[f"ar_acc_k{k}"] = total_ar_per_k[k] / max(n, 1)
    return metrics


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
    cfg["_runtime_max_singer_id"] = int(train_set.meta.get("max_singer_id", 0))

    model = build_model(cfg, num_codebooks, tokens_per_step, codebook_size).to(dev)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params / 1e6:.2f}M")

    train_sampler = ShardAwareRandomSampler(train_set, seed=int(cfg["seed"])) if getattr(train_set, "storage", "single") == "sharded" else None
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    current_epoch = 0

    try:
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            current_epoch = epoch
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
                    "train/coarse_acc": float(tr["coarse_acc"]),
                    "train/fine_acc": float(tr["fine_acc"]),
                    **{
                        f"train/acc_k{k}": float(v)
                        for k, v in sorted(
                            ((int(key.replace("acc_k", "")), value) for key, value in tr.items() if key.startswith("acc_k")),
                            key=lambda x: x[0],
                        )
                    },
                    "train/epoch": float(epoch),
                },
                step=global_step,
            )

            train_per_k_str = " ".join(
                f"train/acc_k{k}={tr[key]:.5f}"
                for k, key in sorted(
                    ((int(k.replace("acc_k", "")), k) for k in tr.keys() if k.startswith("acc_k")),
                    key=lambda x: x[0],
                )
            )

            if epoch % cfg["train"]["val_interval"] == 0:
                va = run_validation(model, valid_loader, dev)
                val_per_k_str = " ".join(
                    f"val/acc_k{k}={va[key]:.5f}"
                    for k, key in sorted(
                        ((int(k.replace("acc_k", "")), k) for k in va.keys() if k.startswith("acc_k")),
                        key=lambda x: x[0],
                    )
                )
                val_ar_per_k_str = " ".join(
                    f"val/ar_acc_k{k}={va[key]:.5f}"
                    for k, key in sorted(
                        ((int(k.replace("ar_acc_k", "")), k) for k in va.keys() if k.startswith("ar_acc_k")),
                        key=lambda x: x[0],
                    )
                )
                print(
                    f"epoch={epoch:03d} "
                    f"train/ce={tr['ce']:.5f} train/acc={tr['acc']:.5f} "
                    f"train/coarse_acc={tr['coarse_acc']:.5f} train/fine_acc={tr['fine_acc']:.5f} "
                    f"{train_per_k_str} "
                    f"val/ce={va['ce']:.5f} val/acc={va['acc']:.5f} "
                    f"val/coarse_acc={va['coarse_acc']:.5f} val/fine_acc={va['fine_acc']:.5f} "
                    f"val/on_acc={va['on_boundary_acc']:.5f} val/off_acc={va['off_boundary_acc']:.5f} "
                    f"val/ar_acc={va['ar_acc']:.5f} "
                    f"val/ar_coarse_acc={va['ar_coarse_acc']:.5f} val/ar_fine_acc={va['ar_fine_acc']:.5f} "
                    f"{val_per_k_str} {val_ar_per_k_str}"
                )
                logger.log_metrics(
                    {
                        "val/ce": float(va["ce"]),
                        "val/acc": float(va["acc"]),
                        "val/coarse_acc": float(va["coarse_acc"]),
                        "val/fine_acc": float(va["fine_acc"]),
                        "val/on_boundary_acc": float(va["on_boundary_acc"]),
                        "val/off_boundary_acc": float(va["off_boundary_acc"]),
                        "val/ar_acc": float(va["ar_acc"]),
                        "val/ar_coarse_acc": float(va["ar_coarse_acc"]),
                        "val/ar_fine_acc": float(va["ar_fine_acc"]),
                        "val/ar_on_boundary_acc": float(va["ar_on_boundary_acc"]),
                        "val/ar_off_boundary_acc": float(va["ar_off_boundary_acc"]),
                        **{
                            f"val/acc_k{k}": float(v)
                            for k, v in sorted(
                                ((int(key.replace("acc_k", "")), value) for key, value in va.items() if key.startswith("acc_k")),
                                key=lambda x: x[0],
                            )
                        },
                        **{
                            f"val/ar_acc_k{k}": float(v)
                            for k, v in sorted(
                                ((int(key.replace("ar_acc_k", "")), value) for key, value in va.items() if key.startswith("ar_acc_k")),
                                key=lambda x: x[0],
                            )
                        },
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

                ckpt_interval = int(cfg["train"].get("ckpt_interval_epochs", 10) or 10)
                if ckpt_interval > 0 and (epoch % ckpt_interval) == 0:
                    epoch_ckpt_path = Path(ckpt_dir) / f"epoch_{epoch:03d}.pt"
                    save_checkpoint(str(epoch_ckpt_path), payload)
                    try:
                        run_periodic_checkpoint_qc(cfg, epoch_ckpt_path, epoch)
                    except Exception as exc:
                        print(f"[qc] skipped epoch={epoch:03d}: {exc}")
    except KeyboardInterrupt:
        interrupted_epoch = max(current_epoch - 1, 0)
        payload = {
            "epoch": interrupted_epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "meta": train_set.meta,
            "val": {"ce": best_val} if math.isfinite(best_val) else {},
        }
        save_checkpoint(os.path.join(ckpt_dir, "interrupt_last.pt"), payload)
        print(
            f"Interrupted during epoch={current_epoch}. "
            f"Saved interrupt snapshot with resume_epoch={interrupted_epoch + 1} global_step={global_step}."
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
