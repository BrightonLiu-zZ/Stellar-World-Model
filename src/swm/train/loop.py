from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb
from swm.data.dataset import SeqWindowDataset
from swm.models import WorldModel
from swm.train.losses import dynamics_loss, kl_free_bits, recon_loss
from swm.utils.seed import set_seed

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig, device: str) -> WorldModel:
    """Instantiate the locked Conv1D-VAE + GRU world model and move it to the device."""
    model = WorldModel(
        in_ch=cfg.model.in_ch,
        enc_channels=list(cfg.model.enc_channels),
        kernel_size=cfg.model.kernel_size,
        z_dim=cfg.model.z_dim,
        window=cfg.data.window,
        gru_hidden=cfg.model.gru_hidden,
        gru_layers=cfg.model.gru_layers,
    )
    return model.to(device)


def make_loader(cfg: DictConfig, split: str, randomize: bool, shuffle: bool) -> DataLoader:
    """Build a DataLoader of seq_len-window sequences for one split."""
    dataset = SeqWindowDataset(cfg.paths.packed_dir, split, cfg.data.seq_len, cfg.data.window, randomize)
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=shuffle, # keep training batches full; val keeps every sequence
    )


def beta_at_epoch(epoch: int, warmup: int, target: float) -> float:
    """Linear KL warmup: beta rises from 0 to target over `warmup` epochs, then stays at target."""
    if warmup <= 0:
        return target
    return target * min(1.0, epoch / warmup)


def run_epoch(
    model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    cfg: DictConfig,
    beta: float,
    device: str,
    train: bool,
) -> dict[str, float]:
    """
    Run one pass over a split.
    For each batch of sequences: forward --> recon + beta*KL(free-bits) + lambda*dynamics --> (if
    training) backprop with AMP, gradient accumulation, and grad-norm clipping. Accumulates the loss
    terms plus the per-dim mean KL so the caller can report total KL and the active-unit count.
    """
    model.train(train)
    accum = max(1, int(cfg.train.accum_steps))
    sums = {"recon": 0.0, "kl_total": 0.0, "kl_loss": 0.0, "dyn": 0.0, "total": 0.0}
    kl_dim_sum = torch.zeros(cfg.model.z_dim)
    n_batches = 0

    if train:
        optimizer.zero_grad()
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_idx, x in enumerate(loader):
            x = x.to(device, non_blocking=True) # (B, S, window, 1)
            with autocast("cuda", enabled=bool(cfg.train.amp)):
                out = model(x)
                rl = recon_loss(out["recon"], x)
                kl_loss, kl_total, kl_dim = kl_free_bits(out["mu_seq"], out["logvar_seq"], cfg.train.free_bits)
                dl = dynamics_loss(out["pred_next"], out["target_next"])
                loss = rl + beta * kl_loss + cfg.train.lambda_dyn * dl
            if train:
                scaler.scale(loss / accum).backward()
                if (batch_idx + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            sums["recon"] += float(rl)
            sums["kl_total"] += float(kl_total)
            sums["kl_loss"] += float(kl_loss)
            sums["dyn"] += float(dl)
            sums["total"] += float(loss)
            kl_dim_sum += kl_dim.detach().float().cpu()
            n_batches += 1

    metrics = {}
    for key, value in sums.items():
        metrics[key] = value / max(1, n_batches)
    kl_dim_mean = kl_dim_sum / max(1, n_batches)
    metrics["n_active_units"] = int((kl_dim_mean > cfg.train.active_unit_kl_threshold).sum())
    return metrics


def save_checkpoint(path: Path, model: WorldModel, optimizer, scaler, epoch: int, best_val: float, cfg: DictConfig) -> None:
    """Persist model, optimizer, AMP scaler, epoch, best val, and RNG state so a run resumes bit-identically."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def train(cfg: DictConfig) -> None:
    """
    Pretrain one variant-by-seed run end to end.
    Sets up W&B, the model, the train/val loaders, then loops epochs with KL warmup, checkpointing
    best (by val total) and last, early-stopping on val total. Logs the A-vs-B comparison curves
    (recon, total KL, active units, dynamics) grouped by variant so runs overlay on one chart.
    """
    set_seed(cfg.seed)
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available; this run targets the GPU"

    run_name = f"{cfg.variant_name}_seed{cfg.seed}"
    out_dir = Path(cfg.paths.models_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"

    wandb.init(
        project=cfg.train.wandb.project,
        entity=cfg.train.wandb.entity,
        group=cfg.variant_name, # A and B share a group so they overlay on one chart
        name=run_name,
        mode=cfg.train.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scaler = GradScaler("cuda", enabled=bool(cfg.train.amp))
    train_loader = make_loader(cfg, "train", randomize=True, shuffle=True)
    val_loader = make_loader(cfg, "val", randomize=False, shuffle=False)

    start_epoch = 0
    best_val = float("inf")
    patience_ctr = 0
    if cfg.train.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False) # ckpt holds cfg dict + RNG state
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        log.info(f"resumed {run_name} from epoch {start_epoch}, best_val {best_val}")

    for epoch in range(start_epoch, int(cfg.train.max_epochs)):
        beta = beta_at_epoch(epoch, int(cfg.train.beta_warmup_epochs), float(cfg.train.beta_target))
        tr = run_epoch(model, train_loader, optimizer, scaler, cfg, beta, device, train=True)
        va = run_epoch(model, val_loader, None, scaler, cfg, beta, device, train=False)

        # Select checkpoints on the validation training loss at the steady TARGET beta, using the same
        # free-bits KL the model actually optimizes (kl_loss): monitor = recon + beta_target*kl_loss + lambda*dyn.
        # Restricted to AFTER warmup. The scheduled-beta total is minimized at beta=0 (untrained), and during
        # warmup beta/KL are in flux (a transient KL dip can falsely win); judging only post-warmup epochs at a
        # fixed beta makes the metric comparable so it tracks genuine fit, not the warmup transient.
        warmup = int(cfg.train.beta_warmup_epochs)
        val_monitor = va["recon"] + float(cfg.train.beta_target) * va["kl_loss"] + float(cfg.train.lambda_dyn) * va["dyn"]

        record = {"epoch": epoch, "beta": beta, "lr": cfg.train.lr, "val/monitor": val_monitor}
        for key, value in tr.items():
            record[f"train/{key}"] = value
        for key, value in va.items():
            record[f"val/{key}"] = value
        wandb.log(record, step=epoch)
        log.info(
            f"[{run_name}] ep {epoch} beta {beta} "
            f"train recon {tr['recon']} KL {tr['kl_total']} dyn {tr['dyn']} "
            f"val recon {va['recon']} KL {va['kl_total']} monitor {val_monitor} active {va['n_active_units']}"
        )

        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val, cfg)
        if epoch >= warmup and val_monitor < best_val: # only steady-beta epochs are eligible as best
            best_val = val_monitor
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val, cfg)
            patience_ctr = 0
        elif epoch >= warmup:
            patience_ctr += 1
        if patience_ctr >= int(cfg.train.patience) and epoch >= warmup:
            log.info(f"[{run_name}] early stop at epoch {epoch} (no monitor improvement for {patience_ctr})")
            break

    wandb.finish()
