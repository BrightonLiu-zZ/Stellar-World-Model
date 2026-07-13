from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from swm.data.dataset import SeqWindowDataset
from swm.models import WorldModel
from swm.train.losses import (
    dynamics_loss,
    hf_time_loss,
    kl_free_bits,
    make_keep_mask,
    recon_loss,
    spectral_recon_loss,
)
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


def additive_aux_loss(recon: torch.Tensor, target: torch.Tensor, aux_cfg: DictConfig) -> torch.Tensor:
    """
    Auxiliary reconstruction term for the exp02 objective sweep, selected by aux_cfg.type.
    log_psd is the log-power-spectrum MSE; hf_time is the high-pass first-difference MSE; combined sums the
    log_psd term and hf_weight-scaled hf_time term into one general (pretrain-once) objective.
    The masked and none types add nothing here (masked corrupts the input upstream and still uses the plain
    time-MSE), so this returns a zero scalar for them.
    """
    # recon, target: (B, S, window, 1)
    atype = aux_cfg.type
    if atype == "log_psd":
        return spectral_recon_loss(recon, target, normalize=bool(aux_cfg.psd_normalize), eps=float(aux_cfg.psd_eps))
    if atype == "hf_time":
        return hf_time_loss(recon, target)
    if atype == "combined":
        spectral = spectral_recon_loss(recon, target, normalize=bool(aux_cfg.psd_normalize), eps=float(aux_cfg.psd_eps))
        return spectral + float(aux_cfg.hf_weight) * hf_time_loss(recon, target) # one objective over all bands
    return torch.zeros((), device=recon.device) # none, masked


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
    aux_cfg = cfg.train.recon_aux
    aux_weight = float(aux_cfg.weight)
    window = int(cfg.data.window)
    sums = {"recon": 0.0, "aux": 0.0, "kl_total": 0.0, "kl_loss": 0.0, "dyn": 0.0, "total": 0.0}
    kl_dim_sum = torch.zeros(cfg.model.z_dim)
    n_batches = 0

    if train:
        optimizer.zero_grad()
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_idx, x in enumerate(tqdm(loader, desc="train" if train else "val", total=len(loader), leave=False)):
            x = x.to(device, non_blocking=True) # (B, S, window, 1)
            x_in = x
            if aux_cfg.type == "masked":
                keep = make_keep_mask(x.shape[0] * x.shape[1], window, float(aux_cfg.mask_frac), int(aux_cfg.mask_span), device)
                x_in = x * keep.view(x.shape[0], x.shape[1], window, 1) # corrupt the input; the target stays clean
            with autocast("cuda", enabled=bool(cfg.train.amp)):
                out = model(x_in)
                rl = recon_loss(out["recon"], x) # always reconstruct the CLEAN window
                kl_loss, kl_total, kl_dim = kl_free_bits(out["mu_seq"], out["logvar_seq"], cfg.train.free_bits)
                dl = dynamics_loss(out["pred_next"], out["target_next"])
                al = additive_aux_loss(out["recon"], x, aux_cfg)
                loss = rl + aux_weight * al + beta * kl_loss + cfg.train.lambda_dyn * dl
            if train:
                scaler.scale(loss / accum).backward()
                if (batch_idx + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            sums["recon"] += float(rl)
            sums["aux"] += float(al)
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
        group=cfg.exp_name, # one W&B group per experiment (A/B/C of a sweep combo overlay within it)
        name=f"{cfg.exp_name}_{run_name}", # include exp_name so sweep combos are distinguishable in W&B
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
        # RNG states must be CPU ByteTensors; map_location=device moved them to the GPU, so pull them back.
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        np.random.set_state(ckpt["numpy_rng"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        log.info(f"resumed {run_name} from epoch {start_epoch}, best_val {best_val}")

    for epoch in range(start_epoch, int(cfg.train.max_epochs)):
        beta = beta_at_epoch(epoch, int(cfg.train.beta_warmup_epochs), float(cfg.train.beta_target))
        tr = run_epoch(model, train_loader, optimizer, scaler, cfg, beta, device, train=True)
        va = run_epoch(model, val_loader, None, scaler, cfg, beta, device, train=False)

        # Select checkpoints on the validation training loss at the steady TARGET beta, using the same
        # free-bits KL the model actually optimizes (kl_loss): monitor = recon + aux_weight*aux +
        # beta_target*kl_loss + lambda*dyn. The exp02 aux term MUST appear here or best-checkpoint selection
        # would ignore the new objective (the same bug class as the old beta=0 untrained-epoch selection).
        # Restricted to AFTER warmup. The scheduled-beta total is minimized at beta=0 (untrained), and during
        # warmup beta/KL are in flux (a transient KL dip can falsely win); judging only post-warmup epochs at a
        # fixed beta makes the metric comparable so it tracks genuine fit, not the warmup transient.
        warmup = int(cfg.train.beta_warmup_epochs)
        val_monitor = (
            va["recon"] + float(cfg.train.recon_aux.weight) * va["aux"]
            + float(cfg.train.beta_target) * va["kl_loss"] + float(cfg.train.lambda_dyn) * va["dyn"]
        )

        record = {"epoch": epoch, "beta": beta, "lr": cfg.train.lr, "val/monitor": val_monitor}
        for key, value in tr.items():
            record[f"train/{key}"] = value
        for key, value in va.items():
            record[f"val/{key}"] = value
        wandb.log(record, step=epoch)
        log.info(
            f"[{run_name}] ep {epoch} beta {beta} "
            f"train recon {tr['recon']} aux {tr['aux']} KL {tr['kl_total']} dyn {tr['dyn']} "
            f"val recon {va['recon']} aux {va['aux']} KL {va['kl_total']} monitor {val_monitor} active {va['n_active_units']}"
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
