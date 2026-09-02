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
    decorr_loss,
    dynamics_loss,
    hf_noise_augment,
    hf_noise_min_bin,
    hf_time_loss,
    impulse_penalty_loss,
    kl_free_bits,
    make_keep_mask,
    recon_loss,
    smoothness_loss,
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
        dyn_mode=cfg.model.get("dyn_mode", "fwd"), # exp05 fwd/fwd_bwd toggle; .get keeps pre-exp05 configs valid
        decoder_cond_dim=int(cfg.model.get("decoder_cond_dim", 0)), # exp10 E1; 0 = exp00-09 decoder
    )
    return model.to(device)


def features_path(cfg: DictConfig) -> Path | None:
    """
    Resolve train.features_path to an absolute path, and fail loud if a consumer needs it and it is unset.
    exp10's two feature-consuming knobs are independent of each other, so either one silently training
    against no features (or against zeros) would produce a plausible-looking run of the WRONG cell.
    """
    raw = cfg.train.get("features_path", None)
    needs = int(cfg.model.get("decoder_cond_dim", 0)) > 0 or float(cfg.train.get("decorr_weight", 0.0)) > 0.0
    if raw is None:
        assert not needs, "decoder_cond_dim/decorr_weight are set but train.features_path is null"
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(str(cfg.paths.repo_root)) / path
    assert path.exists(), f"train.features_path {path} does not exist; run experiments/exp10_build_features.py"
    return path


def make_loader(cfg: DictConfig, split: str, randomize: bool, shuffle: bool) -> DataLoader:
    """Build a DataLoader of seq_len-window sequences for one split."""
    feats = features_path(cfg)
    dataset = SeqWindowDataset(cfg.paths.packed_dir, split, cfg.data.seq_len, cfg.data.window, randomize,
                               features_path=feats)
    if feats is not None:
        log.info(f"[{split}] joined {feats.name} to {len(dataset)} segments, "
                 f"{dataset.n_missing_features} without a feature row")
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
    exp09 adds three knobs, all inert by default so exp00-08 configs stay bit-identical: psd_window="dpss"
    routes the taper family (psd_taper_nw / psd_taper_k), spectral_floor puts a floor under the SPECTRAL
    SUB-TERM, and spectral_floor_mode chooses how that floor behaves (clamp = the original exp09 aux_clip,
    hinge = the Y8 rebuild).
    """
    # recon, target: (B, S, window, 1)
    atype = aux_cfg.type
    window_fn = str(aux_cfg.get("psd_window", "none")) # .get keeps pre-exp07 configs/checkpoints valid
    taper_nw = float(aux_cfg.get("psd_taper_nw", 4.0)) # dpss only; inert otherwise
    taper_k = int(aux_cfg.get("psd_taper_k", 7))

    def _spectral() -> torch.Tensor:
        """Spectral sub-term, with the exp09 floor applied if one is configured."""
        value = spectral_recon_loss(recon, target, normalize=bool(aux_cfg.psd_normalize),
                                    eps=float(aux_cfg.psd_eps), window_fn=window_fn,
                                    taper_nw=taper_nw, taper_k=taper_k)
        floor = aux_cfg.get("spectral_floor", None)
        if floor is None:
            return value
        # The floor is MEASURED by impulse ablation (roadmap Y9-F/Y9-G), never guessed, and it applies to
        # the spectral piece ALONE because exp07 pre-check C1 measured the spectral sub-term as the entire
        # effect (hf_time moved only -0.5 to -4% under the same ablation).
        mode = str(aux_cfg.get("spectral_floor_mode", "clamp"))
        if mode == "clamp":
            # Zeroes the gradient once the term is at or under the floor, so the optimizer has nothing to
            # gain by driving it further. FALSIFIED as a design (P3): a floor on the LOSS is not a floor
            # on the ACHIEVED value. The flat region has no restoring force, so once reconstruction
            # improvement incidentally carries the spectrum under the floor the term switches itself off
            # permanently and the value coasts downward (measured 4.217 against a floor of 5.23).
            return torch.clamp(value, min=float(floor))
        if mode == "hinge":
            # The rebuild (Yue Ma's Y8). Symmetric V about the floor: full descent gradient above it, unit
            # RESTORING gradient below it, so the floor is an attractor rather than a dead zone and cannot
            # self-deactivate. Equivalent to value + 2*relu(floor - value) up to the same constant, i.e.
            # the one-sided relu penalty plus the descent pressure it would otherwise discard -- and that
            # pressure is not optional: waves 3-4 measured the spectral term's real job as HOLDING THE
            # LATENT OPEN (every probe failure had a seed at <= 3 active units), which a one-sided penalty
            # inert above the floor would surrender.
            # The constant `+ floor` is deliberate: it keeps val/aux on the same scale as the clamp cell,
            # so the two are directly comparable in W&B, and a constant changes no gradient.
            return float(floor) + (value - float(floor)).abs()
        raise ValueError(f"unknown spectral_floor_mode {mode!r}; expected 'clamp' or 'hinge'")

    if atype == "log_psd":
        return _spectral()
    if atype == "hf_time":
        return hf_time_loss(recon, target)
    if atype == "combined":
        return _spectral() + float(aux_cfg.hf_weight) * hf_time_loss(recon, target) # one objective over all bands
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
    imp_weight = float(cfg.train.get("impulse_penalty_weight", 0.0)) # exp09; 0.0 = inert
    decorr_weight = float(cfg.train.get("decorr_weight", 0.0)) # exp10 E2; 0.0 = inert
    # exp09 band-limited HF-noise augmentation. TRAIN SPLIT ONLY: noising validation would make
    # val/recon incomparable across cells, and val/recon is both the checkpoint selector and the
    # x-axis of the G9-select gate. Sigma 0.0 (the default) leaves every pre-exp09 run bit-identical.
    aug_cfg = cfg.train.get("augment", None)
    noise_sigma = float(aug_cfg.get("hf_noise_sigma", 0.0)) if aug_cfg is not None else 0.0
    noise_min_bin = (
        hf_noise_min_bin(window, float(aug_cfg.get("hf_noise_min_uhz", 1000.0)))
        if aug_cfg is not None and noise_sigma > 0.0
        else 0
    )
    sums = {"recon": 0.0, "aux": 0.0, "imp": 0.0, "decorr": 0.0, "kl_total": 0.0, "kl_loss": 0.0, "dyn": 0.0,
            "mu_var": 0.0, "total": 0.0}
    kl_dim_sum = torch.zeros(cfg.model.z_dim)
    n_batches = 0

    if train:
        optimizer.zero_grad()
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_idx, batch in enumerate(tqdm(loader, desc="train" if train else "val", total=len(loader),
                                               leave=False)):
            # exp10: a features-carrying dataset yields (x, feats); every earlier config yields x alone.
            feats = None
            if isinstance(batch, (tuple, list)):
                x, feats = batch
                feats = feats.to(device, non_blocking=True) # (B, n_feat) standardized per-star features
            else:
                x = batch
            x = x.to(device, non_blocking=True) # (B, S, window, 1)
            if train and noise_sigma > 0.0:
                x = hf_noise_augment(x, noise_sigma, noise_min_bin) # noise the SAMPLE: input and target together
            x_in = x
            if aux_cfg.type == "masked":
                keep = make_keep_mask(x.shape[0] * x.shape[1], window, float(aux_cfg.mask_frac), int(aux_cfg.mask_span), device)
                x_in = x * keep.view(x.shape[0], x.shape[1], window, 1) # corrupt the input; the target stays clean
            with autocast("cuda", enabled=bool(cfg.train.amp)):
                out = model(x_in, feats)
                rl = recon_loss(out["recon"], x) # always reconstruct the CLEAN window
                kl_loss, kl_total, kl_dim = kl_free_bits(out["mu_seq"], out["logvar_seq"], cfg.train.free_bits)
                if "pred_roll" in out: # multistep mode: the optimized dyn term is the free-running rollout MSE
                    dl = dynamics_loss(out["pred_roll"], out["target_roll"])
                elif "pred_next" in out:
                    dl = dynamics_loss(out["pred_next"], out["target_next"])
                    if "pred_prev" in out: # fwd_bwd mode adds the reverse-time term under the same lambda (sum)
                        dl = dl + dynamics_loss(out["pred_prev"], out["target_prev"])
                else: # smooth mode (exp08): dynamics-free first-difference penalty under the same lambda
                    dl = smoothness_loss(out["mu_seq"])
                al = additive_aux_loss(out["recon"], x, aux_cfg)
                # exp09 impulse penalty carries its OWN weight and its own logged channel rather than
                # riding inside `aux`, so val/aux stays comparable across cells and the penalty gets
                # independent dose accounting.
                ip = (
                    impulse_penalty_loss(out["recon"], x) if imp_weight > 0.0
                    else torch.zeros((), device=out["recon"].device)
                )
                # exp10 decorrelation penalty, on its own channel for the same reason as the impulse
                # penalty above: val/aux must stay comparable across cells and the term needs its own
                # dose accounting. Computed outside autocast -- a correlation of fp16 standardized
                # columns is numerically fragile, and the term is cheap (one 128x25 matmul per batch).
                if decorr_weight > 0.0:
                    assert feats is not None, "train.decorr_weight > 0 but the batch carried no features"
                    with autocast("cuda", enabled=False):
                        dc = decorr_loss(out["mu_seq"], feats)
                else:
                    dc = torch.zeros((), device=out["recon"].device)
                loss = (rl + aux_weight * al + imp_weight * ip + beta * kl_loss
                        + cfg.train.lambda_dyn * dl + decorr_weight * dc)
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
            sums["imp"] += float(ip)
            sums["decorr"] += float(dc)
            sums["kl_total"] += float(kl_total)
            sums["kl_loss"] += float(kl_loss)
            sums["dyn"] += float(dl)
            sums["mu_var"] += float(out["mu_seq"].detach().float().var()) # latent scale; collapse monitor (exp05 high-lambda)
            sums["total"] += float(loss)
            kl_dim_sum += kl_dim.detach().float().cpu()
            n_batches += 1

    metrics = {}
    for key, value in sums.items():
        metrics[key] = value / max(1, n_batches)
    kl_dim_mean = kl_dim_sum / max(1, n_batches)
    metrics["n_active_units"] = int((kl_dim_mean > cfg.train.active_unit_kl_threshold).sum())
    return metrics


def save_checkpoint(
    path: Path, model: WorldModel, optimizer, scaler, epoch: int, best_val: float, cfg: DictConfig,
    best_select: float | None = None, scheduler=None,
) -> None:
    """Persist model, optimizer, AMP scaler, epoch, best val(s), and RNG state so a run resumes bit-identically."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None, # cosine LR state (exp05); None pre-exp05
            "epoch": epoch,
            "best_val": best_val,
            "best_select": best_select, # best KL-free selection value (dual-checkpoint tracking); None pre-exp03
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
    best (by val/monitor) and last, early-stopping when no tracked best improves for `patience` epochs.
    With train.track_recon_aux_best a second best checkpoint (best_recon_aux.pt) is kept on the KL-free
    selection metric (exp03 dual-checkpoint tracking).
    Logs the A-vs-B comparison curves (recon, total KL, active units, dynamics) grouped by variant so
    runs overlay on one chart.
    """
    set_seed(cfg.seed)
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available; this run targets the GPU"

    run_name = f"{cfg.variant_name}_seed{cfg.seed}"
    out_dir = Path(cfg.paths.models_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    # Dual-checkpoint tracking (exp03, grill 2026-07-13): the monitor's KL term is clamp-saturated noise
    # (~90% of the metric; see experiments/exp03_forensics/README.md), so alongside best.pt we can track a
    # second best on the KL-free selection metric recon + w*aux + lambda*dyn (dyn kept: it is a genuine fit
    # term; only the indicted KL term is excluded). Default false reproduces exp00-02 exactly.
    track_select = bool(cfg.train.get("track_recon_aux_best", False))
    best_select_path = out_dir / "best_recon_aux.pt"
    # exp09 THIRD checkpoint (roadmap decision A3). exp09's sweep axis IS the aux term, so
    # `val/monitor_recon_aux` differs in every cell and cells would otherwise be compared at checkpoints
    # chosen by five different rules -- the same problem exp05 solved on the dynamics axis with
    # select_include_dyn. best_recon_only.pt is selected on an AUX-INDEPENDENT metric so the aux axis is
    # not confounded with the selection rule. Both are shipped, and their difference is reported, because
    # per-epoch checkpoints do not exist and the reused exp07 ladder ends therefore cannot be re-selected
    # without retraining. Default false leaves exp00-08 untouched.
    track_recon_only = bool(cfg.train.get("track_recon_only_best", False))
    best_recon_only_path = out_dir / "best_recon_only.pt"
    best_recon_only = float("inf")

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
    # Optional cosine LR decay (exp05): smooths the constant-lr val sawtooth seen through exp04, giving a
    # cleaner post-warmup checkpoint minimum. lr_schedule=none (default) reproduces exp00-04's fixed lr.
    scheduler = None
    if str(cfg.train.get("lr_schedule", "none")) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.train.max_epochs), eta_min=float(cfg.train.get("lr_min", 3.0e-6))
        )
    scaler = GradScaler("cuda", enabled=bool(cfg.train.amp))
    train_loader = make_loader(cfg, "train", randomize=True, shuffle=True)
    val_loader = make_loader(cfg, "val", randomize=False, shuffle=False)

    start_epoch = 0
    best_val = float("inf")
    best_select = float("inf")
    patience_ctr = 0
    if cfg.train.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False) # ckpt holds cfg dict + RNG state
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None: # restore cosine phase for bit-identical resume
            scheduler.load_state_dict(ckpt["scheduler"])
        # RNG states must be CPU ByteTensors; map_location=device moved them to the GPU, so pull them back.
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        np.random.set_state(ckpt["numpy_rng"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        if ckpt.get("best_select") is not None: # key absent in pre-exp03 checkpoints
            best_select = float(ckpt["best_select"])
        log.info(f"resumed {run_name} from epoch {start_epoch}, best_val {best_val}, best_select {best_select}")

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
        # KL-free selection metric for the dual checkpoint: the same fit terms minus the clamp-saturated
        # KL noise that dominates val_monitor (exp03 forensic H2/H3). The dyn term is included by default
        # (exp00-04), but exp05 sets select_include_dyn=false so mu is selected lambda-independently and the
        # dynamics-axis comparison is apples-to-apples (a high-lambda dyn term would otherwise dominate the
        # selection and pick a dynamics-fit checkpoint rather than the best representation).
        include_dyn = bool(cfg.train.get("select_include_dyn", True))
        val_select = (
            va["recon"] + float(cfg.train.recon_aux.weight) * va["aux"]
            + (float(cfg.train.lambda_dyn) * va["dyn"] if include_dyn else 0.0)
        )
        # exp09 aux-independent selection metric: the same fit terms with the aux dropped entirely, so a
        # cell that changes the aux term is not also changing its own yardstick.
        val_recon_only = va["recon"] + (float(cfg.train.lambda_dyn) * va["dyn"] if include_dyn else 0.0)

        cur_lr = optimizer.param_groups[0]["lr"] # live lr (tracks the cosine schedule when enabled)
        record = {"epoch": epoch, "beta": beta, "lr": cur_lr, "val/monitor": val_monitor,
                  "val/monitor_recon_aux": val_select, "val/monitor_recon_only": val_recon_only}
        for key, value in tr.items():
            record[f"train/{key}"] = value
        for key, value in va.items():
            record[f"val/{key}"] = value
        wandb.log(record, step=epoch)
        log.info(
            f"[{run_name}] ep {epoch} beta {beta} "
            f"train recon {tr['recon']} aux {tr['aux']} imp {tr['imp']} decorr {tr['decorr']} "
            f"KL {tr['kl_total']} dyn {tr['dyn']} "
            f"val recon {va['recon']} aux {va['aux']} imp {va['imp']} decorr {va['decorr']} "
            f"KL {va['kl_total']} monitor {val_monitor} active {va['n_active_units']}"
        )

        # Advance the LR schedule BEFORE checkpointing so the stored scheduler state matches the lr the
        # NEXT epoch will use -- keeps resume bit-identical (no-op when scheduler is None). cur_lr above
        # already captured the lr actually used this epoch, so logging is unaffected.
        if scheduler is not None:
            scheduler.step()

        improved_monitor = epoch >= warmup and val_monitor < best_val # only steady-beta epochs are eligible as best
        improved_select = track_select and epoch >= warmup and val_select < best_select
        improved_recon_only = track_recon_only and epoch >= warmup and val_recon_only < best_recon_only
        if improved_monitor:
            best_val = val_monitor
        if improved_select:
            best_select = val_select
        if improved_recon_only:
            best_recon_only = val_recon_only
        # last.pt is written AFTER the best-value updates so a crash-resume sees the true bests (the old
        # order stored pre-update values, letting a resumed run overwrite best.pt with a worse epoch).
        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val, cfg, best_select=best_select, scheduler=scheduler)
        if improved_monitor:
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val, cfg, best_select=best_select, scheduler=scheduler)
        if improved_select:
            save_checkpoint(best_select_path, model, optimizer, scaler, epoch, best_val, cfg, best_select=best_select, scheduler=scheduler)
        if improved_recon_only:
            save_checkpoint(best_recon_only_path, model, optimizer, scaler, epoch, best_val, cfg, best_select=best_select, scheduler=scheduler)
        if epoch >= warmup:
            # With dual tracking, patience resets while ANY tracked best improves; stopping on the monitor
            # alone would kill the run on KL noise while a KL-free metric is still improving.
            if improved_monitor or improved_select or improved_recon_only:
                patience_ctr = 0
            else:
                patience_ctr += 1
        if patience_ctr >= int(cfg.train.patience) and epoch >= warmup:
            log.info(f"[{run_name}] early stop at epoch {epoch} (no improvement on any tracked best for {patience_ctr})")
            break

    wandb.finish()
