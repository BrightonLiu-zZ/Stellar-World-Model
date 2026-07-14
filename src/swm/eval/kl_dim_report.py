"""Per-dim KL report at saved checkpoints (exp03 forensic step 1b, H2/H3 evidence).

The training loop logs only n_active_units, not the per-dim KL vector, so this script re-runs one
validation pass per checkpoint and records where each latent dim sits relative to the free-bits
floor. It also recomputes every val loss component at that checkpoint, giving point metrics for
best.pt vs last.pt (the crude selection-metric probe when no per-epoch checkpoints exist).
Outputs under --out: kl_dim_long.csv (one row per checkpoint x dim) and ckpt_summary.csv (one row
per checkpoint with recon / aux / kl / dyn / recomputed monitor / active-unit counts).

Run (from repo root, in the swm env):
    python -m swm.eval.kl_dim_report --ckpt-glob "experiments/exp0[12]*/models/B_seed0/*.pt" --out experiments/exp03_forensics
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from tqdm.auto import tqdm

from swm.train.loop import additive_aux_loss, build_model, make_loader
from swm.train.losses import dynamics_loss, kl_free_bits, recon_loss

log = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parents[3]


@torch.no_grad()
def val_pass(model, loader, cfg, device: str) -> tuple[dict[str, float], torch.Tensor]:
    """
    Run one deterministic validation pass and return (mean loss components, mean per-dim KL).
    Mirrors run_epoch's val branch (same losses, same free-bits KL) minus the masked-input
    corruption: the point is the checkpoint's latent state, and best.pt/last.pt for masked runs
    are still comparable because recon/KL are computed on clean inputs here for every run.
    """
    model.eval()  # switch off dropout / batchnorm train-mode behaviour
    aux_cfg = cfg.train.recon_aux
    sums = {"recon": 0.0, "aux": 0.0, "kl_loss": 0.0, "kl_total": 0.0, "dyn": 0.0}
    kl_dim_sum = torch.zeros(cfg.model.z_dim)
    n_batches = 0
    for x in loader:
        x = x.to(device, non_blocking=True)  # (B, S, window, 1)
        with autocast("cuda", enabled=bool(cfg.train.amp)):
            out = model(x)
            rl = recon_loss(out["recon"], x)
            kl_loss, kl_total, kl_dim = kl_free_bits(out["mu_seq"], out["logvar_seq"], cfg.train.free_bits)
            dl = dynamics_loss(out["pred_next"], out["target_next"])
            al = additive_aux_loss(out["recon"], x, aux_cfg)
        sums["recon"] += float(rl)
        sums["aux"] += float(al)
        sums["kl_loss"] += float(kl_loss)
        sums["kl_total"] += float(kl_total)
        sums["dyn"] += float(dl)
        kl_dim_sum += kl_dim.detach().float().cpu()
        n_batches += 1
    means = {}
    for key, value in sums.items():
        means[key] = value / max(1, n_batches)
    kl_dim_mean = kl_dim_sum / max(1, n_batches)
    return means, kl_dim_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="per-dim KL + loss components at saved checkpoints")
    parser.add_argument("--ckpt-glob", required=True, help="glob under the repo root matching .pt checkpoints")
    parser.add_argument("--out", required=True, help="output directory for the two CSVs")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available; the val pass targets the GPU"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = sorted(repo_root.glob(args.ckpt_glob))
    assert len(ckpt_paths) > 0, f"no checkpoints match {args.ckpt_glob}"

    summary_rows = []
    long_rows = []
    for ckpt_path in tqdm(ckpt_paths, desc="checkpoints", total=len(ckpt_paths)):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # ckpt holds the resolved cfg dict
        cfg = OmegaConf.create(ckpt["cfg"])
        if "recon_aux" not in cfg.train:  # checkpoints saved before the exp02 aux objective landed
            cfg.train.recon_aux = OmegaConf.create({"type": "none", "weight": 0.0})
        cfg.data.num_workers = 0  # single in-process reader; avoids Windows spawn overhead per checkpoint
        model = build_model(cfg, device)
        model.load_state_dict(ckpt["model"])
        loader = make_loader(cfg, "val", randomize=False, shuffle=False)
        means, kl_dim = val_pass(model, loader, cfg, device)

        monitor = (
            means["recon"] + float(cfg.train.recon_aux.weight) * means["aux"]
            + float(cfg.train.beta_target) * means["kl_loss"] + float(cfg.train.lambda_dyn) * means["dyn"]
        )
        floor = float(cfg.train.free_bits)
        n_active = int((kl_dim > float(cfg.train.active_unit_kl_threshold)).sum())
        n_above_floor = int((kl_dim > floor * 1.05).sum())  # dims carrying real info, not just pinned at the floor
        row = {
            "exp_name": str(cfg.exp_name),
            "ckpt": ckpt_path.stem,
            "epoch": int(ckpt["epoch"]),
            "monitor_recomputed": monitor,
            "n_active_units": n_active,
            "n_above_floor_1p05x": n_above_floor,
            "free_bits": floor,
            "kl_dim_max": float(kl_dim.max()),
            "kl_dim_median": float(kl_dim.median()),
        }
        for key, value in means.items():
            row[key] = value
        summary_rows.append(row)
        for dim in range(len(kl_dim)):
            long_rows.append({
                "exp_name": str(cfg.exp_name),
                "ckpt": ckpt_path.stem,
                "dim": dim,
                "kl_nats": float(kl_dim[dim]),
            })
        log.info(f"{cfg.exp_name}/{ckpt_path.stem} ep {row['epoch']}: monitor {monitor} kl {means['kl_loss']} above-floor {n_above_floor}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "ckpt_summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / "kl_dim_long.csv", index=False)
    log.info(f"wrote ckpt_summary.csv ({len(summary_rows)} ckpts) + kl_dim_long.csv to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
