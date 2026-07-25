"""exp05 rollout-vs-persistence eval: the "learned physics" test (plan 2026-07-22, success criterion 2).

For each trained cell (best_recon_aux), free-run the GRU forward from a star's first latent z_1 over the
sequence horizon and compare the multi-step rollout to the trivial copy-last (persistence) baseline, in
the SAME raw-mu MSE space as dynamics_loss. Stratified by class: dynamics that learned real temporal
structure should beat persistence on PERIODIC stars (eb/pulsating/rotation) far more than on QUIET stars.
Also decodes a few rolled latents back to flux (periodic vs quiet) for the qualitative figure.

Quantitative output : experiments/<exp>/results/rollout_vs_persistence.csv  (one row per cell x seed x class)
Qualitative output  : experiments/<exp>/figs/rollout_examples_<class>_<tic>.png

Run (repo root, swm env, PYTHONPATH=src), after training + the eval scan:
    python -m swm.eval.rollout_eval --exp-glob "exp05_*" --seeds 0 1 2 3
    python -m swm.eval.rollout_eval --exp-glob "exp05_comb_multi_c1p0" --seeds 0 --examples 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from swm.eval.skyline import load_first_segment_blocks
from swm.models import WorldModel

log = logging.getLogger(__name__)
repo_root = Path(__file__).resolve().parents[3]

# v1 class strata: periodic = any localized/periodic variability; quiet = none of the four v1 positives.
PERIODIC_TASKS = ("eb", "pulsating", "rotation")
ALL_TASKS = ("eb", "pulsating", "rotation", "transit")


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Build the model recorded in the checkpoint, passing dyn_mode so fwd_bwd cells strict-load."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]; mc = cfg["model"]
    model = WorldModel(
        in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
        z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
        dyn_mode=str(mc.get("dyn_mode", "fwd")), # fwd_bwd ckpts carry dynamics_bwd.* -> needed for strict load
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_class_map(labels_csv: Path) -> dict[int, str]:
    """Map tic_id -> 'periodic' | 'quiet' | 'other' from v1 star labels."""
    df = pd.read_csv(labels_csv)
    df["tic_id"] = df["tic_id"].astype(int)
    cls = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        periodic = any(int(d.get(t, 0) or 0) == 1 for t in PERIODIC_TASKS)
        anypos = any(int(d.get(t, 0) or 0) == 1 for t in ALL_TASKS)
        cls[int(d["tic_id"])] = "periodic" if periodic else ("quiet" if not anypos else "other")
    return cls


@torch.no_grad()
def rollout_vs_persistence(model: WorldModel, block: np.ndarray, horizon: int, device: str):
    """
    For one star's first-segment windows (n_win, window): encode to mu, free-run the GRU from mu_1 over
    K=min(horizon, n_win-1) steps, and return (rollout_mse, persistence_mse, K) in raw-mu space plus the
    per-step arrays for plotting. Returns None if the star has <2 windows (no transition to predict).
    """
    n_win = block.shape[0]
    if n_win < 2:
        return None
    k = min(int(horizon), n_win - 1)
    x = torch.from_numpy(block[: k + 1]).unsqueeze(-1).to(device) # (k+1, window, 1)
    mu, _ = model.encoder(x) # (k+1, z)
    z0 = mu[0:1] # (1, z) -- the real first latent
    target = mu[1 : k + 1] # (k, z) -- z_2..z_{k+1}
    roll = model.dynamics.rollout(z0, k)[0] # (k, z) -- free-running predictions
    pers = z0.expand(k, -1) # (k, z) -- copy-last baseline
    roll_mse = float(((roll - target) ** 2).mean())
    pers_mse = float(((pers - target) ** 2).mean())
    per_step_roll = ((roll - target) ** 2).mean(dim=1).cpu().numpy() # (k,)
    per_step_pers = ((pers - target) ** 2).mean(dim=1).cpu().numpy()
    return dict(roll_mse=roll_mse, pers_mse=pers_mse, k=k,
                per_step_roll=per_step_roll, per_step_pers=per_step_pers, mu=mu, roll=roll)


@torch.no_grad()
def decode_example(model: WorldModel, block: np.ndarray, horizon: int, device: str, out_png: Path, title: str):
    """Decode the free-run rollout back to flux and plot true vs rolled-predicted windows (periodic/quiet)."""
    n_win = block.shape[0]
    k = min(int(horizon), n_win - 1)
    x = torch.from_numpy(block[: k + 1]).unsqueeze(-1).to(device)
    mu, _ = model.encoder(x)
    roll = model.dynamics.rollout(mu[0:1], k)[0] # (k, z)
    recon_roll = model.decoder(roll).squeeze(-1).cpu().numpy() # (k, window)
    steps = [s for s in (0, k // 2, k - 1) if s >= 0] # first, middle, last predicted step
    fig, axes = plt.subplots(len(steps), 1, figsize=(9, 2.2 * len(steps)), squeeze=False)
    for ax, s in zip(axes[:, 0], steps):
        ax.plot(block[s + 1], lw=1.0, label=f"true window t={s+2}")
        ax.plot(recon_roll[s], lw=1.0, alpha=0.8, label=f"rolled-decoded (step {s+1})")
        ax.legend(fontsize=7); ax.set_xlabel("cadence"); ax.set_ylabel("norm flux")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp05 rollout-vs-persistence learned-physics eval")
    parser.add_argument("--exp-glob", required=True, help="glob under experiments/ selecting cell folders")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--ckpt", default="best_recon_aux")
    parser.add_argument("--variant", default="B")
    parser.add_argument("--horizon", type=int, default=15, help="max rollout steps (<= seq_len-1)")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--labels-csv", default="labels/variability_labels_star.csv")
    parser.add_argument("--examples", type=int, default=3, help="decoded example stars per class per cell")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_map = load_class_map(repo_root / args.labels_csv)
    exp_dirs = sorted(p for p in (repo_root / "experiments").glob(args.exp_glob) if p.is_dir())
    if not exp_dirs:
        log.warning(f"no experiments match {args.exp_glob}"); return

    for exp_dir in exp_dirs:
        window = None
        blocks_cache = None # (tics, blocks) loaded once per exp (geometry-shared across seeds)
        for seed in args.seeds:
            ckpt_path = exp_dir / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            if not ckpt_path.exists():
                log.warning(f"{exp_dir.name} seed {seed}: no {args.ckpt}.pt; skipped"); continue
            model, cfg = load_model(ckpt_path, device)
            window = int(cfg["data"]["window"])
            if blocks_cache is None:
                tics, blocks = load_first_segment_blocks(exp_dir / "packed", args.split, window)
                blocks_cache = (tics, blocks)
            tics, blocks = blocks_cache

            # accumulate per-class MSEs
            agg: dict[str, dict[str, list]] = {c: {"roll": [], "pers": []} for c in ("periodic", "quiet", "other")}
            examples_done = {"periodic": 0, "quiet": 0}
            for tic, block in zip(tics, blocks):
                cls = class_map.get(int(tic), "other")
                res = rollout_vs_persistence(model, block, args.horizon, device)
                if res is None:
                    continue
                agg[cls]["roll"].append(res["roll_mse"]); agg[cls]["pers"].append(res["pers_mse"])
                if cls in examples_done and examples_done[cls] < args.examples:
                    png = exp_dir / "figs" / f"rollout_{cls}_seed{seed}_tic{int(tic)}.png"
                    decode_example(model, block, args.horizon, device, png,
                                   f"{exp_dir.name} seed{seed} {cls} TIC {int(tic)}")
                    examples_done[cls] += 1

            rows = []
            for cls in ("periodic", "quiet", "other"):
                r = np.array(agg[cls]["roll"]); p = np.array(agg[cls]["pers"])
                if len(r) == 0:
                    continue
                rows.append({
                    "exp": exp_dir.name, "seed": seed, "ckpt": args.ckpt, "dyn_mode": cfg["model"].get("dyn_mode", "fwd"),
                    "lambda_dyn": cfg["train"]["lambda_dyn"], "split": args.split, "class": cls, "n_stars": int(len(r)),
                    "rollout_mse": float(r.mean()), "persistence_mse": float(p.mean()),
                    "gain_ratio": float(p.mean() / r.mean()) if r.mean() > 0 else float("nan"),
                    "beats_persistence_frac": float((r < p).mean()),
                })
            out_csv = exp_dir / "results" / "rollout_vs_persistence.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            # append-only, auditable (like readout_sweep.csv)
            df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            summary = " | ".join(f"{r['class']}: roll {r['rollout_mse']:.4f} vs pers {r['persistence_mse']:.4f} "
                                 f"({r['gain_ratio']:.2f}x, {r['n_stars']}n)" for r in rows if r["class"] != "other")
            log.info(f"{exp_dir.name} seed{seed} [{cfg['model'].get('dyn_mode','fwd')} l={cfg['train']['lambda_dyn']}]  {summary}")


if __name__ == "__main__":
    main()
