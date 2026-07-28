"""exp05 criterion 2, the missing null: what gain_ratio does a model with NO learned dynamics already score?

`analyze_exp05_rollout.py` reports gain_ratio = persistence_mse / rollout_mse, where persistence means
"repeat the first latent mu_1 for every step". For a mu-trajectory that wanders around a per-star mean,
mu_1 is itself one noisy draw, so persistence error is roughly TWICE the error of simply predicting that
mean. A ratio near 2 is therefore available with no learned dynamics at all -- and crucially the size of
that free ratio is CLASS-DEPENDENT (a quiet star's mu wanders more randomly; a periodic star's mu is
autocorrelated, which makes mu_1 more informative and shrinks the free ratio). The pre-registered
"periodic > quiet" ordering was tested against zero, not against this floor.

This script measures the floor directly, per cell x seed x class, using an ORACLE CONSTANT predictor
(the mean of the target latents -- the best any flat prediction could do):

    pers_mse   persistence: repeat mu_1                     (the existing baseline)
    clim_mse   oracle constant: the target latents' own mean (the floor)
    roll_mse   the free-running GRU rollout                  (the model)

  gain_ratio      = pers/roll   what F1 plots
  floor_ratio     = pers/clim   what a constant already scores
  roll_over_clim  = roll/clim   1.0 means the rollout learned nothing beyond a mean

Quiet stars are subsampled (they outnumber periodic ~3.5:1) with a fixed RNG so every cell and seed
sees the identical star set -- the comparison is between cells, so the sample must not move.

Writes experiments/exp05_rollout_floor.csv. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp05_rollout_floor.py
    python experiments/analyze_exp05_rollout_floor.py --cells exp05_comb_multi_c1p0 --seeds 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from swm.eval.skyline import load_first_segment_blocks
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

# multistep cells are the in-distribution ones (trained free-running); the off cells are the controls
DEFAULT_CELLS = ("exp05_comb_multi_c1p0", "exp05_lpsd_multi_c1p0", "exp05_comb_off", "exp05_lpsd_off")
PERIODIC_TASKS = ("eb", "pulsating", "rotation")
ALL_TASKS = ("eb", "pulsating", "rotation", "transit")


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild the checkpoint's model. dyn_mode is passed so fwd_bwd cells strict-load their bwd head."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    mc = cfg["model"]
    model = WorldModel(
        in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
        z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
        dyn_mode=str(mc.get("dyn_mode", "fwd")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_class_map(labels_csv: Path) -> dict[int, str]:
    """tic_id -> 'periodic' | 'quiet' | 'other', identical to rollout_eval's stratification."""
    df = pd.read_csv(labels_csv)
    df["tic_id"] = df["tic_id"].astype(int)
    out = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        periodic = any(int(d.get(t, 0) or 0) == 1 for t in PERIODIC_TASKS)
        anypos = any(int(d.get(t, 0) or 0) == 1 for t in ALL_TASKS)
        out[int(d["tic_id"])] = "periodic" if periodic else ("quiet" if not anypos else "other")
    return out


@torch.no_grad()
def star_mses(model: WorldModel, block: np.ndarray, horizon: int, device: str) -> dict[str, float] | None:
    """persistence / oracle-constant / rollout MSE for one star, in the same raw-mu space as dynamics_loss."""
    if block.shape[0] < 2:
        return None
    k = min(int(horizon), block.shape[0] - 1)
    x = torch.from_numpy(block[: k + 1]).unsqueeze(-1).to(device)   # (k+1, window, 1)
    mu, _ = model.encoder(x)                                        # (k+1, z)
    target = mu[1: k + 1]                                           # (k, z)
    pers = mu[0:1].expand(k, -1)                                    # repeat mu_1
    clim = target.mean(dim=0, keepdim=True).expand(k, -1)           # oracle constant
    roll = model.dynamics.rollout(mu[0:1], k)[0]                    # (k, z)
    return {"pers": float(((pers - target) ** 2).mean()),
            "clim": float(((clim - target) ** 2).mean()),
            "roll": float(((roll - target) ** 2).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description="exp05 criterion-2 free-floor measurement")
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--variant", default="B")
    ap.add_argument("--horizon", type=int, default=15, help="must match rollout_eval's horizon")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n-quiet", type=int, default=500, help="quiet stars subsampled (fixed RNG)")
    ap.add_argument("--labels-csv", default="labels/variability_labels_star.csv")
    ap.add_argument("--out", default="experiments/exp05_rollout_floor.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_map = load_class_map(ROOT / args.labels_csv)
    log.info(f"device {device} | horizon {args.horizon} | cells {len(args.cells)}")

    rows = []
    for cell in args.cells:
        cell_dir = ROOT / "experiments" / cell
        if not cell_dir.exists():
            log.warning(f"{cell}: no such experiment dir; skipped")
            continue
        tics = blocks = idx = None
        for seed in args.seeds:
            ckpt = cell_dir / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            if not ckpt.exists():
                log.warning(f"{cell} seed {seed}: no {args.ckpt}.pt; skipped")
                continue
            model, cfg = load_model(ckpt, device)
            if tics is None:
                tics, blocks = load_first_segment_blocks(cell_dir / "packed", args.split,
                                                         int(cfg["data"]["window"]))
                usable = [i for i in range(len(tics)) if blocks[i].shape[0] > 1]
                quiet = [i for i in usable if class_map.get(int(tics[i])) == "quiet"]
                # fixed RNG: the same star set for every cell and seed, so cells stay comparable
                keep = set(np.random.default_rng(0).choice(quiet, min(args.n_quiet, len(quiet)),
                                                           replace=False).tolist())
                keep |= {i for i in usable if class_map.get(int(tics[i])) != "quiet"}
                idx = sorted(keep)
                n_per = sum(class_map.get(int(tics[i])) == "periodic" for i in idx)
                log.info(f"{cell}: {len(idx)} stars ({n_per} periodic, {len(idx) - n_per} quiet/other)")
            for i in idx:
                m = star_mses(model, blocks[i], args.horizon, device)
                if m is None:
                    continue
                rows.append({"cell": cell.replace("exp05_", ""), "seed": seed,
                             "cls": class_map.get(int(tics[i]), "other"), "tic_id": int(tics[i]), **m})
            log.info(f"  {cell} seed {seed} done ({len(rows)} rows)")

    if not rows:
        raise SystemExit("no rows produced - check --cells and that checkpoints exist")
    per_star = pd.DataFrame(rows)
    agg = per_star.groupby(["cell", "seed", "cls"])[["pers", "clim", "roll"]].mean().reset_index()
    agg["n_stars"] = per_star.groupby(["cell", "seed", "cls"]).size().to_numpy()
    agg["gain_ratio"] = agg["pers"] / agg["roll"]        # what F1 plots
    agg["floor_ratio"] = agg["pers"] / agg["clim"]       # what a constant already scores
    agg["roll_over_clim"] = agg["roll"] / agg["clim"]    # 1.0 = nothing learned beyond a mean
    out_path = ROOT / args.out
    agg.to_csv(out_path, index=False)
    log.info(f"\nwrote {out_path}  ({len(agg)} cell x seed x class rows)")

    pd.set_option("display.width", 200)
    print()
    print(agg.groupby(["cell", "cls"])[["gain_ratio", "floor_ratio", "roll_over_clim"]]
          .mean().round(4).to_string())
    print("\nthe pre-registered ordering, tested against the floor instead of against zero:")
    for cell in agg.cell.unique():
        w = agg[agg.cell == cell].pivot_table(index="seed", columns="cls", values="roll_over_clim")
        if not {"periodic", "quiet"} <= set(w.columns):
            continue
        # roll/clim SMALLER = rollout further ahead of a constant, so quiet - periodic > 0 supports it
        delta = (w["quiet"] - w["periodic"]).to_numpy()
        se2 = 2 * delta.std(ddof=1) / np.sqrt(len(delta))
        print(f"  {cell:24s} quiet - periodic (roll/clim) = {delta.mean():+.4f} +/- {se2:.4f}"
              f"  {'PASS' if abs(delta.mean()) > se2 else 'no gate'}")


if __name__ == "__main__":
    main()
