"""exp06 pre-design forensics, part 3: is there any temporal signal for a dynamics model to learn?

exp05 section F2b measured the FLOOR of criterion 2 and found the free-running rollout sitting 1-7%
WORSE than an oracle per-star constant, having started 67-104% worse at lambda=0. That established the
rollout has not learned to beat a flat line, but it left the decisive question open: is the GRU weak, or
is there nothing in the mu-trajectory to predict at this geometry? A floor alone cannot separate those.

This script supplies the missing upper half, on the same stars, horizon and metric as the floor script:

  budget    decompose the variance of mu over (star, window) into a between-star part and a within-star
            temporal part. If the within-star share is small, "predict this star's mean" is near-optimal
            by construction, the constant is close to unbeatable, and criterion 2 was close to
            unwinnable at window=256 regardless of how good the GRU is.

  acf       autocorrelation of the mu trajectory by lag, split into periodic and quiet stars. A flat ACF
            means the within-star variation is white and genuinely unpredictable. An oscillatory ACF on
            periodic stars means real structure exists that the GRU failed to capture, which would make
            this a modelling failure rather than an ill-posed task.

  ar        a linear AR(1) skyline. The transition matrix is fitted in closed form on TRAIN stars and
            run free-running on test from mu_1, exactly as the GRU is, so the three predictors are
            directly comparable. The ordering of {oracle constant, AR(1), GRU} is the result: an AR(1)
            that beats the constant while the GRU does not means the GRU is the problem; an AR(1) that
            also fails means the signal is not there.

  period    per-star within-star mu variance against the star's catalogued period, which tests the
            geometry explanation directly. Variance that dies as the period grows past the 8.53-hour
            window is the signature of a window too short to see the physics.

Writes experiments/exp06_temporal.csv, experiments/exp06_temporal_acf.csv and
experiments/exp06_temporal_stars.parquet. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp06_temporal.py
    python experiments/analyze_exp06_temporal.py --cells exp05_comb_multi_c1p0 --seeds 0 --n-quiet 100
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.eval.skyline import load_first_segment_blocks
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CELLS = ("exp05_comb_off", "exp05_comb_fbwd_c1p0", "exp05_comb_multi_c1p0", "exp05_lpsd_multi_c1p0")
PERIODIC_TASKS = ("eb", "pulsating", "rotation")
ALL_TASKS = ("eb", "pulsating", "rotation", "transit")
PERIOD_COLS = ("eb_period", "pulsating_period", "rotation_period")
MAX_LAG = 8


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild one run's world model from its checkpoint, strict-loading the backward head when present."""
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


def load_labels(labels_csv: Path) -> tuple[dict[int, str], pd.DataFrame]:
    """
    Build the tic -> class map used by rollout_eval, plus the period columns for the geometry test.
    Classes are identical to the floor script's so the two CSVs can be read side by side.
    """
    df = pd.read_csv(labels_csv)
    df["tic_id"] = df["tic_id"].astype(int)
    class_map = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        periodic = False
        anypos = False
        for task in ALL_TASKS:
            if int(d.get(task, 0) or 0) == 1:
                anypos = True
                if task in PERIODIC_TASKS:
                    periodic = True
        if periodic:
            class_map[int(d["tic_id"])] = "periodic"
        elif anypos:
            class_map[int(d["tic_id"])] = "other"
        else:
            class_map[int(d["tic_id"])] = "quiet"
    keep = ["tic_id"]
    for col in PERIOD_COLS:
        if col in df.columns:
            keep.append(col)
    return class_map, df[keep]


@torch.no_grad()
def encode_block(model: WorldModel, block: np.ndarray, horizon: int, device: str) -> torch.Tensor | None:
    """Encode a star's first-segment windows into its mu trajectory, truncated to the rollout horizon."""
    if block.shape[0] < 2:
        return None
    k = min(int(horizon), block.shape[0] - 1)
    x = torch.from_numpy(block[: k + 1]).unsqueeze(-1).to(device) # (k+1, window, 1)
    mu, _ = model.encoder(x) # (k+1, z)
    return mu


def trajectory_acf(mu: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Autocorrelation of one star's mu trajectory, averaged over latent dimensions.
    Each dimension is demeaned over time first, so this measures temporal structure rather than the
    star's mean position. Lags beyond the trajectory length return NaN and are dropped on aggregation.
    """
    centred = mu - mu.mean(axis=0, keepdims=True) # (T, z)
    denom = (centred ** 2).sum(axis=0) # (z,)
    out = np.full(max_lag, np.nan)
    for lag in range(1, max_lag + 1):
        if lag >= centred.shape[0]:
            break
        num = (centred[:-lag] * centred[lag:]).sum(axis=0) # (z,)
        usable = denom > 1e-12
        if usable.any():
            out[lag - 1] = float(np.mean(num[usable] / denom[usable]))
    return out


def fit_ar1(trajectories: list[np.ndarray], ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form least-squares AR(1) transition for the latent, fitted across many stars.
    Solving mu_{t+1} = A mu_t + b jointly over every consecutive pair gives the best LINEAR one-step
    predictor, which is the natural skyline for a GRU that is also predicting one step at a time.
    Ridge regularisation keeps the 128x128 solve stable when some latent dimensions are near-dead.
    """
    xs = []
    ys = []
    for mu in trajectories:
        xs.append(mu[:-1])
        ys.append(mu[1:])
    x = np.concatenate(xs, axis=0) # (n_pairs, z)
    y = np.concatenate(ys, axis=0)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) # (n_pairs, z+1) absorbs the intercept
    gram = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1])
    coef = np.linalg.solve(gram, x_aug.T @ y) # (z+1, z)
    return coef[:-1].T, coef[-1] # A (z, z), b (z,)


def ar_rollout(mu0: np.ndarray, steps: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run the fitted AR(1) free-running from mu_1, feeding each prediction back as the GRU rollout does."""
    out = np.zeros((steps, mu0.shape[0]))
    state = mu0
    for t in range(steps):
        state = a @ state + b
        out[t] = state
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 pre-design forensics: temporal signal in mu")
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--horizon", type=int, default=15, help="must match rollout_eval and the floor script")
    ap.add_argument("--n-quiet", type=int, default=500)
    ap.add_argument("--n-ar-train", type=int, default=600, help="train stars used to fit the AR(1) skyline")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--labels-csv", default="labels/variability_labels_star.csv")
    ap.add_argument("--out", default="experiments/exp06_temporal.csv")
    ap.add_argument("--out-acf", default="experiments/exp06_temporal_acf.csv")
    ap.add_argument("--out-stars", default="experiments/exp06_temporal_stars.parquet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_map, periods = load_labels(ROOT / args.labels_csv)
    log.info(f"device {device} | horizon {args.horizon}")

    runs = []
    cfg = None
    for cell in args.cells:
        for seed in args.seeds:
            path = ROOT / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            if path.exists():
                runs.append((cell, seed, path))
            else:
                log.warning(f"{cell} seed {seed}: no {args.ckpt}.pt; skipped")
    assert runs, "no checkpoints found - check --cells/--seeds"

    packed = ROOT / "experiments" / args.cells[0] / "packed"
    test_tics, test_blocks = load_first_segment_blocks(packed, "test", 256)
    train_tics, train_blocks = load_first_segment_blocks(packed, "train", 256)

    usable = []
    for i in range(len(test_tics)):
        if test_blocks[i].shape[0] > 1:
            usable.append(i)
    quiet = []
    for i in usable:
        if class_map.get(int(test_tics[i])) == "quiet":
            quiet.append(i)
    # fixed RNG so every arm and seed scores the identical star set, as the floor script does
    keep = set(np.random.default_rng(0).choice(quiet, min(args.n_quiet, len(quiet)), replace=False).tolist())
    for i in usable:
        if class_map.get(int(test_tics[i])) != "quiet":
            keep.add(i)
    test_idx = sorted(keep)

    ar_usable = []
    for i in range(len(train_tics)):
        if train_blocks[i].shape[0] > 1:
            ar_usable.append(i)
    ar_idx = np.random.default_rng(1).choice(ar_usable, min(args.n_ar_train, len(ar_usable)),
                                             replace=False).tolist()
    log.info(f"{len(test_idx)} test stars scored | {len(ar_idx)} train stars fit the AR(1) skyline")

    star_rows = []
    acf_rows = []
    for cell, seed, path in tqdm(runs, desc="arms", total=len(runs)):
        model, cfg = load_model(path, device)
        arm = cell.replace("exp05_", "")

        ar_traj = []
        for i in ar_idx:
            mu = encode_block(model, train_blocks[i], args.horizon, device)
            if mu is not None:
                ar_traj.append(mu.cpu().numpy())
        a_mat, b_vec = fit_ar1(ar_traj, args.ridge)

        for i in test_idx:
            mu_t = encode_block(model, test_blocks[i], args.horizon, device)
            if mu_t is None:
                continue
            mu = mu_t.cpu().numpy() # (k+1, z)
            k = mu.shape[0] - 1
            target = mu[1:]
            pers = np.repeat(mu[0:1], k, axis=0)
            clim = np.repeat(target.mean(axis=0, keepdims=True), k, axis=0)
            with torch.no_grad():
                roll = model.dynamics.rollout(mu_t[0:1], k)[0].cpu().numpy()
            ar = ar_rollout(mu[0], k, a_mat, b_vec)
            star_rows.append({
                "arm": arm, "seed": seed, "tic_id": int(test_tics[i]),
                "cls": class_map.get(int(test_tics[i]), "other"),
                "pers": float(((pers - target) ** 2).mean()),
                "clim": float(((clim - target) ** 2).mean()),
                "roll": float(((roll - target) ** 2).mean()),
                "ar": float(((ar - target) ** 2).mean()),
                "within_var": float(mu.var(axis=0).sum()),
                "star_mean_norm": float((mu.mean(axis=0) ** 2).sum()),
                "n_win": int(mu.shape[0]),
            })
            acf = trajectory_acf(mu, MAX_LAG)
            for lag in range(MAX_LAG):
                acf_rows.append({"arm": arm, "seed": seed, "cls": class_map.get(int(test_tics[i]), "other"),
                                 "lag": lag + 1, "acf": acf[lag]})

    stars = pd.DataFrame(star_rows)
    stars = stars.merge(periods, on="tic_id", how="left")
    stars.to_parquet(ROOT / args.out_stars, index=False)

    acf = pd.DataFrame(acf_rows).groupby(["arm", "seed", "cls", "lag"], as_index=False)["acf"].mean()
    acf.to_csv(ROOT / args.out_acf, index=False)

    agg = stars.groupby(["arm", "seed", "cls"])[["pers", "clim", "roll", "ar"]].mean().reset_index()
    agg["n_stars"] = stars.groupby(["arm", "seed", "cls"]).size().to_numpy()
    agg["roll_over_clim"] = agg["roll"] / agg["clim"]
    agg["ar_over_clim"] = agg["ar"] / agg["clim"]
    agg["gain_ratio"] = agg["pers"] / agg["roll"]

    # Variance budget: between-star spread of the per-star mean versus the temporal spread within a star.
    budget = []
    for (arm, seed), grp in stars.groupby(["arm", "seed"]):
        within = float(grp["within_var"].mean())
        between = float(grp["star_mean_norm"].var())
        budget.append({"arm": arm, "seed": seed, "within_var": within, "between_var": between,
                       "within_frac": within / (within + between)})
    agg = agg.merge(pd.DataFrame(budget), on=["arm", "seed"], how="left")
    agg.to_csv(ROOT / args.out, index=False)
    log.info(f"\nwrote {ROOT / args.out}, {ROOT / args.out_acf}, {ROOT / args.out_stars}")

    pd.set_option("display.width", 220)
    print()
    print(agg.groupby(["arm", "cls"])[["roll_over_clim", "ar_over_clim", "gain_ratio", "within_frac"]]
          .mean().round(4).to_string())
    print()
    print(acf.groupby(["arm", "cls", "lag"])["acf"].mean().unstack("lag").round(3).to_string())
    print("\nroll_over_clim and ar_over_clim below 1.0 mean the predictor beat an oracle constant.")
    print("If AR(1) clears 1.0 where the GRU does not, the signal exists and the GRU is the bottleneck.")


if __name__ == "__main__":
    main()
