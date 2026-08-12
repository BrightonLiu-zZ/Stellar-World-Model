"""exp07 pre-check C4: does the lambda ~= 60 starting dial transfer to the log-PSD aux recipe?

The exp06 curve-forensics addendum backed out lambda_needed = lambda * target / dose_steady at
49 / 67 / 65 / 53 for the four fwd+bwd comb cells, a narrow band despite recon and dyn each rescaling
with window. Every one of those cells runs the `combined` aux at weight 0.3. exp07 puts log_psd cells
on the same dynamics axis, and lambda is calibrated against the recon scale, so the open question is
whether a different aux recipe shifts that scale enough to move the dial.

Two measurements, both from the dumped W&B histories already on disk (no GPU, no checkpoints):

  dose        steady-state (last-10-epoch) lambda * train/dyn / train/recon for exp05_lpsd_multi_c1p0
              (lambda 32, target contribution 1.0), per seed and seed-averaged, then the back-out
              lambda_needed = lambda * target / dose_steady. The comb fwd+bwd cell at the same
              geometry is carried alongside as an estimator check: it must reproduce the addendum's 49.

  recon scale steady-state train/recon of exp05_lpsd_off vs exp05_comb_off. Both are lambda = 0, so
              the only difference is the aux recipe (log_psd @ 0.1 + free_bits 0.02 vs combined @ 0.3 +
              free_bits 0.0). A ratio near 1 means lambda calibrated on comb transfers as a dial;
              a ratio far from 1 means the lpsd cells need their own pilot.

The dose estimator is deliberately identical to notebook exp06_diagnostics section I4 (mean the
per-seed dose trajectories, then average the last 10 epochs) so the numbers are comparable to the
addendum's table rather than merely similar.

Writes experiments/exp07_c4_lambda_band.csv. Run (repo root, swm env):
    python experiments/analyze_exp07_c4_lambda_band.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "experiments" / "exp05_forensics" / "curves_exp05"

STEADY_EPOCHS = 10 # last-N-epoch window defining "steady state", matching notebook I4

# lambda and target contribution as launched (experiments/run_exp05_train_sweep.ps1 lambda table).
# The only lpsd dynamics cell exp05 ran is multistep, so comb_multi_c1p0 is the aux-recipe-matched
# comparison (same dyn_mode, same target, different aux); comb_fbwd_c1p0 is the fwd+bwd anchor the
# addendum's lambda ~= 60 dial is quoted from, carried as an estimator check against its published 49.
DOSE_CELLS = {
    "exp05_lpsd_multi_c1p0": {"lam": 32, "target": 1.0, "aux": "log_psd@0.1", "dyn_mode": "multistep"},
    "exp05_comb_multi_c1p0": {"lam": 48, "target": 1.0, "aux": "combined@0.3", "dyn_mode": "multistep"},
    "exp05_comb_fbwd_c1p0": {"lam": 66, "target": 1.0, "aux": "combined@0.3", "dyn_mode": "fwd_bwd"},
}
# lambda = 0 cells: the recon scale that lambda is calibrated against, with the dynamics term removed.
RECON_CELLS = {
    "exp05_lpsd_off": "log_psd@0.1",
    "exp05_comb_off": "combined@0.3",
}


def load_curve(cell: str, seed: int, variant: str) -> pd.DataFrame:
    """Load one run's dumped W&B history, indexed by epoch so seeds align before averaging."""
    path = CURVES / f"{cell}_{variant}_seed{seed}.csv"
    assert path.exists(), f"missing curve {path}"
    curve = pd.read_csv(path).set_index("epoch").sort_index()
    return curve


def seed_mean_series(cell: str, seeds: list[int], variant: str, column: str) -> pd.Series:
    """Average one logged metric across seeds at each epoch (the I4 order: mean first, steady after)."""
    per_seed = []
    for seed in seeds:
        per_seed.append(load_curve(cell, seed, variant)[column])
    return pd.concat(per_seed, axis=1).mean(axis=1)


def steady(series: pd.Series) -> float:
    """Mean of a per-epoch series over its last STEADY_EPOCHS epochs."""
    tail = series[series.index >= series.index.max() - (STEADY_EPOCHS - 1)]
    return float(tail.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="exp07 C4: lambda-band transfer across aux recipes")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--out", default="experiments/exp07_c4_lambda_band.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = []
    for cell, spec in DOSE_CELLS.items():
        lam = float(spec["lam"])
        per_seed_dose = {}
        for seed in args.seeds:
            curve = load_curve(cell, seed, args.variant)
            dose = lam * curve["train/dyn"] / curve["train/recon"] # achieved contribution, per epoch
            per_seed_dose[seed] = steady(dose)
        dose_series = lam * seed_mean_series(cell, args.seeds, args.variant, "train/dyn") / \
            seed_mean_series(cell, args.seeds, args.variant, "train/recon")
        dose_steady = steady(dose_series)
        rows.append({
            "cell": cell, "aux": spec["aux"], "dyn_mode": spec["dyn_mode"], "lam": lam,
            "target": spec["target"],
            "dose_ep7": float(dose_series.loc[7]), # the epoch the exp05/exp06 pilots read out at
            "dose_steady": dose_steady,
            "dose_steady_sd_over_seeds": float(np.std(list(per_seed_dose.values()), ddof=1)),
            "dose_steady_min": float(min(per_seed_dose.values())),
            "dose_steady_max": float(max(per_seed_dose.values())),
            "lambda_needed": lam * float(spec["target"]) / dose_steady,
            "recon_steady": steady(seed_mean_series(cell, args.seeds, args.variant, "train/recon")),
            "dyn_steady": steady(seed_mean_series(cell, args.seeds, args.variant, "train/dyn")),
            "aux_steady": steady(seed_mean_series(cell, args.seeds, args.variant, "train/aux")),
        })

    recon_scale = {}
    for cell, aux in RECON_CELLS.items():
        recon_scale[cell] = steady(seed_mean_series(cell, args.seeds, args.variant, "train/recon"))
        rows.append({
            "cell": cell, "aux": aux, "dyn_mode": "off", "lam": 0.0, "target": 0.0,
            "dose_ep7": np.nan, "dose_steady": np.nan, "dose_steady_sd_over_seeds": np.nan,
            "dose_steady_min": np.nan, "dose_steady_max": np.nan, "lambda_needed": np.nan,
            "recon_steady": recon_scale[cell],
            "dyn_steady": steady(seed_mean_series(cell, args.seeds, args.variant, "train/dyn")),
            "aux_steady": steady(seed_mean_series(cell, args.seeds, args.variant, "train/aux")),
        })

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / args.out, index=False)

    # The back-out is confounded with lambda itself (raising lambda suppresses dyn), and the two
    # multistep cells ran at different lambdas, so the lambda = 0 cells give the unconfounded dial:
    # lambda = target * recon / dyn evaluated where the dynamics term exerts no pressure at all.
    off = table.set_index("cell")
    recon_ratio = off.loc["exp05_lpsd_off", "recon_steady"] / off.loc["exp05_comb_off", "recon_steady"]
    dyn_ratio = off.loc["exp05_lpsd_off", "dyn_steady"] / off.loc["exp05_comb_off", "dyn_steady"]
    dial_ratio_off = recon_ratio / dyn_ratio
    multi = table.set_index("cell")
    dial_ratio_backout = (multi.loc["exp05_lpsd_multi_c1p0", "lambda_needed"]
                          / multi.loc["exp05_comb_multi_c1p0", "lambda_needed"])

    pd.set_option("display.width", 220)
    print()
    print(table.round(4).to_string(index=False))
    print(f"\nrecon-scale ratio lpsd_off / comb_off at steady state: {recon_ratio}")
    print(f"dyn-scale ratio lpsd_off / comb_off at steady state: {dyn_ratio}")
    print(f"implied lambda dial ratio lpsd / comb, from the lambda=0 cells: {dial_ratio_off}")
    print(f"implied lambda dial ratio lpsd / comb, from the matched multistep back-out: {dial_ratio_backout}")
    print("lambda is calibrated as target * recon / dyn, so a ratio near 1 means the comb-derived")
    print("dial transfers to the log_psd cells; far from 1 means the lpsd arm needs its own pilot.")
    log.info(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
