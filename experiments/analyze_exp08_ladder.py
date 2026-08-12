"""exp08 ladder gates: paired probe deltas along the dynamics ladder, against the exp07 arms.

The ladder cells (smooth / linear / frozen, plus the calibration arms smooth_lo / frozen_lo when
they exist) were scored by analyze_exp07_diagnostics.py --stages stars with the exact estimator
that produced exp07_aux_gap_6seed.csv, so the exp07 hann0p3 off/fbwd rows can be used as the two
ends of the ladder without re-scoring them. Everything here is paired by seed and quoted per
pooling; nothing is pooled across arms (F21 - the arm axis IS the experiment).

Gates (manifest exp08_dynamics_ladder.yaml, pre-registered):
  G-prior  smooth - off    > 2*SE paired, on eb AND rotation, mean_resid pooling
  G-gru    fwd_bwd - smooth > 2*SE paired, on eb OR rotation, mean_resid pooling
The gate smoothness arm is exp08_smooth_lo if scored (the max-pressure-without-collapse arm added
after the dose gate measured saturation at lambda 270), else exp08_smooth.

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp08_ladder.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

TASKS = ("pulsating", "eb", "rotation", "transit")
POOLINGS = ("mean", "mean_resid", "mean_std")
GATE_POOLING = "mean_resid"

# ladder order, off -> fwd_bwd; exp07 arms carry their published names
EXP07_GAP = ROOT / "experiments" / "exp07_aux_gap_6seed.csv"
# the first 24 runs and the _lo calibration arms were scored in two invocations of the stars stage,
# so the summary exists as two files with one schema; concat, drop the duplicated untrained row
EXP08_SUMMARIES = [ROOT / "experiments" / "exp08_diag_probe_summary.csv",
                   ROOT / "experiments" / "exp08_diag_lo_probe_summary.csv"]
OUT = ROOT / "experiments" / "exp08_ladder_gap.csv"

LADDER = ["exp07_hann0p3_off", "exp08_smooth_lo", "exp08_smooth", "exp08_smooth_half",
          "exp08_linear", "exp08_frozen_lo", "exp08_frozen", "exp07_hann0p3_fbwd"]


def load_scores() -> pd.DataFrame:
    exp07 = pd.read_csv(EXP07_GAP)
    exp07 = exp07[exp07["arm"] == "trained"][["cell", "seed", "pooling", "task", "pr_auc"]]
    exp07 = exp07[exp07["cell"].isin(["exp07_hann0p3_off", "exp07_hann0p3_fbwd"])]
    parts = [pd.read_csv(p)[["cell", "seed", "pooling", "task", "pr_auc"]]
             for p in EXP08_SUMMARIES if p.exists()]
    exp08 = pd.concat(parts, ignore_index=True).drop_duplicates(["cell", "seed", "pooling", "task"])
    return pd.concat([exp07, exp08], ignore_index=True)


def paired_delta(scores: pd.DataFrame, cell_a: str, cell_b: str, task: str, pooling: str) -> dict | None:
    """Seed-paired cell_a - cell_b; both sides' seed spread enters the SE (F17)."""
    a = scores[(scores.cell == cell_a) & (scores.task == task) & (scores.pooling == pooling)]
    b = scores[(scores.cell == cell_b) & (scores.task == task) & (scores.pooling == pooling)]
    merged = a.merge(b, on="seed", suffixes=("_a", "_b"))
    merged = merged[merged["seed"] >= 0] # the untrained row carries seed -1 and pairs with nothing
    if len(merged) < 2:
        return None
    d = merged["pr_auc_a"] - merged["pr_auc_b"]
    return {"cell_a": cell_a, "cell_b": cell_b, "task": task, "pooling": pooling,
            "n_seeds": len(d), "delta": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "mean_a": float(merged["pr_auc_a"].mean()), "mean_b": float(merged["pr_auc_b"].mean())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scores = load_scores()
    cells = [c for c in LADDER if c in set(scores["cell"])]
    log.info(f"ladder cells scored: {cells}")

    # absolute table, seed means per pooling - the selection quantity (F4: absolute, never the gap)
    absolute = (scores[scores.seed >= 0].groupby(["cell", "pooling", "task"])["pr_auc"]
                .agg(["mean", "std", "count"]).reset_index())
    untrained = scores[scores.cell == "untrained_w256"]

    rows = []
    for pooling in POOLINGS:
        for task in TASKS:
            for cell in cells:
                if cell == "exp07_hann0p3_off":
                    continue
                for ref in ("exp07_hann0p3_off", "exp07_hann0p3_fbwd"):
                    if cell == ref:
                        continue
                    d = paired_delta(scores, cell, ref, task, pooling)
                    if d is not None:
                        rows.append(d)
    deltas = pd.DataFrame(rows)
    deltas.to_csv(OUT, index=False)
    log.info(f"wrote {OUT} ({len(deltas)} rows)")

    smooth_arm = "exp08_smooth_lo" if "exp08_smooth_lo" in cells else "exp08_smooth"
    log.info(f"\n=== gates on {GATE_POOLING}, smoothness arm = {smooth_arm} ===")
    for task in ("eb", "rotation"):
        g = deltas[(deltas.cell_a == smooth_arm) & (deltas.cell_b == "exp07_hann0p3_off")
                   & (deltas.task == task) & (deltas.pooling == GATE_POOLING)].iloc[0]
        log.info(f"G-prior {task}: smooth-off {g.delta:+.4f} +/- {g.se:.4f} "
                 f"({'>2SE' if g.delta > 2 * g.se else 'ns'})")
    for task in ("eb", "rotation"):
        g = deltas[(deltas.cell_a == smooth_arm) & (deltas.cell_b == "exp07_hann0p3_fbwd")
                   & (deltas.task == task) & (deltas.pooling == GATE_POOLING)].iloc[0]
        log.info(f"G-gru {task}: fbwd-smooth {-g.delta:+.4f} +/- {g.se:.4f} "
                 f"({'>2SE' if -g.delta > 2 * g.se else 'ns'})")

    log.info(f"\n=== ladder absolute ({GATE_POOLING}, seed mean +/- sd) ===")
    for task in TASKS:
        parts = []
        for cell in cells:
            r = absolute[(absolute.cell == cell) & (absolute.pooling == GATE_POOLING) & (absolute.task == task)]
            if len(r):
                parts.append(f"{cell.replace('exp07_hann0p3_', '').replace('exp08_', '')} "
                             f"{r['mean'].iloc[0]:.3f}±{r['std'].iloc[0]:.3f}")
        u = untrained[(untrained.pooling == GATE_POOLING) & (untrained.task == task)]
        floor = f" | untrained {u['pr_auc'].iloc[0]:.3f}" if len(u) else ""
        log.info(f"{task}: " + " -> ".join(parts) + floor)


if __name__ == "__main__":
    main()
