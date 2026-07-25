"""MIL (window_score x logistic) 3-seed confirm on cached exp03 leader-arm mu (handoff task 3).

Feeds ADR-0008-lite. The exp04 leader fan reported the MIL second-protocol eb/transit advantage at
seed 0 only (eb +0.123 winner, +0.097 comb). This asks whether that advantage holds across 3 seeds,
reusing readout_sweep's exact scoring on the first-segment window-mu caches already on disk for
fb0_b0p1_comb (the promoted reference recipe) and fb0p02_b0p1_lpsd (the exp03 winner), seeds 0/1/2,
checkpoint best_recon_aux, v1 labels.

Two reported statistics (grill 2026-07-22):
  #1 primary MIL gap vs the capacity-matched untrained MIL reference, mean +/- SD over 3 seeds,
     confirmed when mean > 2*SE with SE = SD/sqrt(3); the exp04 H-confirm convention (sample SD).
  #2 paired (window_score - mean) gap delta per seed: is MIL the better eval shape than mean pooling.

CPU-only and self-contained: importing readout_sweep pulls in score_cells/cached_mu without touching
its main() CUDA assert, and every mu block is already cached, so no encoder pass runs. Writes
experiments/exp04_mil_confirm.{csv,md} and deliberately does NOT append to any per-arm
readout_sweep.csv (a parallel session may be writing those).

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_mil_confirm.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from swm.eval.readout_sweep import cached_mu, score_cells

log = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parent.parent
arms = {
    "exp03_fb0_b0p1_comb": "comb (reference recipe)",
    "exp03_fb0p02_b0p1_lpsd": "winner (exp03)",
}
seeds = (0, 1, 2)
tasks = ("pulsating", "eb", "rotation", "transit")
poolings = ("window_score", "mean")  # window_score = MIL; mean = the v1 headline it is paired against
ckpt = "best_recon_aux"
untrained_cache = repo_root / "experiments" / "exp03_eval_cache" / "untrained_mu_w256.npz"
window = 256  # nominal: every cache exists, so cached_mu never re-encodes and score_cells is z-only
gate_seed0_eb = {"exp03_fb0_b0p1_comb": 0.0973, "exp03_fb0p02_b0p1_lpsd": 0.1228}  # leader-fan reference


def score_arm_gaps(mu: dict, subset: pd.DataFrame, untrained_pr_auc: pd.Series, label: str) -> pd.DataFrame:
    """Score one encoder arm's window_score+mean logistic cells and gap them against the untrained ref.
    Returns one row per (pooling, task) with pr_auc, the matched untrained pr_auc, and their gap."""
    cells = score_cells(mu, subset, tasks, ("logistic",), poolings, label)
    keys = list(zip(cells["pooling"], cells["task"]))
    cells["pr_auc_untrained"] = untrained_pr_auc.reindex(keys).to_numpy()
    cells["gap"] = cells["pr_auc"] - cells["pr_auc_untrained"]
    return cells[["pooling", "task", "pr_auc", "pr_auc_untrained", "gap"]]


def fmt(x: float) -> str:
    """Signed 3-decimal string matching the exp04 H-confirm README house style (e.g. +0.066)."""
    return f"{x:+.3f}"


def confirm_table(agg: pd.DataFrame, pooling: str) -> str:
    """Render the exp04-README gap grid (row per arm, column per task) for one pooling, with +/-SD and
    a ✓/✗ flag where the 3-seed mean clears 2*SE."""
    lines = ["| cell | pulsating | eb | rotation | transit |", "|---|---|---|---|---|"]
    for arm in arms:
        cells = []
        for task in tasks:
            row = agg[(agg["arm"] == arm) & (agg["pooling"] == pooling) & (agg["task"] == task)].iloc[0]
            flag = "✓" if row["confirm"] else "✗"
            cells.append(f"{fmt(row['gap_mean'])} ± {row['gap_sd']:.3f} {flag}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(agg: pd.DataFrame) -> str:
    """Render the paired (window_score - mean) gap delta per arm/task, mean over seeds."""
    lines = ["| cell | pulsating | eb | rotation | transit |", "|---|---|---|---|---|"]
    for arm in arms:
        cells = []
        for task in tasks:
            row = agg[(agg["arm"] == arm) & (agg["pooling"] == "window_score") & (agg["task"] == task)].iloc[0]
            cells.append(f"{fmt(row['delta_mean'])} ± {row['delta_sd']:.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    packed_dir = repo_root / "experiments" / list(arms)[0] / "packed"

    # untrained arm: geometry-shared (z128), scored once, every trained seed gaps against it
    mu_untrained = cached_mu(untrained_cache, None, packed_dir, window, "cpu", "untrained")
    untrained_cells = score_cells(mu_untrained, subset, tasks, ("logistic",), poolings, "untrained")
    untrained_pr_auc = untrained_cells.set_index(["pooling", "task"])["pr_auc"]

    long_rows = []
    for arm in arms:
        for seed in seeds:
            cache = repo_root / "experiments" / arm / "models" / f"B_seed{seed}" / "extracted" / f"first_segment_window_mu_{ckpt}.npz"
            assert cache.exists(), f"missing mu cache: {cache}"
            mu = cached_mu(cache, None, repo_root / "experiments" / arm / "packed", window, "cpu", f"{arm}:seed{seed}")
            gaps = score_arm_gaps(mu, subset, untrained_pr_auc, f"{arm} seed{seed}")
            gaps["arm"] = arm
            gaps["seed"] = seed
            long_rows.append(gaps)
    long = pd.concat(long_rows, ignore_index=True)

    # validation gate: recomputed seed-0 window_score eb must reproduce the leader-fan headline
    for arm in arms:
        got = long[(long["arm"] == arm) & (long["seed"] == 0) & (long["pooling"] == "window_score")
                   & (long["task"] == "eb")]["gap"].iloc[0]
        assert abs(got - gate_seed0_eb[arm]) < 2e-3, f"{arm} seed0 eb gap {got} != leader-fan {gate_seed0_eb[arm]}"
    log.info("validation gate passed: seed-0 window_score eb reproduces the leader fan")

    # #2 paired delta: window_score gap minus mean gap, per arm/task/seed
    pivot = long.pivot_table(index=["arm", "task", "seed"], columns="pooling", values="gap").reset_index()
    pivot["delta"] = pivot["window_score"] - pivot["mean"]
    delta_of = {}
    for row in pivot.itertuples(index=False):
        delta_of[(row.arm, row.task, row.seed)] = row.delta

    # aggregate both poolings across seeds; #1 confirm flag on the raw gap, #2 on the paired delta
    agg_rows = []
    for arm in arms:
        for pooling in poolings:
            for task in tasks:
                sub = long[(long["arm"] == arm) & (long["pooling"] == pooling) & (long["task"] == task)].sort_values("seed")
                gap_by_seed = sub.set_index("seed")["gap"]
                gaps = gap_by_seed.to_numpy()
                gap_mean = float(gaps.mean())
                gap_sd = float(gaps.std(ddof=1))  # sample SD, matches the exp04 H-confirm table
                se = gap_sd / np.sqrt(len(seeds))
                deltas = np.array([delta_of[(arm, task, s)] for s in seeds], dtype=float)
                agg_rows.append({
                    "arm": arm, "pooling": pooling, "task": task,
                    "gap_s0": float(gap_by_seed.get(0, np.nan)), "gap_s1": float(gap_by_seed.get(1, np.nan)),
                    "gap_s2": float(gap_by_seed.get(2, np.nan)),
                    "gap_mean": gap_mean, "gap_sd": gap_sd, "gap_2se": float(2 * se),
                    "confirm": bool(gap_mean > 2 * se),
                    "delta_s0": float(deltas[0]), "delta_s1": float(deltas[1]), "delta_s2": float(deltas[2]),
                    "delta_mean": float(deltas.mean()), "delta_sd": float(deltas.std(ddof=1)),
                })
    agg = pd.DataFrame(agg_rows)

    long.to_csv(repo_root / "experiments" / "exp04_mil_confirm_perseed.csv", index=False)
    agg.to_csv(repo_root / "experiments" / "exp04_mil_confirm.csv", index=False)

    ws_eb = agg[(agg["pooling"] == "window_score") & (agg["task"] == "eb")]
    headline = []
    for arm in arms:
        row = ws_eb[ws_eb["arm"] == arm].iloc[0]
        headline.append(f"{arm} eb {fmt(row['gap_mean'])} ± {row['gap_sd']:.3f} ({'✓' if row['confirm'] else '✗'})")
    md = f"""# exp04 MIL (window_score x logistic) 3-seed confirm

Handoff task 3, grill 2026-07-22. Does the exp04 seed-0 leader-fan MIL advantage (eb +0.123 winner,
+0.097 comb) survive 3 seeds? Scored by `experiments/analyze_mil_confirm.py` on the cached
first-segment window-mu (`{ckpt}`, v1 labels), reusing `readout_sweep.score_cells` verbatim; the
untrained MIL reference is the capacity-matched z128 `exp03_eval_cache` arm. Machine-readable:
`exp04_mil_confirm.csv` (aggregated) + `exp04_mil_confirm_perseed.csv` (raw per-seed). Feeds
ADR-0008-lite; the linear-probe mean-pooling probe stays the v1 headline.

## #1 MIL gap vs untrained (gap mean +/- SD over 3 seeds; ✓ = mean > 2*SE)

{confirm_table(agg, "window_score")}

Headline: {"; ".join(headline)}.

## Mean-pooling reference (same arms/seeds, the v1 headline protocol)

{confirm_table(agg, "mean")}

## #2 MIL is the better eval shape? paired (window_score - mean) gap delta, mean +/- SD over 3 seeds

{delta_table(agg)}

## Notes

- Per-seed raw gaps are in `exp04_mil_confirm.csv` (`gap_s0/s1/s2`); showing the spread is the point
  (per-seed transparency is what caught the exp04 pulsating fluke).
- Untrained MIL reference is seed-0 only (geometry-shared), so SD reflects trained-seed variance, exactly
  as the exp04 H-confirm mean-pooling table did.
- eb is skyline-closed (exp04); MIL is an eval-shape gain, not a new SSL win.
"""
    (repo_root / "experiments" / "exp04_mil_confirm.md").write_text(md, encoding="utf-8")
    log.info("wrote experiments/exp04_mil_confirm.{csv,md} + _perseed.csv")
    print(agg[["arm", "pooling", "task", "gap_mean", "gap_sd", "confirm", "delta_mean"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
