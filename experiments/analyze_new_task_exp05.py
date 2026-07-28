"""Did the exp05 dynamics-weighting gain TRANSFER off the four v1 tasks? (plan 2026-07-25)

exp05 established criterion 1 -- dynamics-weighted SSL beats dynamics-off SSL -- on transit/eb/
pulsating/rotation only. The downstream menu (asteroseismic block + flare + P_rot + Ijspeert) was last
scored on the *exp03* leader recipe and has never seen exp05. This script asks whether the lambda gain
carries over, using the same paired contrast exp05 itself used.

Three statistics per task (decision 10 of the plan):
  delta_lambda  pr_auc(comb_fbwd_c1p0 seed i) - pr_auc(comb_off seed i), paired within seed,
                mean +/- SD over 4 seeds, confirmed when mean > 2*SE with SE = SD/sqrt(4).
                Per-seed values are ALWAYS printed -- per-seed transparency is what caught the
                exp04 pulsating fluke.
  gap           each cell minus the shared capacity-matched untrained arm (the older, weaker claim).
  boot_ci       for probes with n_test < BOOT_N_MAX, a paired star-level bootstrap 95% CI on the
                pooled-seed delta. Seed SE measures run-to-run noise only; on a 138-star probe the
                sampling noise dominates and must be quantified separately. A claim stands only if
                BOTH the seed gate and the bootstrap CI exclude zero.

Run (swm env, from repo root, PYTHONPATH=src; CPU-only, seconds):
    python experiments/analyze_new_task_exp05.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("analyze_new_task_exp05")

repo_root = Path(__file__).resolve().parent.parent
RESULT_DIR = repo_root / "experiments" / "new_task_exp05"
OFF_CELL = "comb_off"
DYN_CELL = "comb_fbwd_c1p0"
SEEDS = (0, 1, 2, 3)
BOOT_N_MAX = 400   # below this test-set size the seed SE alone is not an honest error bar
BOOT_POS_MAX = 100  # ...and a wide test set with very few POSITIVES needs it just as much
                    # (ijspeert_excl_villanova: 1,825 test stars but only ~19 positives)
N_BOOT = 2000
POOLING = "mean"   # the v1 headline protocol; window_score rows are a byproduct, not the claim

# Which metric is the headline for each task shape, matching how the scorecard reports it.
HEADLINE = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}


def _metric(shape: str, y: np.ndarray, score: np.ndarray) -> float:
    """Recompute a task's headline metric from raw predictions (used inside the bootstrap)."""
    if shape == "detection":
        if y.min() == y.max():
            return np.nan  # a resample with one class carries no information; dropped by nanpercentile
        return float(average_precision_score(y, score))
    if shape == "contrastive":
        if y.min() == y.max():
            return np.nan
        return float(roc_auc_score(y, score))
    return float(r2_score(y, score))


def bootstrap_delta(dumps: pd.DataFrame, task: str, shape: str, rng: np.random.Generator) -> tuple[float, float]:
    """Paired star-level bootstrap 95% CI on the seed-averaged (dyn - off) delta for one task.

    The SAME resampled star set is scored for every arm, so the CI reflects sampling noise in the star
    population, not independent noise per arm -- that pairing is the whole point.
    """
    frames = {}
    for seed in SEEDS:
        for cell in (OFF_CELL, DYN_CELL):
            sub = dumps[(dumps["arm"] == f"{cell}_s{seed}") & (dumps["task"] == task)
                        & (dumps["pooling"] == POOLING)]
            if sub.empty:
                return np.nan, np.nan
            frames[(cell, seed)] = sub.sort_values("tic_id").reset_index(drop=True)

    reference = frames[(OFF_CELL, SEEDS[0])]
    n = len(reference)
    # Every arm scores the same star list in the same order, so one index array indexes all of them.
    for frame in frames.values():
        assert len(frame) == n, f"{task}: arms disagree on test-set size ({len(frame)} vs {n})"
        assert np.array_equal(frame["tic_id"].to_numpy(), reference["tic_id"].to_numpy()), \
            f"{task}: arms disagree on test-set membership"

    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        per_seed = []
        for seed in SEEDS:
            off = frames[(OFF_CELL, seed)]
            dyn = frames[(DYN_CELL, seed)]
            m_off = _metric(shape, off["y"].to_numpy()[idx], off["score"].to_numpy()[idx])
            m_dyn = _metric(shape, dyn["y"].to_numpy()[idx], dyn["score"].to_numpy()[idx])
            per_seed.append(m_dyn - m_off)
        deltas[b] = np.nanmean(per_seed)
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def main() -> int:
    score_path = RESULT_DIR / "new_task_scorecard.csv"
    dump_path = RESULT_DIR / "per_star_scores.parquet"
    assert score_path.exists(), f"missing {score_path} -- run swm.eval.new_task_scorecard first"
    scores = pd.read_csv(score_path)
    scores = scores[scores["pooling"] == POOLING]
    dumps = pd.read_parquet(dump_path) if dump_path.exists() else None
    if dumps is None:
        log.warning(f"no {dump_path}: bootstrap CIs will be NaN (re-run the scorecard with --dump-scores)")

    ceiling_path = RESULT_DIR / "ceiling_A1A2.csv"
    ceilings = pd.read_csv(ceiling_path) if ceiling_path.exists() else pd.DataFrame()

    rng = np.random.default_rng(0)  # fixed: this is an inference statistic, not a display sample
    rows = []
    for (task, shape), group in scores.groupby(["task", "shape"], sort=False):
        metric = HEADLINE[shape]
        by_arm = group.set_index("arm")[metric]
        per_seed = {}
        for seed in SEEDS:
            off_arm, dyn_arm = f"{OFF_CELL}_s{seed}", f"{DYN_CELL}_s{seed}"
            if off_arm in by_arm.index and dyn_arm in by_arm.index:
                per_seed[seed] = float(by_arm[dyn_arm]) - float(by_arm[off_arm])
        if not per_seed:
            log.warning(f"{task}: no paired seeds found, skipping")
            continue
        deltas = np.array(list(per_seed.values()))
        sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else np.nan
        two_se = 2 * sd / np.sqrt(len(deltas)) if len(deltas) > 1 else np.nan

        n_test = int(group["n_test"].max())
        n_pos = int(group["n_test_pos"].max()) if group["n_test_pos"].notna().any() else n_test
        small = (n_test < BOOT_N_MAX) or (n_pos < BOOT_POS_MAX)
        lo, hi = (np.nan, np.nan)
        if dumps is not None and small:
            lo, hi = bootstrap_delta(dumps, task, shape, rng)

        untrained = float(by_arm["untrained"]) if "untrained" in by_arm.index else np.nan
        dyn_mean = float(np.mean([by_arm[f"{DYN_CELL}_s{s}"] for s in per_seed]))
        off_mean = float(np.mean([by_arm[f"{OFF_CELL}_s{s}"] for s in per_seed]))
        row = {
            "task": task, "shape": shape, "metric": metric, "n_test": n_test, "n_test_pos": n_pos,
            "n_seeds": len(deltas),
            "delta_lambda_mean": float(deltas.mean()), "delta_sd": sd, "delta_2se": two_se,
            "confirm_seed_gate": bool(len(deltas) > 1 and deltas.mean() > two_se),
            "boot_lo": lo, "boot_hi": hi,
            "confirm_bootstrap": bool(np.isfinite(lo) and lo > 0),
            "dyn_mean": dyn_mean, "off_mean": off_mean, "untrained": untrained,
            "gap_dyn_vs_untrained": dyn_mean - untrained,
            "gap_off_vs_untrained": off_mean - untrained,
        }
        for seed in SEEDS:
            row[f"delta_s{seed}"] = per_seed.get(seed, np.nan)
        if not ceilings.empty:
            for arm in ("A1_feats", "A2_feats"):
                cell = ceilings[(ceilings["task"] == task) & (ceilings["arm"] == arm)]
                row[arm] = float(cell[metric].iloc[0]) if len(cell) else np.nan
                row[f"beats_{arm}"] = bool(np.isfinite(row[arm]) and dyn_mean > row[arm])
            # The features win outright, so "beats_A1" is False everywhere and says little. The
            # informative quantity is how much of the lambda=0 -> A1 shortfall the dynamics weight closes.
            shortfall = row["A1_feats"] - off_mean
            row["frac_gap_to_A1_closed"] = (
                float((dyn_mean - off_mean) / shortfall) if np.isfinite(shortfall) and shortfall > 0 else np.nan)
        rows.append(row)

    result = pd.DataFrame(rows)
    # A claim needs the seed gate AND, where the probe is small, the bootstrap too.
    needs_boot = (result["n_test"] < BOOT_N_MAX) | (result["n_test_pos"] < BOOT_POS_MAX)
    result["claimable"] = result["confirm_seed_gate"] & (~needs_boot | result["confirm_bootstrap"])
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    out_path = RESULT_DIR / "transfer_summary.csv"
    result.to_csv(out_path, index=False)

    show = ["task", "n_test", "delta_lambda_mean", "delta_2se", "confirm_seed_gate",
            "boot_lo", "boot_hi", "claimable", "gap_dyn_vs_untrained"]
    log.info("transfer summary (delta = fbwd_c1p0 - off, paired per seed)\n"
             + result[show].round(4).to_string(index=False))
    log.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
