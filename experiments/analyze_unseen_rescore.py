"""Strict unseen-pre-training re-score (Yue Ma revision plan 2026-09-15, item 1).

Question: does any downstream test star also sit in the self-supervised pre-training data, and if the
scoring is restricted to stars the encoder never saw, do the paper's fusion gains change?

Nothing is refit. The paper's per-star test predictions for every linear arm are already on disk
(`experiments/paper_bootstrap/linear_star_scores.parquet`, footing-checked against `f1_probe.csv`), so
"restrict to unseen" is a row filter followed by the same metric and the same paired star-bootstrap.

Two definitions of "seen", both reported so the choice is explicit rather than buried:
    fit        the star was in the pre-training TRAIN split (its windows shaped the encoder weights)
    fit+val    also the pre-training VAL split (used to pick the checkpoint, never for a gradient)
The five v1 tasks score the pre-training subset's own TEST split, so their overlap is zero by
construction; the script asserts that rather than assuming it. The five pool tasks score a second
population of stars, some of which do overlap.

Run (swm env, repo root, PYTHONPATH=src, CPU, ~3 min):
    python experiments/analyze_unseen_rescore.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("unseen")

PAPER_TASKS = ["pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "rgb_vs_heb",
               "numax_hon", "rotation_period", "flare"]
ARMS = {"features": ("features", "features_only"),
        "fusion": ("hann0p3_fbwd", "features_plus_mu"),
        "untrained_fusion": ("untrained", "features_plus_mu"),
        "mu": ("hann0p3_fbwd", "mu")}
SEEN_DEFINITIONS = {"fit": ("train",), "fit+val": ("train", "val")}


def metric_value(shape: str, y: np.ndarray, score: np.ndarray) -> float:
    """One arm's headline metric on one star list; NaN when the list carries a single class."""
    if shape == "regression":
        return float(r2_score(y, score))
    if y.min() == y.max():
        return np.nan
    if shape == "contrastive":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def arm_matrix(dumps: pd.DataFrame, family: str, arm_set: str, task: str, tics: np.ndarray) -> np.ndarray:
    """(n_seeds, n_star) score matrix for one arm on one task, aligned to `tics`."""
    rows = dumps[(dumps["family"] == family) & (dumps["arm_set"] == arm_set) & (dumps["task"] == task)]
    assert not rows.empty, f"no dumped predictions for {family}/{arm_set}/{task}"
    seeds = sorted(rows["seed"].unique())
    matrix = np.empty((len(seeds), len(tics)), dtype=float)  # (S, N)
    for i, seed in enumerate(seeds):
        one = rows[rows["seed"] == seed].set_index("tic_id")["score"]
        assert set(one.index) == set(tics), f"{family}/{arm_set}/{task} seed {seed}: star list differs"
        matrix[i] = one.reindex(tics).to_numpy()
    return matrix


def delta_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and 2*SE over seeds of (a - b); b is broadcast when it is seedless."""
    if len(b) == 1:
        d = a - b[0]
    else:
        assert len(a) == len(b)
        d = a - b
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
    return float(d.mean()), 2 * se


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-score the paper's linear arms on stars unseen in pre-training.")
    ap.add_argument("--in-dir", default="experiments/paper_bootstrap")
    ap.add_argument("--out-dir", default="experiments/unseen_pretraining")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dumps = pd.read_parquet(repo_root / args.in_dir / "linear_star_scores.parquet")
    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    seen_sets = {name: set(subset.loc[subset["split"].isin(splits), "tic_id"].astype(int))
                 for name, splits in SEEN_DEFINITIONS.items()}
    log.info("pre-training stars: " + ", ".join(f"{k} {len(v)}" for k, v in seen_sets.items()))
    rng = np.random.default_rng(args.seed)
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in tqdm(PAPER_TASKS, desc="tasks", total=len(PAPER_TASKS)):
        ref = dumps[(dumps["family"] == "hann0p3_fbwd") & (dumps["arm_set"] == "features_plus_mu")
                    & (dumps["task"] == task)]
        shape = ref["shape"].iloc[0]
        ref0 = ref[ref["seed"] == ref["seed"].min()].sort_values("tic_id")
        tics = ref0["tic_id"].to_numpy().astype(int)
        y = ref0["y"].to_numpy()
        matrices = {name: arm_matrix(dumps, fam, aset, task, tics) for name, (fam, aset) in ARMS.items()}
        block = "v1" if task in ("pulsating", "eb", "rotation", "transit", "rotation_period") else "pool"

        for seen_name, seen in seen_sets.items():
            keep = np.array([t not in seen for t in tics])
            if block == "v1":
                assert keep.all(), f"{task}: a v1 test star sits in the pre-training {seen_name} set"
            n_all, n_unseen = len(tics), int(keep.sum())
            n_pos_all = int(y.sum()) if shape != "regression" else -1
            n_pos_unseen = int(y[keep].sum()) if shape != "regression" else -1

            full = {n: np.array([metric_value(shape, y, m[i]) for i in range(len(m))])
                    for n, m in matrices.items()}
            sub = {n: np.array([metric_value(shape, y[keep], m[i, keep]) for i in range(len(m))])
                   for n, m in matrices.items()}

            # paired star-bootstrap of fusion - features on the unseen rows (seeds averaged per resample)
            idx_pool = np.flatnonzero(keep)
            boot = np.full(args.n_boot, np.nan)
            for b in range(args.n_boot):
                idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
                yb = y[idx]
                if shape != "regression" and yb.min() == yb.max():
                    continue
                f = np.mean([metric_value(shape, yb, matrices["fusion"][i, idx])
                             for i in range(len(matrices["fusion"]))])
                g = metric_value(shape, yb, matrices["features"][0, idx])
                boot[b] = f - g
            boot = boot[np.isfinite(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])

            d_all, se_all = delta_stats(full["fusion"], full["features"])
            d_un, se_un = delta_stats(sub["fusion"], sub["features"])
            du_all, _ = delta_stats(full["untrained_fusion"], full["features"])
            du_un, _ = delta_stats(sub["untrained_fusion"], sub["features"])
            rows.append({"task": task, "block": block, "shape": shape, "seen_definition": seen_name,
                         "n_test": n_all, "n_test_pos": n_pos_all, "n_unseen": n_unseen,
                         "n_unseen_pos": n_pos_unseen, "n_seen": n_all - n_unseen,
                         "features_all": float(full["features"].mean()),
                         "features_unseen": float(sub["features"].mean()),
                         "fusion_all": float(full["fusion"].mean()), "fusion_unseen": float(sub["fusion"].mean()),
                         "gain_all": d_all, "gain_all_2se": se_all,
                         "gain_unseen": d_un, "gain_unseen_2se": se_un,
                         "gain_unseen_ci_lo": float(lo), "gain_unseen_ci_hi": float(hi),
                         "gain_unseen_frac_pos": float((boot > 0).mean()),
                         "untrained_gain_all": du_all, "untrained_gain_unseen": du_un,
                         "beats_2se_all": bool(d_all > se_all), "beats_2se_unseen": bool(d_un > se_un)})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "unseen_rescore.csv", index=False)
    log.info(f"wrote {out_dir / 'unseen_rescore.csv'} ({len(out)} rows)")
    show = out[["task", "seen_definition", "n_test", "n_unseen", "n_seen", "gain_all", "gain_all_2se",
                "gain_unseen", "gain_unseen_2se", "gain_unseen_ci_lo", "gain_unseen_ci_hi",
                "beats_2se_all", "beats_2se_unseen"]]
    with pd.option_context("display.width", 220, "display.max_rows", 100, "display.float_format", "{:.4f}".format):
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
