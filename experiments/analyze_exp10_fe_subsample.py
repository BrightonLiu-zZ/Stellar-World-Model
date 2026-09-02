"""F-E (exp10 forensics) -- is small n_train CAUSAL for the fusion losses, or are those tasks different?

THE QUESTION. F1's fusion delta is negative on exactly the two smallest probes (`rgb_vs_heb` 755 train
stars, `ijspeert` 93 positives) and on `osc_giant`, and the standing reading is that 128 extra collinear
columns dilute a readout that has too few rows to afford them. That reading has never been tested: the
small probes are also astrophysically different from the big ones, so size and task are confounded.
This forensic breaks the confound the only way it can be broken without new labels -- by shrinking the
TRAIN set of three tasks that currently win, and asking whether their deltas die at the small probes'
size.

METHOD. `eb`, `pulsating` (v1 subset) and `osc_giant` (menu pool), which between them cover both
populations and both ends of the prevalence range. Train stars are subsampled STRATIFIED BY LABEL to
n_train in {160, 755, full}; the two small numbers are the actual train sizes of `ijspeert`'s positive
count regime and `rgb_vs_heb`. The TEST set is never touched, so every draw is scored on the same stars
and the delta series is directly comparable to the published one.

DECISION RULES (pre-registered in the 2026-08-29 forensics handoff, scored verbatim):
    R-E1  deltas positive at full n turn <= 0 at n ~ 755 on >= 2 of the 3 tasks --> the small-n fusion
          losses are n-driven, exp10's gate excludes small-n rows from its scored set, and the paper's
          loss rows gain a measured explanation.
    R-E2  deltas stay positive at n ~ 755 --> those probes lose for task-specific reasons and the gate
          keeps every reportable row.

ESTIMATOR. Two spreads are reported side by side and never merged: the ENCODER-seed spread (6 arms, the
convention every other table here uses) and the DRAW spread (20 resamples of the train set). At full n
there is nothing to resample, so that level runs 6 draws whose only variation is the GBM random_state,
and the CSV says so. GBM random_state is PAIRED between the features arm and the fusion arm within a
draw, which is the C3b fix; differencing unpaired nonlinear fits put a 0.0062 offset on margins of the
same size once already.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~25 min with --jobs 12):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp10_fe_subsample.py
    python experiments/analyze_exp10_fe_subsample.py --tasks eb --draws 3 --seeds 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, load_mu_cache, stacked  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import label_frame  # noqa: E402
from swm.eval.readout_sweep import fit_readout_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_fe")

source_home = repo_root / "experiments" / "exp08_menu_channel"
out_home = repo_root / "experiments" / "exp10_forensics" / "fe_subsample"
# task -> population. Both v1 tasks and osc_giant are scored on their FULL population (no keep-mask),
# which is why these three can be subsampled cleanly and `rgb_vs_heb` could not.
task_population = {"eb": "subset", "pulsating": "subset", "osc_giant": "pool"}
# The two small-n probes this forensic is trying to explain: rgb_vs_heb trains on 755 stars, and 160 is
# the order of ijspeert's positive count -- the two sizes the losses actually occur at.
n_train_levels = [160, 755, None]
families = {"linear": "logistic", "gbm": "gbm"}


def pooled(population: str, arm: str) -> dict:
    """One arm's mean-pooled mu, in the (tics, one-row blocks) layout the feature aligner expects."""
    sub = "subset_mu_cache" if population == "subset" else "mu_cache"
    cache = load_mu_cache(source_home / sub / f"{arm}.npz")
    table = {}
    for split in ["train", "test"]:
        tics, blocks = cache[split]
        rows = []
        for block in blocks:
            rows.append(block.mean(axis=0).reshape(1, -1))
        table[split] = (list(tics), rows)
    return table


def labels_for(task: str, tics: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Star-level 0/1 labels for one task on one population, in the cache's own star order."""
    if task in ("eb", "pulsating"):
        frame = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
                 [["tic_id", task]].drop_duplicates("tic_id").set_index("tic_id"))
        column = frame[task]
    else:
        column = label_frame()[task]
    out = {}
    for split in ["train", "test"]:
        values = column.reindex(tics[split])
        assert values.notna().all(), f"{task}/{split}: label table does not cover the mu cache stars"
        out[split] = values.to_numpy(dtype=int)
    return out


def stratified_draw(y: np.ndarray, n_train: int, draw: int) -> np.ndarray:
    """
    Row indices of one train subsample that keeps the task's prevalence.
    Stratifying matters more than usual here: at n=160 an unstratified draw of a 6 % prevalence task
    can land with a handful of positives, and the resulting spread would be about the draw, not about n.
    """
    rng = np.random.default_rng(1000 + draw)
    keep = []
    for value in [0, 1]:
        pool = np.flatnonzero(y == value)
        take = int(round(n_train * len(pool) / len(y)))
        take = max(1, min(take, len(pool)))
        keep.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(keep))


def one_cell(x_feat_tr, x_mu_tr, y_tr, x_feat_te, x_mu_te, y_te, readout: str,
             random_state: int) -> tuple[float, float]:
    """
    Score the two arms of one (task, n, draw, encoder seed, family) cell on the identical rows.
    Returns (features_only PR-AUC, features (+) mu PR-AUC); the caller differences them, so the GBM
    random_state is shared between the arms by construction rather than by a later join.
    """
    base = fit_readout_scores(readout, x_feat_tr, y_tr, x_feat_te, random_state)
    fused_tr = np.concatenate([x_feat_tr, x_mu_tr], axis=1)
    fused_te = np.concatenate([x_feat_te, x_mu_te], axis=1)
    fused = fit_readout_scores(readout, fused_tr, y_tr, fused_te, random_state)
    return float(average_precision_score(y_te, base)), float(average_precision_score(y_te, fused))


def run_cell(task: str, population: str, n_train, draw: int, index: np.ndarray, seed: int, family: str,
             readout: str, x_feat_tr, x_feat_te, mu_tr, mu_te, y_tr, y_te) -> dict:
    """One (n level, draw, encoder seed, family) cell, at module level so joblib can ship it to a worker."""
    base, fused = one_cell(x_feat_tr[index], mu_tr[index], y_tr[index], x_feat_te, mu_te, y_te,
                           readout, draw)
    if n_train is None:
        kind = "gbm random_state only (full n has nothing to resample)"
    else:
        kind = "stratified train resample"
    return {"task": task, "population": population,
            "n_train_target": -1 if n_train is None else n_train,
            "n_train_actual": int(len(index)), "n_train_pos": int(y_tr[index].sum()),
            "draw": draw, "seed": seed, "family": family, "features_only": base,
            "features_plus_mu": fused, "delta": fused - base,
            "n_test": int(len(y_te)), "n_test_pos": int(y_te.sum()), "draw_kind": kind}


def task_rows(task: str, feats: dict, seeds: list[int], draws: int, jobs: int) -> pd.DataFrame:
    """Every (n level, draw, encoder seed, family) cell for one task, as long-form rows."""
    population = task_population[task]
    mu_by_seed, tics = {}, None
    aligned = None
    for seed in seeds:
        table = pooled(population, f"hann0p3_fbwd_s{seed}")
        if tics is None:
            tics = {"train": list(table["train"][0]), "test": list(table["test"][0])}
            aligned = align_features(feats, table)
        mu_by_seed[seed] = (stacked(table, "train"), stacked(table, "test"))
    x_feat_tr, x_feat_te = stacked(aligned, "train"), stacked(aligned, "test")
    y = labels_for(task, tics)

    jobs_list = []
    for n_train in n_train_levels:
        n_draws = draws if n_train is not None else len(seeds)
        for draw in range(n_draws):
            if n_train is None:
                index = np.arange(len(y["train"]))
            else:
                index = stratified_draw(y["train"], n_train, draw)
            for seed in seeds:
                for family, readout in families.items():
                    jobs_list.append((n_train, draw, index, seed, family, readout))

    payload = []
    for n_train, draw, index, seed, family, readout in jobs_list:
        mu_tr, mu_te = mu_by_seed[seed]
        payload.append(delayed(run_cell)(task, population, n_train, draw, index, seed, family, readout,
                                         x_feat_tr, x_feat_te, mu_tr, mu_te, y["train"], y["test"]))
    results = Parallel(n_jobs=jobs)(tqdm(payload, desc=f"cells[{task}]", total=len(payload)))
    return pd.DataFrame(results)


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Per (task, n level, family) mean delta with the encoder-seed and draw spreads kept apart."""
    rows = []
    for (task, n_target, family), group in probe.groupby(["task", "n_train_target", "family"], sort=False):
        per_draw = group.groupby("draw")["delta"].mean()
        per_seed = group.groupby("seed")["delta"].mean()
        deltas = group["delta"].to_numpy(dtype=float)
        seed_sd = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan
        rows.append({
            "task": task, "n_train_target": n_target,
            "n_train_actual": int(group["n_train_actual"].median()),
            "n_train_pos": int(group["n_train_pos"].median()), "family": family,
            "n_cells": len(deltas), "delta_mean": float(deltas.mean()),
            "draw_sd": float(per_draw.std(ddof=1)) if len(per_draw) > 1 else np.nan,
            "seed_sd": seed_sd,
            "seed_2se": 2 * seed_sd / np.sqrt(len(per_seed)) if len(per_seed) > 1 else np.nan,
            "delta_q05": float(np.quantile(deltas, 0.05)), "delta_q95": float(np.quantile(deltas, 0.95)),
            "frac_draws_positive": float((per_draw > 0).mean()),
            "n_test": int(group["n_test"].max()), "n_test_pos": int(group["n_test_pos"].max()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="F-E: is small n_train causal for the fusion losses?")
    ap.add_argument("--tasks", nargs="+", default=list(task_population), choices=list(task_population))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else out_home
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {}
    if any(task_population[t] == "subset" for t in args.tasks):
        feats["subset"] = cached_subset_features(
            repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    if any(task_population[t] == "pool" for t in args.tasks):
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        feats["pool"] = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                             repo_root / "processed" / "sequences", None)

    frames = []
    for task in args.tasks:
        frames.append(task_rows(task, feats[task_population[task]], args.seeds, args.draws, args.jobs))
    probe = pd.concat(frames, ignore_index=True)
    probe.to_csv(out_dir / "fe_probe.csv", index=False)

    summary = summarize(probe)
    summary.to_csv(out_dir / "fe_summary.csv", index=False)
    print("\nF-E fusion delta by train-set size (mean over draws x encoder seeds):")
    print(summary.pivot_table(index=["task", "family"], columns="n_train_target",
                              values="delta_mean").round(4).to_string())
    print("\nfraction of draws with a positive delta:")
    print(summary.pivot_table(index=["task", "family"], columns="n_train_target",
                              values="frac_draws_positive").round(3).to_string())
    log.info(f"wrote {out_dir}/fe_{{probe,summary}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
