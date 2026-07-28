"""
MIL pooling sweep (plan 2026-07-25): score every pooling operator on every task, for every encoder
arm and bag scope, and write one tidy long CSV the notebook turns into the beta* figure.

Protocol, as grilled:
  fit on TRAIN, tune each operator's single hyperparameter on VAL, report on TEST. The val split has
  never been used by readout_sweep or new_task_scorecard, so it is a clean selection set. Every
  arm - including the untrained reference - tunes its own hyperparameter on val, otherwise the
  trained-minus-untrained gap would just be the gap between a tuned and an untuned operator.
  Both val and test PR-AUC are written for every cell, but the WINNER IS DECLARED ON VAL ONLY; the
  test column exists so the published table is complete, not so the winner can be picked from it.

Efficiency: the unsupervised feature poolings are fitted once per arm and reused across tasks, and
the window-level readout that every score-space operator consumes is fitted once per (arm, task)
and reused across all operators and hyperparameters.

Run (swm env, from repo root, PYTHONPATH=src), after building caches with swm.eval.mil_cache:
    PYTHONUNBUFFERED=1 python -m swm.eval.mil_sweep --scope first
    PYTHONUNBUFFERED=1 python -m swm.eval.mil_sweep --scope all --kmatch 16
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from swm.eval.mil_cache import cache_path, load_cache
from swm.eval.pooling import (FeaturePooling, aggregate_scores, bag_size_features, feature_poolings,
                              pooling_grids, score_poolings, subsample_bags)
from swm.eval.readout_sweep import fit_readout_scores
from swm.eval.skyline import _git_sha

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

repo_root = Path(__file__).resolve().parents[3]
tasks_default = ("pulsating", "eb", "rotation", "transit")
new_task_tasks_default = ("osc_giant", "solar_like_osc", "flare")
# Continuous targets on the new-task pool: {task: (label column, take log10 of the target)}. These
# score with R2 instead of PR-AUC, so their rows carry metric="r2" and the generic score_* columns.
regression_tasks = {
    "numax_hon": ("numax_hon", True),
    "numax_hatt": ("numax_hatt", True),
    "dnu_hatt": ("dnu_hatt", True),
    "prot_kounkel": ("prot_kounkel", False),
}


def regression_population(tics: list[int], labels: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Rows of this split that carry a usable target, plus the (optionally log10) target values.
    Mirrors new_task_scorecard.score_regression_task exactly, including the prot_kounkel cut to periods
    that fit inside one segment, so the population matches the numbers already reported for these probes.
    """
    column, log_target = regression_tasks[task]
    lookup = labels.set_index("tic_id")
    target = lookup[column].reindex(tics)
    keep = target.notna()
    if column == "prot_kounkel":
        gt57 = lookup["prot_kounkel_gt57"].reindex(tics).fillna(1).astype(int)
        keep = keep & (gt57 == 0) # headline population: periods inside one segment length
    keep = keep.to_numpy()
    values = target[keep].to_numpy(dtype=float)
    if log_target:
        values = np.log10(values)
    return keep, values


def star_labels(tics: list[int], subset: pd.DataFrame, task: str) -> np.ndarray:
    """Star-level binary labels in the cache's ascending-tic order; every cached tic must be in the subset."""
    lookup = subset.set_index("tic_id")[task]
    values = lookup.reindex(tics)
    assert not values.isna().any(), f"{int(values.isna().sum())} cached tics missing from the subset labels"
    return values.to_numpy().astype(np.int64)


def load_label_source(pool: str, tasks: tuple[str, ...]) -> pd.DataFrame:
    """
    One tic-keyed frame carrying the binary label column for every requested task.
    v1 reads the frozen packed subset's baked labels; the new-task pool joins its own catalogue frame
    (external asteroseismic flags plus the resurrected flare column) onto the pool membership, filling
    absent tics with 0 exactly as new_task_scorecard.y_for does for detection flags.
    """
    if pool == "v1":
        return pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    from swm.eval.new_task_scorecard import label_frame
    labels = label_frame().reset_index()
    pool_frame = pd.read_parquet(repo_root / "processed" / "subset" / "new_task_pool.parquet")
    merged = pool_frame[["tic_id"]].merge(labels, on="tic_id", how="left")
    for task in tasks:
        column = regression_tasks[task][0] if task in regression_tasks else task
        assert column in merged.columns, f"new-task pool has no label column {column!r}"
        if task in regression_tasks:
            # continuous target: keep NaN, it is what defines this probe's population
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        else:
            merged[task] = pd.to_numeric(merged[task], errors="coerce").fillna(0).astype(int)
    return merged


def broadcast_labels(y_star: np.ndarray, blocks: list[np.ndarray]) -> np.ndarray:
    """Repeat each star's label across its windows: the weak supervision every score-space operator starts from."""
    parts = []
    for i, block in enumerate(blocks):
        parts.append(np.full(block.shape[0], y_star[i], dtype=np.int64))
    return np.concatenate(parts)


def window_counts(blocks: list[np.ndarray]) -> np.ndarray:
    counts = np.zeros(len(blocks), dtype=np.int64)
    for i, block in enumerate(blocks):
        counts[i] = block.shape[0]
    return counts


def fit_window_scores(readout: str, train_blocks: list[np.ndarray], y_train_rows: np.ndarray,
                      eval_blocks: dict[str, list[np.ndarray]], n_folds: int = 2,
                      ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Fit the per-window readout on broadcast labels and score every window of every split.
    Train-split window scores are produced by n_folds cross-fitting (each fold scored by a model that
    never saw it), because ws_ppv_lspv fits a second-stage model on them and in-sample window scores
    would be optimistically separated. Val and test are scored by the full-train model.
    """
    x_train = np.concatenate(train_blocks, axis=0)
    eval_scores = {}
    for split, blocks in eval_blocks.items():
        eval_scores[split] = fit_readout_scores(readout, x_train, y_train_rows,
                                                np.concatenate(blocks, axis=0))
    oof = np.zeros(len(x_train), dtype=np.float64)
    star_counts = window_counts(train_blocks)
    star_fold = np.arange(len(train_blocks)) % n_folds # split by STAR so a star never leaks across folds
    row_fold = np.repeat(star_fold, star_counts)
    for fold in range(n_folds):
        held = row_fold == fold
        oof[held] = fit_readout_scores(readout, x_train[~held], y_train_rows[~held], x_train[held])
    return oof, eval_scores


def score_feature_cells(pooled: dict[str, dict[str, np.ndarray]], y: dict[str, np.ndarray],
                        readout: str) -> list[dict]:
    """Fit the star-level readout on each pooled train feature matrix and record val/test PR-AUC."""
    rows = []
    for key, mats in pooled.items():
        pooling, param = key
        scores_val = fit_readout_scores(readout, mats["train"], y["train"], mats["val"])
        scores_test = fit_readout_scores(readout, mats["train"], y["train"], mats["test"])
        rows.append({"family": "feature", "pooling": pooling, "param": param,
                     "pr_auc_val": float(average_precision_score(y["val"], scores_val)),
                     "pr_auc_test": float(average_precision_score(y["test"], scores_test))})
    return rows


def score_regression_cells(pooled: dict[str, dict[str, np.ndarray]], keep: dict[str, np.ndarray],
                           y: dict[str, np.ndarray]) -> list[dict]:
    """
    Ridge on each pooled feature matrix, restricted to the stars carrying a target, scored by R2.
    Only FEATURE-space poolings run here: score-space MIL operators aggregate per-window scores under
    the standard-MI assumption ("the bag is positive if any window is"), which has no counterpart for
    a continuous whole-star property like nu_max. Every target in this menu is global, so the
    collective assumption holds and the interesting question is only whether dispersion helps.
    """
    rows = []
    for key, mats in pooled.items():
        pooling, param = key
        x_train = mats["train"][keep["train"]]
        fitted = {}
        for split in ["val", "test"]:
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_train) # learn mean/std on train only (no leakage)
            x_ev = scaler.transform(mats[split][keep[split]])
            reg = RidgeCV(alphas=np.logspace(-2, 3, 10)) # L2 linear probe, alpha by leave-one-out CV
            reg.fit(x_tr, y["train"])
            fitted[split] = reg.predict(x_ev)
        rows.append({"family": "feature", "pooling": pooling, "param": param,
                     "score_val": float(r2_score(y["val"], fitted["val"])),
                     "score_test": float(r2_score(y["test"], fitted["test"])),
                     "rmse_test": float(np.sqrt(mean_squared_error(y["test"], fitted["test"]))),
                     "spearman_test": float(spearmanr(y["test"], fitted["test"]).statistic)})
    return rows


def run_arm(cache: dict, subset: pd.DataFrame, tasks: tuple[str, ...], readout: str,
            poolings: tuple[str, ...], kmatch: int | None, kmatch_seed: int = 0) -> pd.DataFrame:
    """Score every (pooling, hyperparameter, task) cell for one encoder arm at one bag scope."""
    blocks = {}
    offsets = {}
    tics = {}
    for split in ["train", "val", "test"]:
        tics[split], blocks[split], offsets[split] = cache[split]

    if kmatch is not None:
        for split in ["train", "val", "test"]:
            blocks[split], offsets[split] = subsample_bags(blocks[split], offsets[split], kmatch,
                                                           seed=kmatch_seed)

    wanted_feature = []
    wanted_score = []
    for pooling in poolings:
        if pooling in feature_poolings:
            wanted_feature.append(pooling)
        elif pooling in score_poolings:
            wanted_score.append(pooling)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")

    # Unsupervised feature maps are label-free, so fit them once per arm and reuse across all tasks.
    pooled: dict = {}
    for pooling in tqdm(wanted_feature, desc="feature maps", total=len(wanted_feature)):
        for param in pooling_grids[pooling]:
            op = FeaturePooling(pooling, param).fit(blocks["train"])
            mats = {}
            for split in ["train", "val", "test"]:
                mats[split] = op.transform(blocks[split])
            pooled[(pooling, param)] = mats
    control = {}
    for split in ["train", "val", "test"]:
        control[split] = bag_size_features(blocks[split])
    pooled[("bagsize_only", None)] = control

    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = window_counts(blocks[split])

    rows = []
    for task in tqdm(tasks, desc="tasks", total=len(tasks)):
        if task in regression_tasks:
            keep = {}
            y = {}
            for split in ["train", "val", "test"]:
                keep[split], y[split] = regression_population(tics[split], subset, task)
            if min(len(y["train"]), len(y["val"]), len(y["test"])) < 10:
                log.warning(f"{task}: a split has under 10 targets; skipped")
                continue
            task_rows = score_regression_cells(pooled, keep, y)
            for row in task_rows:
                row["task"] = task
                row["metric"] = "r2"
                row["n_test"] = int(len(y["test"]))
                row["n_train"] = int(len(y["train"]))
            rows.extend(task_rows)
            continue

        y = {}
        for split in ["train", "val", "test"]:
            y[split] = star_labels(tics[split], subset, task)
        if y["train"].sum() == 0 or y["val"].sum() == 0 or y["test"].sum() == 0:
            log.warning(f"{task}: a split lacks positives; skipped")
            continue

        task_rows = score_feature_cells(pooled, y, readout)

        if len(wanted_score) > 0:
            y_rows = broadcast_labels(y["train"], blocks["train"])
            oof_train, eval_scores = fit_window_scores(
                readout, blocks["train"], y_rows, {"val": blocks["val"], "test": blocks["test"]})
            for pooling in wanted_score:
                for param in pooling_grids[pooling]:
                    if pooling == "ws_ppv_lspv":
                        feats = {"train": aggregate_scores(oof_train, counts["train"], offsets["train"], pooling, param),
                                 "val": aggregate_scores(eval_scores["val"], counts["val"], offsets["val"], pooling, param),
                                 "test": aggregate_scores(eval_scores["test"], counts["test"], offsets["test"], pooling, param)}
                        scores_val = fit_readout_scores(readout, feats["train"], y["train"], feats["val"])
                        scores_test = fit_readout_scores(readout, feats["train"], y["train"], feats["test"])
                    else:
                        scores_val = aggregate_scores(eval_scores["val"], counts["val"], offsets["val"], pooling, param)
                        scores_test = aggregate_scores(eval_scores["test"], counts["test"], offsets["test"], pooling, param)
                    task_rows.append({"family": "score", "pooling": pooling, "param": param,
                                      "pr_auc_val": float(average_precision_score(y["val"], scores_val)),
                                      "pr_auc_test": float(average_precision_score(y["test"], scores_test))})

        for row in task_rows:
            row["task"] = task
            row["metric"] = "pr_auc"
            row["score_val"] = row["pr_auc_val"] # generic columns so both metrics share one schema
            row["score_test"] = row["pr_auc_test"]
            row["base_rate_val"] = float(y["val"].mean())
            row["base_rate_test"] = float(y["test"].mean())
            row["n_val_pos"] = int(y["val"].sum())
            row["n_test_pos"] = int(y["test"].sum())
            row["n_test"] = int(len(y["test"]))
        rows.extend(task_rows)

    frame = pd.DataFrame(rows)
    frame["readout"] = readout
    frame["kmatch"] = kmatch if kmatch is not None else 0
    frame["mean_bag_size"] = float(counts["test"].mean())
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="MIL pooling sweep over cached bags")
    parser.add_argument("--cells", nargs="+", default=["exp05_comb_fbwd_c1p0", "exp05_comb_off"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--scope", default="first", choices=["first", "all"])
    parser.add_argument("--pool", default="v1", choices=["v1", "new_task"],
                        help="v1 = the frozen packed subset; new_task = the separate new-task eval pool")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="default: the four v1 tasks, or the new-task detection probes under --pool new_task")
    parser.add_argument("--readout", default="logistic", choices=["logistic", "gbm", "mlp"])
    parser.add_argument("--poolings", nargs="+", default=list(feature_poolings) + list(score_poolings))
    parser.add_argument("--kmatch", type=int, default=None,
                        help="subsample every bag to this many windows before pooling (bag-size control)")
    parser.add_argument("--kmatch-draws", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="default: mil_sweep.csv for the v1 pool, mil_sweep_new_task.csv for the new-task pool")
    args = parser.parse_args()
    if args.out is None:
        stem = "mil_sweep" if args.pool == "v1" else "mil_sweep_new_task"
        args.out = f"experiments/mil_pooling/{stem}.csv" # pools never share a CSV: different star populations

    if args.tasks is not None:
        tasks = tuple(args.tasks)
    elif args.pool == "new_task":
        tasks = new_task_tasks_default
    else:
        tasks = tasks_default
    subset = load_label_source(args.pool, tasks)
    arms = []
    for cell in args.cells:
        for seed in args.seeds:
            arms.append((cell, seed, "trained"))
    arms.append(("untrained", 0, "untrained"))

    out_path = repo_root / args.out
    frames = []
    for cell, seed, kind in tqdm(arms, desc="arms", total=len(arms)):
        path = cache_path(cell, seed, args.scope, pool=args.pool)
        if not path.exists():
            log.warning(f"{path.name}: no cache; skipped (build it with swm.eval.mil_cache)")
            continue
        cache = load_cache(path)
        draws = args.kmatch_draws if args.kmatch is not None else 1
        for draw in range(draws):
            frame = run_arm(cache, subset, tasks, args.readout, tuple(args.poolings),
                            args.kmatch, kmatch_seed=draw)
            frame["exp_name"] = cell
            frame["seed"] = seed
            frame["arm_kind"] = kind
            frame["bag_scope"] = args.scope
            frame["kmatch_draw"] = draw
            frame["pool"] = args.pool
            frames.append(frame)
        log.info(f"{cell} seed{seed} [{args.scope}]: {len(frames[-1])} cells x {draws} draw(s)")

    assert len(frames) > 0, (
        f"no arms scored for pool={args.pool} scope={args.scope}: every cache was missing. "
        f"Build them with `python -m swm.eval.mil_cache --pool {args.pool} --scope {args.scope}`.")
    result = pd.concat(frames, ignore_index=True)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    result["git_sha"] = _git_sha()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        previous = pd.read_csv(out_path)
        result = pd.concat([previous, result], ignore_index=True) # append-only audit trail
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(result)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
