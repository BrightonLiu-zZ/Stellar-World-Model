"""Per-star held-out predictions for the paper's four linear-readout arms, on the 10 paper tasks (+ijspeert).

Why this exists (Yue Ma review 2026-09-10, item 2): the F1 scorecard keeps one score per (arm, seed,
task) and the 2*SE bands therefore measure seed spread only. A paired bootstrap over TEST STARS needs
the raw per-star predictions, which `analyze_f1_fusion_scorecard.py` computes and discards. This script
re-runs exactly the F1 `linear` / `mean` path -- same caches, same scorers, same keep masks -- and keeps
the predictions. Nothing is retrained: every fit is a logistic / ridge readout on cached mu.

Footing check: the metric recomputed from the dumped predictions must reproduce `f1_probe.csv` row by
row (tolerance 1e-9). If it does not, the dump is not the paper's arm and the script fails.

Run (swm env, repo root, PYTHONPATH=src, ~10 min CPU):
    python experiments/dump_paper_linear_scores.py
    python experiments/dump_paper_linear_scores.py --arms hann0p3_fbwd_s0 untrained --blocks v1   # smoke
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, arm_parts, load_mu_cache  # noqa: E402
from analyze_f1_fusion_scorecard import V1_TASKS, arm_tables  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import (DETECTION, REGRESSION, label_frame, score_contrastive,  # noqa: E402
                                         score_detection, score_ijspeert_from_mu,
                                         score_regression_task, score_rotation_period_from_mu)
from swm.eval.skyline import logistic_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("paper_linear_scores")

# The arms Table 1 / Figure 1 print: six pre-training seeds of the shipped recipe, their dynamics-off
# twins, and the single untrained control. `features` is arm-independent and scored once.
DEFAULT_ARMS = ([f"hann0p3_fbwd_s{s}" for s in range(6)]
                + [f"hann0p3_off_s{s}" for s in range(6)] + ["untrained"])
ARM_SETS = ("features_only", "mu", "features_plus_mu")
PAPER_TASKS = ("pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "rgb_vs_heb",
               "numax_hon", "rotation_period", "flare", "ijspeert")
SHAPE = {"rgb_vs_heb": "contrastive", "numax_hon": "regression", "rotation_period": "regression"}
COLUMN_TO_TASK = {column: name for name, column in DETECTION}
COLUMN_TO_TASK.update({column: name for name, column, _ in REGRESSION})


def metric_value(shape: str, y: np.ndarray, score: np.ndarray) -> float:
    """The task's headline metric from raw predictions -- the same three the F1 tables report."""
    if shape == "regression":
        return float(r2_score(y, score))
    if shape == "contrastive":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def v1_star_scores(table: dict, labels: pd.DataFrame) -> list[dict]:
    """The exact `v1_rows` path of the F1 script, returning per-star (tic, y, score) instead of one number."""
    frames = []
    for split in ["train", "test"]:
        tics, blocks = table[split]
        values = np.concatenate(blocks, axis=0)
        frame = pd.DataFrame(values, columns=[f"f{j}" for j in range(values.shape[1])])
        frame.insert(0, "tic_id", tics)
        frame.insert(1, "split", split)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True).merge(labels, on="tic_id", how="inner")
    assert len(merged) == sum(len(table[s][0]) for s in ["train", "test"]), "a cached star is missing"
    cols = [c for c in merged.columns if c.startswith("f")]
    out = []
    for task in V1_TASKS:
        tics, y, scores = logistic_scores(merged, cols, task)
        out.append({"task": task, "shape": "detection", "tics": tics, "y": y, "scores": scores})
    return out


def menu_star_scores(table: dict, subset_table: dict, labels: pd.DataFrame) -> list[dict]:
    """The `menu_rows` path with the scorers' own sinks switched on; only the paper's tasks are kept."""
    sink: list[dict] = []
    for _, column in DETECTION:
        score_detection(table, labels, column, poolings=("mean",), sink=sink)
    score_contrastive(table, labels, sink=sink)
    for _, column, log_target in REGRESSION:
        if COLUMN_TO_TASK[column] != "numax_hon":
            continue  # the other regressions are ADR-0010 non-reportable; skip the fits
        score_regression_task(table, labels, column, log_target, sink=sink)
    score_rotation_period_from_mu(subset_table, sink=sink)
    score_ijspeert_from_mu(subset_table, sink=sink)
    out = []
    for entry in sink:
        task = COLUMN_TO_TASK.get(entry["task"], entry["task"])
        if task not in PAPER_TASKS:
            continue
        out.append({"task": task, "shape": SHAPE.get(task, "detection"),
                    "tics": entry["tics"], "y": entry["y"], "scores": entry["scores"]})
    return out


def footing_check(dumps: pd.DataFrame, probe_path: Path, tol: float) -> None:
    """Every (arm, arm_set, task) metric recomputed from the dump must equal the published F1 row."""
    probe = pd.read_csv(probe_path)
    probe = probe[(probe["readout"] == "mean") & (probe["readout_family"] == "linear")]
    metric_col = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
    worst = 0.0
    n_checked = 0
    for (arm, arm_set, task, shape), group in dumps.groupby(["arm", "arm_set", "task", "shape"]):
        ref = probe[(probe["arm"] == arm) & (probe["arm_set"] == arm_set) & (probe["task"] == task)]
        if ref.empty:
            log.warning(f"no F1 row for {arm}/{arm_set}/{task}; cannot check it")
            continue
        published = float(ref[metric_col[shape]].iloc[0])
        recomputed = metric_value(shape, group["y"].to_numpy(), group["score"].to_numpy())
        gap = abs(published - recomputed)
        worst = max(worst, gap)
        n_checked += 1
        if gap > tol:
            raise AssertionError(f"{arm}/{arm_set}/{task}: dump gives {recomputed:.6f}, "
                                 f"f1_probe.csv says {published:.6f} (gap {gap:.2e} > {tol})")
    log.info(f"footing check: {n_checked} cells reproduce f1_probe.csv, worst gap {worst:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump per-star predictions for the paper's linear arms.")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--blocks", nargs="+", default=["v1", "menu"], choices=["v1", "menu"])
    ap.add_argument("--out-dir", default="experiments/paper_bootstrap")
    ap.add_argument("--probe-csv", default="experiments/f1_fusion_scorecard/f1_probe.csv")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    home = repo_root / "experiments" / "exp08_menu_channel"
    cache_dir, subset_cache_dir = home / "mu_cache", home / "subset_mu_cache"
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_menu = label_frame()
    v1_labels = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
                 [["tic_id", *V1_TASKS]].drop_duplicates("tic_id"))
    subset_feats = cached_subset_features(repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    pool_feats = None
    if "menu" in args.blocks:
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        pool_feats = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                          repo_root / "processed" / "sequences", None)

    frames = []
    features_done: set[str] = set()
    for arm in tqdm(args.arms, desc="arms", total=len(args.arms)):
        family, seed = arm_parts(arm)
        subset_mu = load_mu_cache(subset_cache_dir / f"{arm}.npz")
        v1_sets = arm_tables(subset_mu, align_features(subset_feats, subset_mu), "mean", ["train", "test"])
        menu_sets = None
        if "menu" in args.blocks:
            pool_mu = load_mu_cache(cache_dir / f"{arm}.npz")
            menu_sets = arm_tables(pool_mu, align_features(pool_feats, pool_mu), "mean", ["train", "test"])
        for arm_set in ARM_SETS:
            if arm_set == "features_only":
                if "features" in features_done:
                    continue
                features_done.add("features")
                arm_label, family_label, seed_label = "features", "features", -1
            else:
                arm_label, family_label, seed_label = arm, family, seed
            entries = []
            if "v1" in args.blocks:
                entries += v1_star_scores(v1_sets[arm_set], v1_labels)
            if menu_sets is not None:
                entries += menu_star_scores(menu_sets[arm_set], v1_sets[arm_set], labels_menu)
            for entry in entries:
                frames.append(pd.DataFrame({"arm": arm_label, "family": family_label, "seed": seed_label,
                                            "arm_set": arm_set, "task": entry["task"],
                                            "shape": entry["shape"],
                                            "tic_id": np.asarray(entry["tics"]).astype(np.int64),
                                            "y": np.asarray(entry["y"]).astype(float),
                                            "score": np.asarray(entry["scores"]).astype(float)}))
        log.info(f"scored {arm}")

    dumps = pd.concat(frames, ignore_index=True)
    footing_check(dumps, repo_root / args.probe_csv, args.tol)
    out_path = out_dir / "linear_star_scores.parquet"
    dumps.to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(dumps)} rows, "
             f"{dumps.groupby(['arm', 'arm_set', 'task']).ngroups} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
