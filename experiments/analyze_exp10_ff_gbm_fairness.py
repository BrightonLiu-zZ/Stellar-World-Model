"""F-F (exp10 forensics) -- is C3b's "4 of 11 survive under GBM" a readout-TUNING artifact?

THE QUESTION. C3b compared `features` against `features (+) mu` under one HistGradientBoosting
configuration: sklearn's defaults. Defaults are a reasonable choice for a control, but they are not a
neutral one -- 153 columns and 25 columns are not equally well served by the same tree depth and the
same iteration budget, and the whole verdict that narrowed the paper's spine rests on that one config.
So: give BOTH arms the identical tuning budget, select on a validation split carved out of train, score
the test set once at the selected config, and see whether the survival count moves.

WHAT IS FAIR HERE, precisely. Same grid, same selection metric, same val split, same seeds, for both
arms. The fusion arm is not allowed a bigger grid because it has more columns, and the features arm is
not allowed to keep an advantage by being run at a config someone once found good for it. Test is
touched once per (task, arm, seed), at the config the val split chose.

DECISION RULES (pre-registered in the 2026-08-29 forensics handoff, scored verbatim):
    R-F1  the GBM survival count moves by >= 2 tasks in either direction --> C3b's 4 of 11 is
          tuning-fragile, exp10's gate is re-baselined on the tuned numbers and must name its readout
          config, and the C3b README gets an addendum (its published table is not edited).
    R-F2  the count stays 3-5 --> C3b is robust and the gate stays on the published baseline.
    R-F3  (either way) mu-column split-importance ~ 0 on tasks where the fusion delta ~ 0 --> the GBM
          IGNORES mu there (redundancy), as opposed to using it without gain. Which of the two it is
          shapes what a complementarity objective would have to do.

IMPORTANCE. HistGradientBoosting exposes no `feature_importances_`, so the gain-based split importance
is summed directly off the fitted trees' node arrays (`_predictors`), which is the same quantity a gain
importance would report. The fraction quoted is mu's share of total split gain at C3b's DEFAULT config,
because that is the config whose verdict R-F3 is interpreting.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~40 min with --jobs 8):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp10_ff_gbm_fairness.py
    python experiments/analyze_exp10_ff_gbm_fairness.py --tasks eb --seeds 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, load_mu_cache, stacked  # noqa: E402
from analyze_exp10_fb_mu_predictability import pooled_mu, probe_masks  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_ff")

out_home = repo_root / "experiments" / "exp10_forensics" / "ff_gbm_fairness"
# Headline metric per task, matching the published scorecard: PR-AUC for detection, ROC-AUC for the
# balanced contrastive probe, R^2 for the two regressions. Model SHAPE (classifier vs regressor) comes
# from probe_masks; only the metric differs for rgb_vs_heb.
task_metric = {"pulsating": "pr_auc", "eb": "pr_auc", "rotation": "pr_auc", "transit": "pr_auc",
               "ijspeert": "pr_auc", "rotation_period": "r2", "osc_giant": "pr_auc",
               "solar_like_osc": "pr_auc", "flare": "pr_auc", "rgb_vs_heb": "roc_auc",
               "numax_hon": "r2"}
subset_tasks = ("pulsating", "eb", "rotation", "transit", "ijspeert", "rotation_period")
val_fraction = 0.25  # carved from train; test is never used for selection


def grid(max_features_supported: bool) -> list[dict]:
    """
    The tuning grid, applied with an identical budget to both arms.
    Depth and iteration count are the two knobs that plausibly favour one column count over the other,
    and the per-split feature fraction is the knob that speaks directly to the dilution hypothesis.
    """
    configs = []
    for max_iter in [200, 800]:
        for max_depth in [3, 6]:
            for feature_fraction in [0.3, 1.0]:
                config = {"max_iter": max_iter, "max_depth": max_depth}
                if max_features_supported:
                    config["max_features"] = feature_fraction
                elif feature_fraction != 1.0:
                    continue  # no per-split subsampling knob on this sklearn; the axis is dropped loudly
                configs.append(config)
    return configs


def fit_predict(shape: str, x_train, y_train, x_test, seed: int, config: dict):
    """One GBM fit under one config, returning held-out predictions and the fitted estimator."""
    if shape == "regression":
        model = HistGradientBoostingRegressor(random_state=seed, **config)
    else:
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **config)
    model.fit(x_train, y_train)
    if shape == "regression":
        return model.predict(x_test), model
    return model.predict_proba(x_test)[:, 1], model


def score(metric: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    """The task's headline metric, so val selection and the test report use the same yardstick."""
    if metric == "r2":
        return float(r2_score(y_true, pred))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, pred))
    return float(average_precision_score(y_true, pred))


def gain_by_feature(model, n_feature: int) -> np.ndarray:
    """
    Total split gain per input column, summed over every tree in the fitted ensemble.
    HistGradientBoosting keeps no `feature_importances_`, so this reads the fitted predictors' node
    arrays directly; each non-leaf node records the column it split on and the gain that split bought.
    """
    totals = np.zeros(n_feature, dtype=float)
    for stage in model._predictors:
        for predictor in stage:
            nodes = predictor.nodes
            internal = nodes[~nodes["is_leaf"].astype(bool)]
            np.add.at(totals, internal["feature_idx"].astype(int), internal["gain"].astype(float))
    return totals


def arm_matrices(arm: str, x_feat: dict, x_mu: dict) -> dict[str, np.ndarray]:
    """The two column sets under comparison, built once per (task, seed) so both arms see the same rows."""
    if arm == "features_only":
        return x_feat
    out = {}
    for split in ["train", "test"]:
        out[split] = np.concatenate([x_feat[split], x_mu[split]], axis=1)
    return out


def run_cell(task: str, shape: str, metric: str, arm: str, seed: int, x_feat: dict, x_mu: dict,
             y: dict, configs: list[dict], n_feat: int) -> dict:
    """
    Tune one (task, arm, encoder seed) on a val split carved from train, then score test once.
    The val split is keyed on the encoder seed alone, so the two arms of a seed are selected on exactly
    the same rows and the delta stays paired -- the C3b estimator fix, carried forward.
    """
    x = arm_matrices(arm, x_feat, x_mu)
    stratify = None
    if shape != "regression":
        stratify = y["train"]
    idx_fit, idx_val = train_test_split(np.arange(len(y["train"])), test_size=val_fraction,
                                        random_state=seed, stratify=stratify)
    best, best_score = None, -np.inf
    for config in configs:
        pred, _ = fit_predict(shape, x["train"][idx_fit], y["train"][idx_fit], x["train"][idx_val],
                              seed, config)
        value = score(metric, y["train"][idx_val], pred)
        if value > best_score:
            best, best_score = config, value

    tuned_pred, _ = fit_predict(shape, x["train"], y["train"], x["test"], seed, best)
    default_pred, default_model = fit_predict(shape, x["train"], y["train"], x["test"], seed, {})
    row = {"task": task, "arm": arm, "seed": seed, "metric": metric,
           "tuned": score(metric, y["test"], tuned_pred),
           "default": score(metric, y["test"], default_pred),
           "val_best": best_score, "n_test": int(len(y["test"])),
           "n_train": int(len(y["train"])), "n_features": x["train"].shape[1]}
    for key, value in best.items():
        row[f"best_{key}"] = value
    if arm == "features_plus_mu":
        gains = gain_by_feature(default_model, x["train"].shape[1])
        total = gains.sum()
        row["mu_gain_fraction"] = float(gains[n_feat:].sum() / total) if total > 0 else np.nan
        row["mu_column_fraction"] = float((x["train"].shape[1] - n_feat) / x["train"].shape[1])
    return row


def summarize(probe: pd.DataFrame, published: pd.DataFrame | None) -> pd.DataFrame:
    """Per-task tuned and default fusion deltas, paired per seed, beside C3b's published GBM delta."""
    rows = []
    for task, group in probe.groupby("task", sort=False):
        wide = group.pivot_table(index="seed", columns="arm", values=["tuned", "default"])
        entry = {"task": task, "metric": group["metric"].iloc[0], "n_test": int(group["n_test"].max()),
                 "n_train": int(group["n_train"].max()),
                 "mu_gain_fraction": float(group["mu_gain_fraction"].mean(skipna=True))}
        for kind in ["tuned", "default"]:
            deltas = (wide[(kind, "features_plus_mu")] - wide[(kind, "features_only")]).to_numpy(float)
            sd = float(deltas.std(ddof=1))
            entry[f"{kind}_delta"] = float(deltas.mean())
            entry[f"{kind}_2se"] = 2 * sd / np.sqrt(len(deltas))
            entry[f"{kind}_survives"] = bool(deltas.mean() > 2 * sd / np.sqrt(len(deltas)))
            entry[f"{kind}_features"] = float(wide[(kind, "features_only")].mean())
            entry[f"{kind}_fusion"] = float(wide[(kind, "features_plus_mu")].mean())
        if published is not None:
            match = published[published["task"] == task]
            if not match.empty:
                entry["c3b_delta"] = float(match["delta_mean"].iloc[0])
                entry["c3b_survives"] = bool(match["beats_2se"].iloc[0])
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="F-F: fair GBM tuning on both arms of the C3b control.")
    ap.add_argument("--tasks", nargs="+", default=list(task_metric), choices=list(task_metric))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else out_home
    out_dir.mkdir(parents=True, exist_ok=True)

    supported = "max_features" in HistGradientBoostingClassifier().get_params()
    configs = grid(supported)
    if not supported:
        log.warning("this sklearn's HistGradientBoosting has no max_features; the per-split feature "
                    "fraction axis is DROPPED and the grid is 4 configs, not 8")
    log.info(f"grid: {len(configs)} configs x 2 arms x {len(args.seeds)} seeds x {len(args.tasks)} tasks")

    feats = {"subset": cached_subset_features(
        repo_root / "experiments" / "exp01_window256_seq16" / "packed")}
    log.info("loading pool feature table (~15 s, no output until it finishes)")
    feats["pool"] = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                         repo_root / "processed" / "sequences", None)

    cells = []
    for population in ["subset", "pool"]:
        tasks_here = []
        for task in args.tasks:
            in_subset = task in subset_tasks
            if (population == "subset") == in_subset:
                tasks_here.append(task)
        if not tasks_here:
            continue
        for seed in args.seeds:
            mu = pooled_mu(population, f"hann0p3_fbwd_s{seed}")
            aligned = align_features(feats[population], mu)
            tics = {"train": list(mu["train"][0]), "test": list(mu["test"][0])}
            specs = probe_masks(population, tics)
            feat_full = {"train": stacked(aligned, "train"), "test": stacked(aligned, "test")}
            mu_full = {"train": stacked(mu, "train"), "test": stacked(mu, "test")}
            for task in tasks_here:
                spec = specs[task]
                x_feat, x_mu, y = {}, {}, {}
                for split in ["train", "test"]:
                    keep = spec["mask"][split]
                    x_feat[split] = feat_full[split][keep]
                    x_mu[split] = mu_full[split][keep]
                    y[split] = spec["y"][split]
                for arm in ["features_only", "features_plus_mu"]:
                    cells.append((task, spec["shape"], task_metric[task], arm, seed, x_feat, x_mu, y,
                                  configs, feat_full["train"].shape[1]))

    rows = Parallel(n_jobs=args.jobs)(delayed(run_cell)(*cell) for cell in
                                      tqdm(cells, desc="task x arm x seed", total=len(cells)))
    probe = pd.DataFrame(rows)
    probe.to_csv(out_dir / "ff_probe.csv", index=False)

    published = None
    path = repo_root / "experiments" / "f1_nonlinear_control" / "f1_summary.csv"
    if path.exists():
        ref = pd.read_csv(path)
        published = ref[(ref["contrast"] == "fusion_minus_features") & (ref["readout"] == "mean")
                        & (ref["readout_family"] == "gbm") & (ref["family"] == "hann0p3_fbwd")]
    else:
        log.warning(f"{path} absent; C3b's published deltas are not joined onto the summary")

    summary = summarize(probe, published)
    summary.to_csv(out_dir / "ff_summary.csv", index=False)
    print("\nF-F fusion delta under GBM, published default vs identically-tuned both arms:")
    print(summary[["task", "metric", "default_delta", "default_survives", "tuned_delta",
                   "tuned_2se", "tuned_survives", "mu_gain_fraction"]].round(4).to_string(index=False))
    print(f"\nsurvival count: default {int(summary['default_survives'].sum())}/{len(summary)}, "
          f"tuned {int(summary['tuned_survives'].sum())}/{len(summary)}")
    log.info(f"wrote {out_dir}/ff_{{probe,summary}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
