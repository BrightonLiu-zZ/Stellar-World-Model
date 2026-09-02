"""F-B (exp10 forensics) -- how much of mu the 25 engineered features can already predict, and whether
the map they need is nonlinear.

THE QUESTION. C3b found that most of what mu adds to a LINEAR readout on the engineered features
disappears once the features get a nonlinear readout. Two very different mechanisms produce that:
either mu is a nonlinear recoding of the same information (a GBM on the features can rebuild it, so mu
buys nothing a strong readout does not), or mu carries genuinely new content that the probes cannot
reach. Regressing mu ON the features separates them, because it measures the recoding directly rather
than inferring it from a probe score.

WHAT IS MEASURED, per encoder seed and per star population:
    ridge   mu_d ~ 25 features, RidgeCV with a per-target alpha, standardized on train only
    gbm     mu_d ~ 25 features, HistGradientBoostingRegressor under 6 random_states, averaged
both scored as test R^2 for each of the 128 latent dims, then aggregated three ways:
    mean            the unweighted mean over dims -- reported, but a dim carrying no variance counts
                    as much as a dim carrying all of it, so it is not the headline
    var_weighted    1 - sum_d SSE_d / sum_d SST_d, i.e. the fraction of mu's TOTAL test variance the
                    features explain. This is the aggregate the decision rules are scored on
    probe_weighted  the same per-dim R^2 averaged with weights |coef_d| taken from each task's own
                    linear probe on mu -- "are the dims the probe actually USES the predictable ones?"

DECISION RULES (pre-registered in the 2026-08-29 forensics handoff, scored verbatim):
    R-B1  gbm var_weighted - ridge var_weighted >= 0.15  --> mu is substantially a NONLINEAR recoding,
          so any exp10 complementarity objective must penalise nonlinear predictability; a linear
          decorrelation penalty is pre-declared insufficient.
    R-B2  1 - gbm var_weighted < 0.20 AND probe-used dims more predictable than average --> mu's used
          content is mostly redundant, so exp10 must CREATE content, not protect it.
    R-B3  large unpredictable fraction but low mu_perp_full probe scores --> the residual exists but is
          not linearly usable, favouring levers that reorganise content over decorrelation.

POPULATION AND MASK FIDELITY. The probe weights need each task's own keep-mask, and the shipped scorers
return scores rather than coefficients, so the masks are rebuilt here -- and then checked against the
n_test the F1 artifact recorded for the same task. A silent population difference is exactly the failure
this project has hit before, so it is asserted rather than trusted.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~10 min with --jobs 12):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp10_fb_mu_predictability.py
    python experiments/analyze_exp10_fb_mu_predictability.py --seeds 0 --populations subset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, load_mu_cache, stacked  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import label_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_fb")

source_home = repo_root / "experiments" / "exp08_menu_channel"
out_home = repo_root / "experiments" / "exp10_forensics" / "fb_predictability"
v1_tasks = ("pulsating", "eb", "rotation", "transit")
gbm_seeds = [0, 1, 2, 3, 4, 5]
ridge_alphas = np.logspace(-2, 3, 10)  # the RidgeCV grid the shipped regression probe uses


def pooled_mu(population: str, arm: str, cache_home: Path | None = None) -> dict:
    """Read one arm's per-window cache and mean-pool it to one row per star, the readout every probe uses."""
    sub = "subset_mu_cache" if population == "subset" else "mu_cache"
    cache = load_mu_cache((cache_home or source_home) / sub / f"{arm}.npz")
    table = {}
    for split in ["train", "test"]:
        tics, blocks = cache[split]
        rows = []
        for block in blocks:
            rows.append(block.mean(axis=0).reshape(1, -1))
        table[split] = (list(tics), rows)
    return table


def per_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score one predicted mu table dimension by dimension, keeping the pieces the aggregates need.
    Returns (R^2 per dim, residual sum of squares per dim, total sum of squares per dim); the
    variance-weighted aggregate is 1 - sum(sse) / sum(sst), which is why sse and sst come back too.
    """
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.full(len(sst), np.nan)
    alive = sst > 0  # a dim with no test variance has no R^2; it is dropped, not scored as 0 or 1
    r2[alive] = 1.0 - sse[alive] / sst[alive]
    return r2, sse, sst


def ridge_predictions(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Linear map features --> every mu dim at once, with the ridge penalty chosen per target by LOO-CV."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)  # learn mean/std on train only (no leakage)
    x_te = scaler.transform(x_test)
    model = RidgeCV(alphas=ridge_alphas, alpha_per_target=True)  # one alpha per mu dim, not one shared
    model.fit(x_tr, y_train)
    return model.predict(x_te)


def gbm_dim_predictions(x_train: np.ndarray, y_train_dim: np.ndarray, x_test: np.ndarray,
                        seed: int) -> np.ndarray:
    """One mu dim's nonlinear fit; trees are scale-invariant so nothing is standardized here."""
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(x_train, y_train_dim)
    return model.predict(x_test)


def probe_masks(population: str, tics: dict[str, list[int]]) -> dict[str, dict]:
    """
    Rebuild each task's keep-mask and target on one population, so a probe's COEFFICIENTS can be read.
    The shipped scorers return predictions only, and the weights F-B needs are per-dim coefficients; the
    masks are therefore mirrored here and checked against the F1 artifact's own n_test downstream.
    Returns {task: {"shape", "mask": {split: bool array}, "y": {split: array}}}.
    """
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    out: dict[str, dict] = {}

    if population == "subset":
        v1 = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
              [["tic_id", *v1_tasks]].drop_duplicates("tic_id").set_index("tic_id"))
        for task in v1_tasks:
            mask, y = {}, {}
            for split in ["train", "test"]:
                values = v1[task].reindex(tics[split])
                assert values.notna().all(), f"{task}: subset stars missing from the v1 label table"
                mask[split] = np.ones(len(tics[split]), dtype=bool)
                y[split] = values.to_numpy(dtype=int)
            out[task] = {"shape": "detection", "mask": mask, "y": y}

        cat = pd.read_csv(repo_root / "labels" / "external" / "ijspeert2024_bright.csv")
        positives = set(cat["TIC"].astype(int))
        mask, y = {}, {}
        for split in ["train", "test"]:
            flags = []
            for tic in tics[split]:
                flags.append(int(int(tic) in positives))
            mask[split] = np.ones(len(tics[split]), dtype=bool)
            y[split] = np.array(flags, dtype=int)
        out["ijspeert"] = {"shape": "detection", "mask": mask, "y": y}

        canon["rotation"] = pd.to_numeric(canon["rotation"], errors="coerce").fillna(0).astype(int)
        canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
        keep = canon.loc[(canon["rotation"] == 1) & canon["rotation_period"].notna()
                         & (canon["rotation_period"] <= 5), ["tic_id", "rotation_period"]]
        lookup = keep.set_index("tic_id")["rotation_period"]
        mask, y = {}, {}
        for split in ["train", "test"]:
            target = lookup.reindex(tics[split])
            mask[split] = target.notna().to_numpy()
            y[split] = target[mask[split]].to_numpy(dtype=float)
        out["rotation_period"] = {"shape": "regression", "mask": mask, "y": y}
        return out

    labels = label_frame()
    for task in ["osc_giant", "solar_like_osc", "flare"]:
        mask, y = {}, {}
        for split in ["train", "test"]:
            mask[split] = np.ones(len(tics[split]), dtype=bool)
            y[split] = labels[task].reindex(tics[split]).to_numpy(dtype=int)
        out[task] = {"shape": "detection", "mask": mask, "y": y}

    mask, y = {}, {}
    for split in ["train", "test"]:
        target = labels["rgb_vs_heb"].reindex(tics[split])
        mask[split] = target.notna().to_numpy()
        y[split] = target[mask[split]].astype(int).to_numpy()
    out["rgb_vs_heb"] = {"shape": "detection", "mask": mask, "y": y}

    mask, y = {}, {}
    for split in ["train", "test"]:
        target = labels["numax_hon"].reindex(tics[split])
        mask[split] = target.notna().to_numpy()
        y[split] = np.log10(target[mask[split]].to_numpy(dtype=float))  # the asteroseismic scaling
    out["numax_hon"] = {"shape": "regression", "mask": mask, "y": y}
    return out


def probe_weights(mu_train: np.ndarray, spec: dict) -> np.ndarray:
    """
    The per-dim |coefficient| of one task's own linear probe on standardized mu.
    Standardizing first is what makes the coefficients comparable across dims: without it the weight
    would mostly measure each dim's scale. Regression probes use the same RidgeCV the scorecard uses.
    """
    x = mu_train[spec["mask"]["train"]]
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    if spec["shape"] == "regression":
        model = RidgeCV(alphas=ridge_alphas)
        model.fit(x_std, spec["y"]["train"])
        coef = np.asarray(model.coef_, dtype=float).ravel()
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=2000)
        model.fit(x_std, spec["y"]["train"])
        coef = np.asarray(model.coef_, dtype=float).ravel()
    weights = np.abs(coef)
    return weights / weights.sum()


def score_one(population: str, seed: int, feats: dict, jobs: int, arm_prefix: str = "hann0p3_fbwd",
              cache_home: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Everything F-B measures for one (population, encoder seed): per-dim R^2 under both families, the
    three aggregates, and the probe-weighted aggregate for every task that population carries.
    The arm prefix and cache home are parameters so exp10's cells can be measured on the same
    instrument as the reference; the defaults reproduce the published F-B numbers exactly.
    """
    arm = f"{arm_prefix}_s{seed}"
    mu = pooled_mu(population, arm, cache_home)
    aligned = align_features(feats, mu)
    tics = {}
    for split in ["train", "test"]:
        tics[split] = list(mu[split][0])
    x_train, x_test = stacked(aligned, "train"), stacked(aligned, "test")
    y_train, y_test = stacked(mu, "train"), stacked(mu, "test")
    n_dim = y_train.shape[1]

    r2_ridge, sse_ridge, sst = per_dim_r2(y_test, ridge_predictions(x_train, y_train, x_test))

    gbm_r2, gbm_sse = [], []
    for gbm_seed in gbm_seeds:
        preds = Parallel(n_jobs=jobs)(
            delayed(gbm_dim_predictions)(x_train, y_train[:, d], x_test, gbm_seed) for d in range(n_dim))
        r2_d, sse_d, _ = per_dim_r2(y_test, np.stack(preds, axis=1))
        gbm_r2.append(r2_d)
        gbm_sse.append(sse_d)
    r2_gbm = np.mean(np.stack(gbm_r2, axis=0), axis=0)
    sse_gbm = np.mean(np.stack(gbm_sse, axis=0), axis=0)

    dims = pd.DataFrame({"population": population, "seed": seed, "dim": np.arange(n_dim),
                         "mu_test_var": sst / len(y_test), "r2_ridge": r2_ridge, "r2_gbm": r2_gbm})

    rows = [{"population": population, "seed": seed, "weighting": "unweighted", "task": "",
             "n_used": n_dim, "ridge": float(np.nanmean(r2_ridge)), "gbm": float(np.nanmean(r2_gbm))},
            {"population": population, "seed": seed, "weighting": "mu_variance", "task": "",
             "n_used": n_dim, "ridge": float(1.0 - sse_ridge.sum() / sst.sum()),
             "gbm": float(1.0 - sse_gbm.sum() / sst.sum())}]

    specs = probe_masks(population, tics)
    mu_train = y_train
    for task, spec in specs.items():
        weights = probe_weights(mu_train, spec)
        alive = np.isfinite(r2_ridge) & np.isfinite(r2_gbm)
        share = weights[alive] / weights[alive].sum()  # renormalise over the dims that carry variance
        rows.append({"population": population, "seed": seed, "weighting": "probe_coef", "task": task,
                     "n_used": int(spec["mask"]["test"].sum()),
                     "ridge": float((share * r2_ridge[alive]).sum()),
                     "gbm": float((share * r2_gbm[alive]).sum())})
    return dims, pd.DataFrame(rows)


def check_populations(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Control: every rebuilt keep-mask must give the n_test the published F1 probe table recorded.
    A mask that quietly drifted would move the probe weights and nothing else would notice.
    """
    path = repo_root / "experiments" / "f1_fusion_scorecard" / "f1_probe.csv"
    if not path.exists():
        log.warning(f"{path} absent; population control skipped")
        return pd.DataFrame()
    ref = pd.read_csv(path)
    ref = ref[(ref["readout"] == "mean") & (ref["readout_family"] == "linear")]
    checks = []
    for task, group in rows[rows["weighting"] == "probe_coef"].groupby("task"):
        match = ref[ref["task"] == task]
        if match.empty:
            continue
        checks.append({"task": task, "n_test_here": int(group["n_used"].max()),
                       "n_test_f1": int(match["n_test"].max())})
    out = pd.DataFrame(checks)
    if out.empty:
        return out
    out["match"] = out["n_test_here"] == out["n_test_f1"]
    log.info("population control (rebuilt keep-mask vs F1's own n_test)\n" + out.to_string(index=False))
    assert out["match"].all(), f"keep-mask drift on {out.loc[~out['match'], 'task'].tolist()}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="F-B: predictability of mu from the 25 engineered features.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--populations", nargs="+", default=["subset", "pool"], choices=["subset", "pool"])
    ap.add_argument("--jobs", type=int, default=12, help="joblib workers for the per-dim GBM fan")
    ap.add_argument("--arm-prefix", default="hann0p3_fbwd",
                    help="Arm family to measure; `<prefix>_s<seed>`. Default reproduces published F-B.")
    ap.add_argument("--cache-home", default=None,
                    help="Directory holding {subset_mu_cache,mu_cache}. Default: exp08_menu_channel.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cache_home = Path(args.cache_home) if args.cache_home else None

    out_dir = Path(args.out_dir) if args.out_dir else out_home
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {}
    if "subset" in args.populations:
        feats["subset"] = cached_subset_features(
            repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    if "pool" in args.populations:
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        feats["pool"] = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                             repo_root / "processed" / "sequences", None)

    jobs = [(p, s) for p in args.populations for s in args.seeds]
    dim_frames, agg_frames = [], []
    for population, seed in tqdm(jobs, desc="population x seed", total=len(jobs)):
        dims, aggs = score_one(population, seed, feats[population], args.jobs, args.arm_prefix, cache_home)
        dim_frames.append(dims)
        agg_frames.append(aggs)

    per_dim = pd.concat(dim_frames, ignore_index=True)
    aggregates = pd.concat(agg_frames, ignore_index=True)
    per_dim.to_csv(out_dir / "fb_per_dim_r2.csv", index=False)
    aggregates.to_csv(out_dir / "fb_aggregates.csv", index=False)

    checks = check_populations(aggregates)
    if not checks.empty:
        checks.to_csv(out_dir / "fb_population_control.csv", index=False)

    headline = aggregates[aggregates["weighting"] == "mu_variance"]
    print("\nF-B mu-variance-weighted R^2 of mu on the 25 features, per population (mean over 6 seeds):")
    print(headline.groupby("population")[["ridge", "gbm"]].agg(["mean", "std"]).round(4).to_string())
    print("\nR-B1 input, gbm - ridge (threshold 0.15):")
    print((headline.groupby("population")["gbm"].mean()
           - headline.groupby("population")["ridge"].mean()).round(4).to_string())
    print("\nprobe-coefficient-weighted R^2 per task (mean over seeds), beside the unweighted aggregate:")
    view = aggregates[aggregates["weighting"].isin(["probe_coef", "unweighted"])]
    print(view.pivot_table(index=["population", "weighting", "task"], values=["ridge", "gbm"])
          .round(4).to_string())
    log.info(f"wrote {out_dir}/fb_{{per_dim_r2,aggregates,population_control}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
