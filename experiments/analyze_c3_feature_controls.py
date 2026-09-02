"""C3 -- GBM and MLP on the 25 engineered features. Scorecard arm 7 (roadmap C3 / Yue Ma's Y13a).

THE QUESTION THIS ANSWERS, and the pre-registered risk it carries. F1's headline is
`features (+) mu` beating `features`, both under a LINEAR readout. That comparison is only
interesting if the linear readout is not itself the bottleneck: if a nonlinear model on the SAME 25
features recovers whatever mu was adding, the honest claim narrows from "SSL adds information the
engineered features do not carry" to "SSL adds to a LINEAR readout on engineered features" -- a much
weaker sentence, and one the paper has to print either way. Recorded in the 08-15 analysis file before
this was run; do not soften it after the fact.

WHAT ALREADY EXISTED, and what this adds. `exp08_prechecks/ceiling_A1A2.csv` holds A1 (linear) and A2
(GBM) on the engineered features, but only for the MENU block and only at a single random_state. So:
    + the four v1 tasks under GBM and MLP        (absent entirely)
    + the menu block under MLP                   (absent)
    + a SEED AXIS on both nonlinear families     (absent -- A2 is one fit)
and A2 becomes this script's reproduction control, the same way A1 was R1's.

WHY A SEED AXIS AT ALL, given the features are deterministic. The mu arms carry a 6-seed spread from
encoder training; a single nonlinear fit has no error bar, and comparing a point estimate to a mean
over 6 seeds is how a difference gets called significant that is not. So each nonlinear readout is fit
under 6 random_states. That is NOT the same kind of variation as the mu arms' (it is estimator noise,
not representation noise) and the output labels it as such rather than letting a reader assume parity.
Where a family's seed spread turns out to be ~0 (HistGradientBoosting is close to deterministic below
its subsampling thresholds), that is reported as a measured fact, not hidden.

READOUT LOCK. This never touches the v1 headline probe. The linear-probe lock (CLAUDE.md) stands:
GBM/MLP appear here as CONTROLS on the feature baseline, never as a readout on mu.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~25 min):
    PYTHONUNBUFFERED=1 python experiments/analyze_c3_feature_controls.py
    python experiments/analyze_c3_feature_controls.py --seeds 0 --blocks v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, load_mu_cache, stacked  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import (DETECTION, REGRESSION, label_frame, score_contrastive,  # noqa: E402
                                         score_detection, score_ijspeert_from_mu,
                                         score_regression_task, score_rotation_period_from_mu)
from swm.eval.readout_sweep import fit_readout_scores  # noqa: E402
from swm.eval.skyline import logistic_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("c3_feature_controls")

V1_TASKS = ("pulsating", "eb", "rotation", "transit")
REPORTABLE_MENU = {"numax_hon", "rotation_period", "osc_giant", "solar_like_osc", "rgb_vs_heb",
                   "ijspeert", "flare"}
HEADLINE = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
# `linear` is the A1 baseline every delta is measured against; it is deterministic, so it is fit once.
FAMILIES = {"linear": ("logistic", "ridge"), "gbm": ("gbm", "gbm"), "mlp": ("mlp", "mlp")}
# Any arm's cache works: only the STAR LIST is read from it, to align the feature table to the exact
# population the mu arms were scored on. No mu values enter this script at all.
ALIGNMENT_ARM = "hann0p3_fbwd_s0"


def menu_rows(table: dict, subset_table: dict, labels: pd.DataFrame, clf: str, reg: str,
              seed: int) -> list[dict]:
    """Every menu probe for one (family, seed), through new_task_scorecard's OWN scorers.

    Going through the shipped scorers rather than re-deriving keeps each probe's keep-mask -- numax
    floor, prot 5.7 d cap, rgb State filter, Villanova exclusion -- byte-identical to the frozen
    scorecard. Re-deriving them here is how two tables end up on quietly different populations.
    """
    rows = []
    for name, column in DETECTION:
        for cell in score_detection(table, labels, column, readout=clf, poolings=("mean",),
                                    random_state=seed):
            rows.append({"task": name, "shape": "detection", **cell})
    for cell in score_contrastive(table, labels, readout=clf, random_state=seed):
        rows.append({"task": "rgb_vs_heb", "shape": "contrastive", **cell})
    for name, column, log_target in REGRESSION:
        for cell in score_regression_task(table, labels, column, log_target, regressor=reg,
                                          random_state=seed):
            rows.append({"task": name, "shape": "regression", **cell})
    for cell in score_rotation_period_from_mu(subset_table, regressor=reg, random_state=seed):
        rows.append({"task": "rotation_period", "shape": "regression", **cell})
    for cell in score_ijspeert_from_mu(subset_table, readout=clf, random_state=seed):
        rows.append(cell)
    return rows


def v1_rows(table: dict, labels: pd.DataFrame, family: str, clf: str, seed: int) -> list[dict]:
    """The four v1 tasks on the engineered basis under one readout family."""
    frames = []
    for split in ["train", "test"]:
        tics, blocks = table[split]
        values = np.concatenate(blocks, axis=0)
        frame = pd.DataFrame(values, columns=[f"f{j}" for j in range(values.shape[1])])
        frame.insert(0, "tic_id", tics)
        frame.insert(1, "split", split)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True).merge(labels, on="tic_id", how="inner")
    cols = [c for c in merged.columns if c.startswith("f")]
    train, test = merged[merged["split"] == "train"], merged[merged["split"] == "test"]
    rows = []
    for task in V1_TASKS:
        if family == "linear":
            # the exact published probe path, so the C3 baseline row IS the F1 features_only row
            _, y, scores = logistic_scores(merged, cols, task)
        else:
            y = test[task].to_numpy()
            scores = fit_readout_scores(clf, train[cols].to_numpy(), train[task].to_numpy(),
                                        test[cols].to_numpy(), seed)
        rows.append({"task": task, "shape": "detection", "pr_auc": float(average_precision_score(y, scores)),
                     "n_test": int(len(y)), "n_test_pos": int(np.asarray(y).sum())})
    return rows


def reproduction_check(probe: pd.DataFrame) -> pd.DataFrame:
    """GBM at seed 0 must reproduce the A2 engineered-feature ceiling, which used exactly that fit."""
    path = repo_root / "experiments" / "exp08_prechecks" / "ceiling_A1A2.csv"
    if not path.exists():
        log.warning("ceiling_A1A2.csv absent; A2 reproduction control skipped")
        return pd.DataFrame()
    ref = pd.read_csv(path)
    rows = []
    for arm, family in (("A1_feats", "linear"), ("A2_feats", "gbm")):
        sub = ref[ref["arm"] == arm]
        for _, cell in probe[(probe["family"] == family) & (probe["seed"] == 0)
                             & (probe["block"] == "menu")].iterrows():
            match = sub[sub["task"] == cell["task"]]
            if match.empty:
                continue
            metric = HEADLINE[cell["shape"]]
            rows.append({"check": f"{family}_vs_{arm}", "task": cell["task"], "metric": metric,
                         "new": float(cell[metric]), "reference": float(match[metric].iloc[-1])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_diff"] = (out["new"] - out["reference"]).abs()
    log.info("A1/A2 reproduction control, five largest deviations\n"
             + out.sort_values("abs_diff", ascending=False).head(5).round(6).to_string(index=False))
    assert out["abs_diff"].max() < 1e-6, f"A1/A2 reproduction FAILED, max |diff| {out['abs_diff'].max()}"
    return out


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Per-family mean and 2*SE over seeds, plus each nonlinear family's delta over the linear baseline."""
    rows = []
    for (block, task, shape), group in probe.groupby(["block", "task", "shape"], sort=False):
        metric = HEADLINE[shape]
        base = group[group["family"] == "linear"][metric]
        if base.empty:
            continue
        linear = float(base.iloc[0])
        n_test = int(group["n_test"].max())
        has_pos = bool(group["n_test_pos"].notna().any())
        n_pos = int(group["n_test_pos"].max()) if has_pos else -1
        for family, fam in group.groupby("family", sort=False):
            values = fam[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append({
                "block": block, "task": task, "shape": shape, "metric": metric, "family": family,
                "n_test": n_test, "n_test_pos": n_pos,
                "prevalence": (n_pos / n_test) if has_pos else np.nan,
                "n_seeds": len(values), "score_mean": float(values.mean()), "score_sd": sd,
                "score_2se": 2 * sd / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                "linear_baseline": linear, "delta_vs_linear": float(values.mean()) - linear,
                "reportable": bool(block == "v1" or task in REPORTABLE_MENU),
                "spread_kind": "estimator noise (random_state), NOT representation noise",
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="C3: GBM + MLP on the 25 engineered features (arm 7).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--blocks", nargs="+", default=["v1", "menu"], choices=["v1", "menu"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--subset-cache-dir", default=None)
    ap.add_argument("--out-dir", default="experiments/c3_feature_controls")
    args = ap.parse_args()

    home = repo_root / "experiments" / "exp08_menu_channel"
    cache_dir = Path(args.cache_dir) if args.cache_dir else home / "mu_cache"
    subset_cache_dir = Path(args.subset_cache_dir) if args.subset_cache_dir else home / "subset_mu_cache"
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_menu = label_frame()
    v1_labels = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
                 [["tic_id", *V1_TASKS]].drop_duplicates("tic_id"))
    subset_feats = align_features(cached_subset_features(
        repo_root / "experiments" / "exp01_window256_seq16" / "packed"),
        load_mu_cache(subset_cache_dir / f"{ALIGNMENT_ARM}.npz"))
    pool_feats = None
    if "menu" in args.blocks:
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        pool_feats = align_features(
            cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                 repo_root / "processed" / "sequences", None),
            load_mu_cache(cache_dir / f"{ALIGNMENT_ARM}.npz"))
    log.info(f"feature basis: {stacked(subset_feats, 'test').shape[1]} columns")

    # `linear` is deterministic: fitting it 6 times would report a fake zero-width error bar as if it
    # had been measured. It is fit once and carries seed -1 to make that visible in the CSV.
    jobs = [(f, s) for f in args.families for s in (args.seeds if f != "linear" else [-1])]
    rows: list[dict] = []
    for family, seed in tqdm(jobs, desc="family x seed", total=len(jobs)):
        clf, reg = FAMILIES[family]
        if "v1" in args.blocks:
            for row in v1_rows(subset_feats, v1_labels, family, clf, max(seed, 0)):
                rows.append({"block": "v1", "family": family, "seed": seed, "readout": clf, **row})
        if "menu" in args.blocks:
            for row in menu_rows(pool_feats, subset_feats, labels_menu, clf, reg, max(seed, 0)):
                rows.append({"block": "menu", "family": family, "seed": seed, "readout": clf, **row})

    probe = pd.DataFrame(rows)
    probe.to_csv(out_dir / "c3_probe.csv", index=False)
    log.info(f"wrote {out_dir / 'c3_probe.csv'} ({len(probe)} rows)")

    if "menu" in args.blocks and 0 in args.seeds:
        checks = reproduction_check(probe)
        if not checks.empty:
            checks.to_csv(out_dir / "c3_repro.csv", index=False)

    summary = summarize(probe)
    summary.to_csv(out_dir / "c3_summary.csv", index=False)
    log.info(f"wrote {out_dir / 'c3_summary.csv'} ({len(summary)} rows)")

    view = summary[summary["reportable"]]
    print("\nC3 -- nonlinear readouts on the 25 engineered features (score_mean by family):")
    print(view.pivot_table(index=["block", "task"], columns="family", values="score_mean")
          .round(4).to_string())
    print("\nseed spread by family (score_sd; ~0 means the family is near-deterministic here):")
    print(view.pivot_table(index="family", values="score_sd", aggfunc=["mean", "max"]).round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
