"""Rotation rescore on pool 3 (design: docs/plans/2026-09-20-rotation-pool-design.md, signed 2026-09-20).

Pool 1 has no rotation-only star, so the ML4PS `rotation` and `rotation_period` rows can be won by
detecting the co-occurring class. This script scores the same arms, readouts and estimators on pool 3
(`swm.eval.rotation_pool`): the 106,284 corpus stars outside pool 1, natural prevalence, none seen in
pre-training. NO ENCODER PASS: R8 already pooled mu (mean and std over first-segment windows) for every
one of these stars under all 18 arms, and T2 recovered the 25 features; this script only reads them.

Vocabulary is `analyze_f1_fusion_scorecard.py`'s, verbatim -- arm sets, readouts, families, the
summary and absolute tables -- so the new rows are commensurable with the other eight. Populations
ride in the `block` column because that is the key `summarize` groups on:
    pool3           the headline (everything kept)
    pool3_noflare   sensitivity: flare_ever = 0 in both classes, probe REFIT on the restricted train
    pool3_strict    sensitivity: rotation-only positives vs negatives in no catalogue at all, refit
    pool1           the ML4PS rows, rescored through THIS code path (footing check F1, and the
                    old-vs-new appendix table)

FOOTING CHECKS run first and abort before any pool-3 score exists:
    F1  this code path on pool 1 reproduces the printed rotation and rotation-period rows
    F2  R8's `untrained_i0` is the paper's single `untrained` arm (compared on stars in pool 2's cache)
    F3  the 25-feature table equals pool 2's cached feature table on the shared stars

Untrained: all six R8 inits are scored; `untrained_i0` is the paper's arm, the rest are the init
spread. The trained-minus-untrained contrast the sign-off's A2 rule reads pairs seed s with init s, so
it carries both sides' spread (F17); fixed here, before any score was read.

Run (repo root, swm env, PYTHONPATH=src; CPU only):
    PYTHONUNBUFFERED=1 python experiments/analyze_rotation_pool.py 2>&1 | tee experiments/rotation_pool/run.log
    python experiments/analyze_rotation_pool.py --stages footing
    python experiments/analyze_rotation_pool.py --families xgb --arms exp07_hann0p3_fbwd_s0   # pilot
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

from analyze_exp08_menu_channel import align_features, as_blocks, load_mu_cache, stacked  # noqa: E402
from analyze_f1_fusion_scorecard import (AMP_COLS, FAMILIES, NONLINEAR_ARM_SETS, absolute_rows,  # noqa: E402
                                         concat, paired_delta, pool, residualize, summarize)
from analyze_t2_beyond_baseline_rotation import arm_parts, load_arm_mu, load_features25  # noqa: E402
from swm.eval.features import FEATURE_NAMES  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import score_regression  # noqa: E402
from swm.eval.readout_sweep import fit_readout_scores  # noqa: E402
from swm.eval.rotation_pool import PERIOD_CAP_DAYS, load_canonical  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("rotation_pool")

out_dir = repo_root / "experiments" / "rotation_pool"
menu_home = repo_root / "experiments" / "exp08_menu_channel"
SPLITS = ["train", "test"]
READOUTS = ("mean", "mean_std", "mean_perp_amp")
SENSITIVITY = {"pool3_noflare": "no_flare", "pool3_strict": "strict"}
R8_ARMS = ([f"exp07_hann0p3_fbwd_s{s}" for s in range(6)] + [f"exp07_hann0p3_off_s{s}" for s in range(6)]
           + [f"untrained_i{i}" for i in range(6)])
# Table 1 as printed in the ML4PS submission (paper/tables/table1_scorecard.tex), linear family, `mean`:
# (features, mu 6-seed mean, fusion 6-seed mean). Three decimals are printed, hence the tolerance.
PRINTED_POOL1 = {"rotation": (0.540, 0.559, 0.555), "rotation_period": (0.703, 0.677, 0.717)}
PRINT_TOL = 5e-4 + 1e-9


def paper_arm(r8_arm: str) -> tuple[str, str, int]:
    """(arm, family, seed) in the paper's naming: R8 prefixes trained cells with `exp07_`, F1 does not."""
    family, seed = arm_parts(r8_arm)
    family = family.removeprefix("exp07_")
    return (f"untrained_i{seed}" if family == "untrained" else f"{family}_s{seed}"), family, seed


# --------------------------------------------------------------------------------------- design tables
def table_from_frame(frame: pd.DataFrame, members: dict[str, np.ndarray]) -> dict:
    """A tic-indexed matrix cut into the {split: (tics, 1-row blocks)} layout the F1 helpers take."""
    table = {}
    for split in SPLITS:
        tics = members[split]
        table[split] = as_blocks(tics.tolist(), frame.loc[tics].to_numpy(dtype=np.float32))
    return table


def pool3_arm_sets(mu: pd.DataFrame, feats: dict, readout: str, members: dict[str, np.ndarray]) -> dict[str, dict]:
    """F1's `arm_tables`, for mu that R8 already pooled: `mean_std` is a column concat, not a re-pool."""
    mean_cols = [c for c in mu.columns if c.startswith("mean")]
    cols = list(mu.columns) if readout == "mean_std" else mean_cols
    pooled = table_from_frame(mu[cols], members)
    if readout == "mean_perp_amp":
        return {"mu": residualize(pooled, feats, AMP_COLS, SPLITS)}
    return {"features_only": feats, "mu": pooled, "features_plus_mu": concat(feats, pooled, SPLITS),
            "mu_perp_full": residualize(pooled, feats, list(FEATURE_NAMES), SPLITS)}


def pool1_arm_sets(mu: dict, feats: dict, readout: str) -> dict[str, dict]:
    """The same four arm sets from pool 1's per-window caches, pooled by F1's own `pool`."""
    pooled = pool(mu, readout, SPLITS)
    if readout == "mean_perp_amp":
        return {"mu": residualize(pooled, feats, AMP_COLS, SPLITS)}
    return {"features_only": feats, "mu": pooled, "features_plus_mu": concat(feats, pooled, SPLITS),
            "mu_perp_full": residualize(pooled, feats, list(FEATURE_NAMES), SPLITS)}


# --------------------------------------------------------------------------------------------- scoring
def score_table(table: dict, labels: pd.DataFrame, family: str, seed: int,
                keep: set[int] | None = None, sink: list | None = None) -> list[dict]:
    """Both rotation rows from one design table; `keep` restricts BOTH splits before the probe is fit."""
    clf, reg = FAMILIES[family]
    x, y_rot, y_per, tics_of = {}, {}, {}, {}
    for split in SPLITS:
        tics = np.asarray(table[split][0])
        values = stacked(table, split)
        if keep is not None:
            mask = np.array([t in keep for t in tics])
            tics, values = tics[mask], values[mask]
        x[split], tics_of[split] = values, tics
        y_rot[split] = labels["rotation"].reindex(tics).to_numpy(dtype=int)
        y_per[split] = labels["rotation_period"].reindex(tics).to_numpy(dtype=float)

    scores = fit_readout_scores(clf, x["train"], y_rot["train"], x["test"], seed)
    rows = [{"task": "rotation", "shape": "detection",
             "pr_auc": float(average_precision_score(y_rot["test"], scores)),
             "n_train": int(len(y_rot["train"])), "n_test": int(len(y_rot["test"])),
             "n_test_pos": int(y_rot["test"].sum())}]

    in_scope = {s: (y_rot[s] == 1) & (y_per[s] <= PERIOD_CAP_DAYS) for s in SPLITS}
    metrics, pred = score_regression(x["train"][in_scope["train"]], y_per["train"][in_scope["train"]],
                                     x["test"][in_scope["test"]], y_per["test"][in_scope["test"]], reg, seed)
    rows.append({"task": "rotation_period", "shape": "regression", **metrics,
                 "n_train": int(in_scope["train"].sum()), "n_test": int(in_scope["test"].sum())})
    if sink is not None:
        sink.append(pd.DataFrame({"task": "rotation", "tic_id": tics_of["test"], "y": y_rot["test"], "score": scores}))
        sink.append(pd.DataFrame({"task": "rotation_period", "tic_id": tics_of["test"][in_scope["test"]],
                                  "y": y_per["test"][in_scope["test"]], "score": pred}))
    return rows


def score_arm(arm_sets_by_readout: dict[str, dict], labels: pd.DataFrame, block: str, names: tuple[str, str, int],
              family: str, features_done: set, keep: set[int] | None, dumps: list | None) -> list[dict]:
    """Every (readout, arm set) cell of one arm in one population, F1's probe-row schema."""
    arm, arm_family, seed = names
    rows = []
    for readout, sets in arm_sets_by_readout.items():
        for arm_set, table in sets.items():
            if family != "linear" and arm_set not in NONLINEAR_ARM_SETS:
                continue
            if arm_set == "features_only":
                key = (block, readout, family) if family == "linear" else (block, readout, family, seed)
                if key in features_done:
                    continue
                features_done.add(key)
                label = ("features", "features", -1 if family == "linear" else seed)
            else:
                label = (arm, arm_family, seed)
            # per-star scores are dumped for the headline readout only: that is all the bootstrap reads
            sink = [] if (dumps is not None and readout == "mean") else None
            for row in score_table(table, labels, family, seed, keep, sink):
                rows.append({"block": block, "arm": label[0], "family": label[1], "seed": label[2],
                             "readout": readout, "readout_family": family, "arm_set": arm_set, **row})
            for frame in sink or []:
                dumps.append(frame.assign(block=block, arm=label[0], family=label[1], seed=label[2],
                                          readout_family=family, arm_set=arm_set))
    return rows


# ---------------------------------------------------------------------------------------------- footing
def footing_f1(labels: pd.DataFrame) -> pd.DataFrame:
    """This code path on pool 1 must reproduce the printed Table 1 rotation rows before pool 3 is read."""
    feats = cached_subset_features(repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    rows, features_done = [], set()
    for seed in tqdm(range(6), desc="F1: pool-1 reproduction", total=6):
        mu = load_mu_cache(menu_home / "subset_mu_cache" / f"hann0p3_fbwd_s{seed}.npz")
        sets = pool1_arm_sets(mu, align_features(feats, mu), "mean")
        rows += score_arm({"mean": sets}, labels, "pool1", (f"hann0p3_fbwd_s{seed}", "hann0p3_fbwd", seed),
                          "linear", features_done, None, None)
    probe = pd.DataFrame(rows)
    for task, printed in PRINTED_POOL1.items():
        metric = "pr_auc" if task == "rotation" else "r2"
        got = [float(probe[(probe["task"] == task) & (probe["arm_set"] == s)][metric].mean())
               for s in ("features_only", "mu", "features_plus_mu")]
        worst = max(abs(round(g, 3) - p) for g, p in zip(got, printed))
        log.info(f"F1 {task}: got {[round(g, 4) for g in got]} vs printed {list(printed)}")
        assert worst < PRINT_TOL, f"F1 FAILED on {task}: {got} vs {printed}"
    log.info("F1 PASS: pool-1 rotation rows reproduce through this code path")
    return probe


def footing_f2() -> None:
    """R8's `untrained_i0` must be the paper's `untrained` arm; checked where the two caches share stars.

    DEVIATION D1 from the design, which asked for max |diff| = 0. The two caches were extracted by
    different scripts with different batching, and float32 convolutions are not bit-stable across that:
    measured 2026-09-20, the SAME init differs by 5.2e-5 max / 3.5e-6 median (a trained arm shows the
    same, 2.7e-4 max), while a DIFFERENT init differs by 2.0e-1 max / 2.4e-2 median. So the check is
    discriminative rather than exact: i0 must match, and must match >= 100x better than i1 does.
    """
    paper = load_mu_cache(menu_home / "mu_cache" / "untrained.npz")
    tics, blocks = paper["test"]
    worst = {}
    for arm in ["untrained_i0", "untrained_i1"]:
        r8 = load_arm_mu(arm)
        mean_cols = [c for c in r8.columns if c.startswith("mean")]
        shared = [i for i, tic in enumerate(tics) if tic in r8.index]
        assert len(shared) > 1000, f"F2 cannot certify anything on {len(shared)} shared stars"
        theirs = np.stack([blocks[i].mean(axis=0) for i in shared])  # (n_shared, z)
        ours = r8.loc[[tics[i] for i in shared], mean_cols].to_numpy()  # (n_shared, z)
        worst[arm] = float(np.abs(theirs - ours).max())
    assert worst["untrained_i0"] < 1e-3 and worst["untrained_i0"] * 100 < worst["untrained_i1"], \
        f"F2 FAILED: untrained_i0 is not the paper's untrained arm: {worst}"
    log.info(f"F2 PASS: untrained_i0 matches the paper's `untrained` on {len(shared)} shared stars "
             f"(max |diff| {worst['untrained_i0']:.2e}; a different init gives {worst['untrained_i1']:.2e})")


def footing_f3(feats25: pd.DataFrame) -> None:
    """The 25-feature table must equal pool 2's cached one on the stars the two share."""
    pool2 = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                 repo_root / "processed" / "sequences", None)
    worst, n_shared = 0.0, 0
    for split in SPLITS:
        tics, blocks = pool2[split]
        shared = [i for i, t in enumerate(tics) if t in feats25.index]
        ours = feats25.loc[[tics[i] for i in shared], list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        theirs = np.concatenate([blocks[i] for i in shared], axis=0)
        assert np.array_equal(np.isnan(ours), np.isnan(theirs)), "F3 FAILED: NaN pattern differs"
        worst = max(worst, float(np.nanmax(np.abs(ours - theirs))))
        n_shared += len(shared)
    assert n_shared > 1000 and worst < 1e-5, f"F3 FAILED: {n_shared} shared stars, max |diff| {worst}"
    log.info(f"F3 PASS: features25 == pool-2 feature cache on {n_shared} shared stars (max |diff| {worst:.2e})")


# ---------------------------------------------------------------------------------------------- summary
def trained_minus_untrained(probe: pd.DataFrame) -> pd.DataFrame:
    """Sign-off A2: the fusion score of trained seed s minus untrained init s, paired, per cell."""
    fusion = probe[probe["arm_set"] == "features_plus_mu"]
    rows = []
    for keys, group in fusion.groupby(["block", "task", "shape", "readout", "readout_family"], sort=False):
        metric = "pr_auc" if keys[2] == "detection" else "r2"
        floor = group[group["family"] == "untrained"].set_index("seed")[metric]
        for family, fam in group.groupby("family", sort=False):
            if family == "untrained" or floor.empty:
                continue
            scores = fam.set_index("seed")[metric]
            shared = sorted(set(scores.index) & set(floor.index))
            if not shared:
                continue
            deltas = {int(s): float(scores[s] - floor[s]) for s in shared}
            rows.append({"block": keys[0], "task": keys[1], "readout": keys[3], "readout_family": keys[4],
                         "metric": metric, "contrast": "fusion_trained_minus_untrained", "family": family,
                         "spread_note": "seed s paired with init s: both sides' spread (F17)",
                         **paired_delta(deltas)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rotation rescore on pool 3, from the R8 caches.")
    ap.add_argument("--arms", nargs="+", default=R8_ARMS)
    ap.add_argument("--families", nargs="+", default=["linear"], choices=list(FAMILIES))
    ap.add_argument("--stages", nargs="+", default=["footing", "score"], choices=["footing", "score"])
    ap.add_argument("--tag", default=None, help="suffix for the output CSVs (pilots must not overwrite the run)")
    args = ap.parse_args()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    labels = load_canonical(repo_root / "labels" / "variability_labels_star.csv").set_index("tic_id")
    feats25 = load_features25()

    pool1_probe = pd.DataFrame()
    if "footing" in args.stages:
        pool1_probe = footing_f1(labels)
        footing_f2()
        footing_f3(feats25)
    if "score" not in args.stages:
        return 0

    pool3 = pd.read_parquet(repo_root / "processed" / "subset" / "rotation_pool.parquet")
    assert set(pool3["tic_id"]) == set(feats25.index), "pool 3 and the cached feature table cover different stars"
    members = {s: pool3.loc[pool3["split"] == s, "tic_id"].to_numpy() for s in SPLITS}
    feats = table_from_frame(feats25[list(FEATURE_NAMES)], members)
    keeps = {block: set(pool3.loc[pool3[flag], "tic_id"]) for block, flag in SENSITIVITY.items()}

    rows, dumps, features_done = [], [], set()
    jobs = [(arm, family) for arm in args.arms for family in args.families]
    for arm, family in tqdm(jobs, desc="arm x family", total=len(jobs)):
        mu = load_arm_mu(arm)
        # a nonlinear family is a control on the headline pooling and population only (F1's rule)
        readouts = READOUTS if family == "linear" else ("mean",)
        sets = {r: pool3_arm_sets(mu, feats, r, members) for r in readouts}
        rows += score_arm(sets, labels, "pool3", paper_arm(arm), family, features_done, None, dumps)
        if family == "linear":
            for block, keep in keeps.items():
                light = {"mean": {k: v for k, v in sets["mean"].items() if k in NONLINEAR_ARM_SETS}}
                rows += score_arm(light, labels, block, paper_arm(arm), family, features_done, keep, None)

    probe = pd.concat([pd.DataFrame(rows), pool1_probe], ignore_index=True)
    probe["mu_cache"] = str(repo_root / "experiments" / "r8_fullpool" / "mu_cache")
    probe.to_csv(out_dir / f"probe{suffix}.csv", index=False)
    summary = pd.concat([summarize(probe), trained_minus_untrained(probe)], ignore_index=True)
    summary.to_csv(out_dir / f"summary{suffix}.csv", index=False)
    absolute_rows(probe).to_csv(out_dir / f"absolute{suffix}.csv", index=False)
    if dumps:
        pd.concat(dumps, ignore_index=True).to_parquet(out_dir / f"star_scores{suffix}.parquet", index=False)
    log.info(f"wrote {out_dir}/{{probe,summary,absolute}}{suffix}.csv ({len(probe)} probe rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
