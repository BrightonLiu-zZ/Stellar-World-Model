"""F1 -- the fusion readout on ALL 11 tasks, at the N1 readout ladder, in ONE table.

Roadmap 2026-08-25, row F1 (YM-W11 + the carried N1 readout switch). This is the paper's headline
table: the claim is that SSL mu from the unlabeled corpus adds information the 25 engineered features
do not already carry, and D18 says that claim is scored on 11 tasks with honest controls beside it.

WHAT DID NOT EXIST BEFORE THIS SCRIPT. Both halves of the measurement existed, in different shapes,
and neither could be printed next to the other:
    v1 (4 tasks)    exp07_channel_probe.csv has the five fusion readouts, but only at `mean` pooling
                    and only via encoder_mu_table's own encode pass.
    menu (7 tasks)  exp08_menu_channel/ (R1) has them at `mean`, through new_task_scorecard's scorers.
So: no task had `mean_std`, the two blocks used different code paths, and no single artifact carried
all 11. This script produces that artifact from the CACHES ALONE -- zero GPU.

READOUTS (roadmap D2 / N1, and the reason this is one pass rather than two):
    mean          the headline. Every table quotes it.
    mean_std      the second readout. Star-level mean-pooled mu concatenated with its std over the
                  star's windows -- 2x the columns, which is why it is reported and never selected on.
    mean_perp_amp the appendix robustness row, formerly `mean_resid`. It is NOT a pooling; it is `mean`
                  with the periodicity-free amplitude basis projected out, so it is named accordingly
                  and never quoted as a headline score.
    EMITTED FOR mu-ONLY ARMS ONLY. Projecting the 4-scalar amplitude basis out of mu and then
    concatenating all 25 engineered features -- which CONTAIN that basis -- back in would re-introduce
    exactly what was removed, so `mean_perp_amp` on a fusion arm is not a meaningful quantity. The
    fusion arm's confound control is `mu_perp_full` (all 25 projected out), which R1 used.

ARM SETS per task, all four mandatory (D18), plus the dyn-off arm the handoff's minimum list omits:
    features_only     the A1 engineered ceiling. Seedless and arm-independent by construction.
    mu                the SSL latent alone.
    features_plus_mu  THE HEADLINE ARM.
    mu_perp_full      mu with all 25 features projected out; what mu holds that they cannot say.
and each mu-bearing arm is scored for hann0p3_fbwd (6 seeds), hann0p3_off (6 seeds) and the untrained
reference. The off arm is not decoration: `delta_fbwd - delta_off` is R1's most robust result (5 of 7
menu probes) and is the only statistic here carrying BOTH arms' seed spreads (F17).

ENCODER-AGNOSTIC BY CONSTRUCTION. Cell name, cache dirs and arm list are all parameters, so if the
D17 encoder decision switches on Sept 1 this re-runs on the new mu with zero code change:
    python experiments/analyze_f1_fusion_scorecard.py --arm-prefix w0p025 \
        --cache-dir experiments/f1_w0p025/mu_cache --subset-cache-dir experiments/f1_w0p025/subset_mu_cache
MU-CACHE TRAP (exp09 method debt): caches are keyed `{arm}.npz` with NO checkpoint in the key and
short-circuit on exists(), so any NEW extraction must be given its OWN --cache-dir or it silently
scores the previous encoder under the new label. This script never writes caches, only reads them, and
records the resolved cache paths in the output so a mislabelled table is detectable after the fact.

ESTIMATOR CONVENTIONS THAT BIND (each violated once historically):
  - paired per seed within an arm; features_only carries no seed, so a fusion delta's spread is the mu
    side's ALONE and is labelled as such rather than dressed up as a two-sided SE.
  - never pool across the dynamics arm (F21). Every statistic is per-arm.
  - every PR-AUC delta names its prevalence (R8-F1): n_test and n_pos ride on every row.
  - absolute score is the selection quantity; gaps are reported, never selected on (F4).

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~20 min):
    PYTHONUNBUFFERED=1 python experiments/analyze_f1_fusion_scorecard.py
    python experiments/analyze_f1_fusion_scorecard.py --arms hann0p3_fbwd_s0 untrained --blocks v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse R1's menu scorers rather than fork them

from analyze_c3_feature_controls import menu_rows  # noqa: E402  (parameterized readout; see FAMILIES)
from analyze_exp08_menu_channel import (align_features, arm_parts, as_blocks, load_mu_cache,  # noqa: E402
                                        stacked)
from swm.eval.features import FEATURE_NAMES  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import label_frame  # noqa: E402
from swm.eval.readout_sweep import fit_readout_scores, pool_stars  # noqa: E402
from swm.eval.skyline import logistic_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("f1_fusion")

V1_TASKS = ("pulsating", "eb", "rotation", "transit")
AMP_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"]  # exp06 basis: scale + roughness, no periodicity
READOUTS = ("mean", "mean_std", "mean_perp_amp")
ARM_SETS = ("features_only", "mu", "features_plus_mu", "mu_perp_full")
# Readout FAMILY -> (classifier, regressor). `linear` is the headline and the only one D18 lists.
# `gbm`/`mlp` exist for the same-readout control the user approved 2026-08-25: C3 measures GBM/MLP on
# the FEATURES, and comparing that to a LINEAR fusion arm confounds readout capacity with information
# content, so the missing cell is (features (+) mu) under the SAME nonlinear readout. Reported as a
# control beside the linear headline, never replacing it -- the same status D18 grants C3's arm 7, and
# it does not touch the v1 headline probe on mu that CLAUDE.md's linear-probe lock governs.
FAMILIES = {"linear": ("logistic", "ridge"), "gbm": ("gbm", "gbm"), "mlp": ("mlp", "mlp")}
# Residualisation is a linear operation defined against a linear readout; running `mu_perp_full` under
# a tree ensemble would answer a question nobody asked, and `mean_perp_amp` likewise. The nonlinear
# families therefore score only the three arm sets the control actually needs.
NONLINEAR_ARM_SETS = ("features_only", "mu", "features_plus_mu")
REPORTABLE_MENU = {"numax_hon", "rotation_period", "osc_giant", "solar_like_osc", "rgb_vs_heb",
                   "ijspeert", "flare"}  # ADR-0010: one probe per physical quantity
HEADLINE = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
DEFAULT_ARMS = ([f"hann0p3_fbwd_s{s}" for s in range(6)]
                + [f"hann0p3_off_s{s}" for s in range(6)] + ["untrained"])
# eb at `mean` on the frozen recipe, 6 seeds -- exp07_aux_gap_6seed.csv. The footing check: if the
# caches do not reproduce this, nothing downstream is measuring the probe the gates were decided on.
REPRO_TARGET = {"cell": "hann0p3_fbwd", "task": "eb", "readout": "mean", "value": 0.7710, "tol": 5e-4,
                "n_seeds": 6}


# ------------------------------------------------------------------------------------------ pooling
def pool(mu: dict, readout: str, splits: list[str]) -> dict:
    """Reduce each star's window-mu block to the one row the probes score, under the named readout.

    `mean_std` concatenates the mean and the per-dimension std over the star's own windows: the std
    half is what a single mean-pooled vector throws away, which is why it is the second readout rather
    than a curiosity. `mean_perp_amp` is built later (it needs the feature table), so it pools as mean.
    """
    base = "mean" if readout == "mean_perp_amp" else readout
    out = {}
    for split in splits:
        tics, blocks = mu[split]
        rows = []
        for block in blocks:
            mean = pool_stars([block], "mean")[0]
            if base == "mean_std":
                rows.append(np.concatenate([mean, block.std(axis=0)]).reshape(1, -1))
            else:
                rows.append(mean.reshape(1, -1))
        out[split] = (tics, rows)
    return out


def residualize(mu: dict, feats: dict, basis: list[str], splits: list[str]) -> dict:
    """Project a feature sub-basis out of mu, fitting the linear map on the TRAIN split only."""
    keep = [FEATURE_NAMES.index(name) for name in basis]
    fitter = LinearRegression().fit(stacked(feats, "train")[:, keep], stacked(mu, "train"))
    out = {}
    for split in splits:
        out[split] = as_blocks(mu[split][0],
                               stacked(mu, split) - fitter.predict(stacked(feats, split)[:, keep]))
    return out


def concat(left: dict, right: dict, splits: list[str]) -> dict:
    """Column-concatenate two aligned one-row-per-star tables (the fusion cell: features then mu)."""
    return {s: as_blocks(left[s][0], np.concatenate([stacked(left, s), stacked(right, s)], axis=1))
            for s in splits}


def arm_tables(mu: dict, feats: dict, readout: str, splits: list[str]) -> dict[str, dict]:
    """The arm sets available at one readout for one arm.

    `mean_perp_amp` deliberately yields ONLY the mu arm: see the module docstring -- a fusion cell with
    the amplitude basis projected out of mu but all 25 features concatenated back in is incoherent.
    """
    pooled = pool(mu, readout, splits)
    if readout == "mean_perp_amp":
        return {"mu": residualize(pooled, feats, AMP_COLS, splits)}
    return {"features_only": feats,
            "mu": pooled,
            "features_plus_mu": concat(feats, pooled, splits),
            "mu_perp_full": residualize(pooled, feats, list(FEATURE_NAMES), splits)}


# --------------------------------------------------------------------------------------- v1 scoring
def v1_rows(table: dict, labels: pd.DataFrame, splits: list[str], family: str = "linear",
            seed: int = 0) -> list[dict]:
    """The four v1 tasks, one row per task. `linear` is the protocol-matched logistic probe verbatim."""
    frames = []
    for split in splits:
        tics, blocks = table[split]
        values = np.concatenate(blocks, axis=0)
        frame = pd.DataFrame(values, columns=[f"f{j}" for j in range(values.shape[1])])
        frame.insert(0, "tic_id", tics)
        frame.insert(1, "split", split)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True).merge(labels, on="tic_id", how="inner")
    assert len(merged) == sum(len(table[s][0]) for s in splits), "a cached star is missing from the labels"
    cols = [c for c in merged.columns if c.startswith("f")]
    train, test = merged[merged["split"] == "train"], merged[merged["split"] == "test"]
    clf = FAMILIES[family][0]
    rows = []
    for task in V1_TASKS:
        if family == "linear":
            _, y, scores = logistic_scores(merged, cols, task)
        else:
            y = test[task].to_numpy()
            scores = fit_readout_scores(clf, train[cols].to_numpy(), train[task].to_numpy(),
                                        test[cols].to_numpy(), seed)
        rows.append({"task": task, "shape": "detection", "pr_auc": float(average_precision_score(y, scores)),
                     "n_test": int(len(y)), "n_test_pos": int(np.asarray(y).sum())})
    return rows


# ------------------------------------------------------------------------------------------ summary
def paired_delta(values: dict[int, float]) -> dict:
    """Mean, SD and 2*SE of a per-seed delta series, per-seed values kept for the CSV."""
    deltas = np.array(list(values.values()), dtype=float)
    sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else np.nan
    two_se = 2 * sd / np.sqrt(len(deltas)) if len(deltas) > 1 else np.nan
    row = {"n_seeds": len(deltas), "delta_mean": float(deltas.mean()), "delta_sd": sd,
           "delta_2se": two_se, "beats_2se": bool(len(deltas) > 1 and deltas.mean() > two_se)}
    row.update({f"delta_s{seed}": value for seed, value in values.items()})
    return row


def absolute_rows(probe: pd.DataFrame) -> pd.DataFrame:
    """Per-arm ABSOLUTE score, mean and 2*SE over seeds, for every readout and arm set.

    Two reasons this exists beside the delta table. (1) Absolute score is the selection quantity and
    gaps are reported, never selected on (F4) -- a summary carrying only deltas invites the opposite.
    (2) `mean_perp_amp` has no features_only arm by construction, so it produces NO delta rows at all;
    without this block the appendix readout would be computed and then silently absent from the summary.
    """
    rows = []
    for keys, group in probe.groupby(["block", "task", "shape", "readout", "readout_family",
                                      "arm_set", "family"], sort=False):
        block, task, shape, readout, readout_family, arm_set, family = keys
        values = group[HEADLINE[shape]].to_numpy(dtype=float)
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        has_pos = bool(group["n_test_pos"].notna().any())
        rows.append({"block": block, "task": task, "shape": shape, "readout": readout,
                     "readout_family": readout_family, "arm_set": arm_set, "family": family,
                     "metric": HEADLINE[shape],
                     "n_seeds": len(values), "score_mean": float(values.mean()), "score_sd": sd,
                     "score_2se": 2 * sd / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                     "n_test": int(group["n_test"].max()),
                     "n_test_pos": int(group["n_test_pos"].max()) if has_pos else -1,
                     "reportable": bool(block == "v1" or task in REPORTABLE_MENU)})
    return pd.DataFrame(rows)


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Fusion delta and mu-vs-features per arm family, plus the fbwd-minus-off arm contrast.

    Prevalence rides on every row by construction (R8-F1): the same PR-AUC delta means different things
    at different base rates, and this project has quoted one without the other before.
    """
    rows = []
    fusion_by_family: dict[str, dict[tuple, dict[int, float]]] = {}
    for (block, task, shape, readout, readout_family), group in probe.groupby(
            ["block", "task", "shape", "readout", "readout_family"], sort=False):
        metric = HEADLINE[shape]
        base_rows = group[group["arm_set"] == "features_only"]
        if base_rows.empty:
            continue  # mean_perp_amp carries no features arm by design
        # Under `linear` the engineered arm is one deterministic fit (seed -1) and every fusion seed is
        # differenced against it. Under gbm/mlp it carries its own seeds, and the delta is PAIRED.
        base_by_seed = {int(r["seed"]): float(r[metric]) for _, r in base_rows.iterrows()}
        paired_base = len(base_by_seed) > 1
        features_only = float(np.mean(list(base_by_seed.values())))
        n_test = int(group["n_test"].max())
        # Regression probes have no positives, so prevalence is undefined rather than zero-or-all. R8-F1
        # says every PR-AUC delta must name its prevalence; it says nothing about R^2, and inventing a
        # number here would put a meaningless column beside a meaningful one.
        has_pos = bool(group["n_test_pos"].notna().any())
        n_pos = int(group["n_test_pos"].max()) if has_pos else -1
        meta = {"block": block, "task": task, "shape": shape, "readout": readout,
                "readout_family": readout_family, "metric": metric,
                "n_test": n_test, "n_test_pos": n_pos,
                "prevalence": (n_pos / n_test) if has_pos else np.nan,
                "features_only": features_only,
                "reportable": bool(block == "v1" or task in REPORTABLE_MENU)}
        for family, fam in group.groupby("family", sort=False):
            if family == "features":
                continue
            def against_base(rows: pd.DataFrame) -> dict[int, float]:
                """Per-seed delta against the engineered arm, paired by seed where that arm has seeds."""
                out = {}
                for _, c in rows.iterrows():
                    s = int(c["seed"])
                    out[s] = float(c[metric]) - base_by_seed.get(s, features_only)
                return out

            fusion = against_base(fam[fam["arm_set"] == "features_plus_mu"])
            mu_only = against_base(fam[fam["arm_set"] == "mu"])
            if fusion:
                # key must carry EVERY field the arm-contrast loop unpacks, readout_family included:
                # the fbwd and off deltas are only comparable within one readout AND one estimator.
                fusion_by_family.setdefault(family, {})[
                    (block, task, shape, readout, readout_family)] = fusion
                rows.append({**meta, "contrast": "fusion_minus_features", "family": family,
                             "spread_note": ("paired per seed against the engineered arm" if paired_base
                                             else "mu-side only (features_only is seedless)"),
                             **paired_delta(fusion)})
            if mu_only:
                rows.append({**meta, "contrast": "mu_minus_features", "family": family,
                             "spread_note": "mu-side only (features_only is seedless)", **paired_delta(mu_only)})

    # The arm contrast: features_only cancels exactly, so this is the ONE statistic here carrying both
    # arms' seed spreads (F17). It is also R1's most robust result, so it gets its own rows.
    fbwd = next((f for f in fusion_by_family if f.endswith("_fbwd")), None)
    off = next((f for f in fusion_by_family if f.endswith("_off")), None)
    if fbwd and off:
        for key, a in fusion_by_family[fbwd].items():
            b = fusion_by_family[off].get(key)
            if not b:
                continue
            block, task, shape, readout, readout_family = key
            shared = {s: a[s] - b[s] for s in sorted(set(a) & set(b))}
            if not shared:
                continue
            rows.append({"block": block, "task": task, "shape": shape, "readout": readout,
                         "readout_family": readout_family,
                         "metric": HEADLINE[shape], "contrast": "fusion_fbwd_minus_off",
                         "family": f"{fbwd}_vs_off", "spread_note": "both arms (F17)",
                         "reportable": bool(block == "v1" or task in REPORTABLE_MENU), **paired_delta(shared)})
    return pd.DataFrame(rows)


def reproduction_check(probe: pd.DataFrame) -> float | None:
    """Footing check: the caches must reproduce a published number before anything new is read.

    The target is a SIX-SEED mean, so a partial fan is not a failed reproduction -- it is a different
    quantity, and asserting on it would fire on every smoke run and teach us to ignore the gate. A
    short fan is therefore SKIPPED loudly rather than passed quietly or failed wrongly (seed 0 alone
    reads 0.7692 against the 6-seed 0.7710, which is seed spread, not cache drift).
    """
    sel = probe[(probe["family"] == REPRO_TARGET["cell"]) & (probe["task"] == REPRO_TARGET["task"])
                & (probe["readout"] == REPRO_TARGET["readout"]) & (probe["arm_set"] == "mu")]
    if len(sel) < REPRO_TARGET["n_seeds"]:
        log.warning(f"REPRO SKIPPED: {len(sel)}/{REPRO_TARGET['n_seeds']} seeds present. The published "
                    f"{REPRO_TARGET['value']:.4f} is a {REPRO_TARGET['n_seeds']}-seed mean and this run "
                    f"is not certified against it.")
        return None
    got = float(sel["pr_auc"].mean())
    diff = abs(got - REPRO_TARGET["value"])
    log.info(f"REPRO {REPRO_TARGET['cell']} {REPRO_TARGET['task']} @{REPRO_TARGET['readout']}: "
             f"{got:.4f} vs published {REPRO_TARGET['value']:.4f} (|diff| {diff:.5f}) over {len(sel)} seeds")
    assert diff < REPRO_TARGET["tol"], f"cache reproduction FAILED: {got} vs {REPRO_TARGET['value']}"
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description="F1: fusion readout on all 11 tasks at the N1 readouts.")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--cache-dir", default=None, help="pool mu cache; default exp08_menu_channel/mu_cache")
    ap.add_argument("--subset-cache-dir", default=None, help="default exp08_menu_channel/subset_mu_cache")
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS), choices=list(READOUTS))
    ap.add_argument("--families", nargs="+", default=["linear"], choices=list(FAMILIES),
                    help="readout family. `linear` is the D18 headline; gbm/mlp are the same-readout "
                         "CONTROL against C3's nonlinear feature baseline and run at `mean` only.")
    ap.add_argument("--blocks", nargs="+", default=["v1", "menu"], choices=["v1", "menu"])
    ap.add_argument("--out-dir", default="experiments/f1_fusion_scorecard")
    ap.add_argument("--summary-only", action="store_true",
                    help="recompute the summary/absolute tables from an existing f1_probe.csv. The probe "
                         "stage is ~35 min of logistic fits and the summary is seconds, so a summary bug "
                         "must not cost a re-score.")
    args = ap.parse_args()

    home = repo_root / "experiments" / "exp08_menu_channel"
    cache_dir = Path(args.cache_dir) if args.cache_dir else home / "mu_cache"
    subset_cache_dir = Path(args.subset_cache_dir) if args.subset_cache_dir else home / "subset_mu_cache"
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"pool cache {cache_dir} | subset cache {subset_cache_dir} | readouts {args.readouts}")

    if args.summary_only:
        probe = pd.read_csv(out_dir / "f1_probe.csv")
        log.info(f"summary-only: loaded {len(probe)} probe rows from {out_dir / 'f1_probe.csv'}")
        if "v1" in args.blocks and "linear" in probe["readout_family"].unique():
            reproduction_check(probe[probe["readout_family"] == "linear"])
        summarize(probe).to_csv(out_dir / "f1_summary.csv", index=False)
        absolute_rows(probe).to_csv(out_dir / "f1_absolute.csv", index=False)
        log.info(f"rewrote {out_dir}/f1_{{summary,absolute}}.csv")
        return 0

    labels_menu = label_frame()
    v1_labels = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
                 [["tic_id", *V1_TASKS]].drop_duplicates("tic_id"))
    subset_feats = cached_subset_features(repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    pool_feats = None
    if "menu" in args.blocks:
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        pool_feats = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                          repo_root / "processed" / "sequences", None)

    probe_rows: list[dict] = []
    # The engineered arm is arm-independent, so it is scored once per key rather than 13 times. But it is
    # only SEEDLESS under `linear`: logistic/ridge are deterministic, while GBM and MLP are not. Keying
    # the nonlinear families by seed too is what lets the delta be PAIRED (fusion seed s minus features
    # seed s). Differencing 6 fusion seeds against one features fit injects a constant offset -- measured
    # at up to 0.0062 here, against nonlinear deltas of 0.012-0.041, i.e. enough to flip a verdict.
    features_done: set[tuple] = set()
    jobs = [(arm, readout, rf) for arm in args.arms for readout in args.readouts
            for rf in args.families
            # a nonlinear readout is a CONTROL on the headline pooling only; running it at mean_std or
            # mean_perp_amp would multiply cost for a question the control does not ask
            if rf == "linear" or readout == "mean"]
    for arm, readout, rf in tqdm(jobs, desc="arm x readout x family", total=len(jobs)):
        family, seed = arm_parts(arm)
        clf, reg = FAMILIES[rf]
        keep_sets = None if rf == "linear" else set(NONLINEAR_ARM_SETS)
        subset_mu = load_mu_cache(subset_cache_dir / f"{arm}.npz")
        subset_aligned = align_features(subset_feats, subset_mu)
        v1_sets = arm_tables(subset_mu, subset_aligned, readout, ["train", "test"])

        if "v1" in args.blocks:
            for arm_set, table in v1_sets.items():
                if keep_sets is not None and arm_set not in keep_sets:
                    continue
                if arm_set == "features_only":
                    key = ("v1", readout, rf) if rf == "linear" else ("v1", readout, rf, seed)
                    if key in features_done:
                        continue
                    features_done.add(key)
                    label = ("features", "features", -1 if rf == "linear" else seed)
                else:
                    label = (arm, family, seed)
                for row in v1_rows(table, v1_labels, ["train", "test"], rf, seed):
                    probe_rows.append({"block": "v1", "arm": label[0], "family": label[1],
                                       "seed": label[2], "readout": readout, "readout_family": rf,
                                       "arm_set": arm_set, **row})

        if "menu" in args.blocks:
            pool_mu = load_mu_cache(cache_dir / f"{arm}.npz")
            pool_aligned = align_features(pool_feats, pool_mu)
            menu_sets = arm_tables(pool_mu, pool_aligned, readout, ["train", "test"])
            for arm_set in menu_sets:
                if keep_sets is not None and arm_set not in keep_sets:
                    continue
                if arm_set == "features_only":
                    key = ("menu", readout, rf) if rf == "linear" else ("menu", readout, rf, seed)
                    if key in features_done:
                        continue
                    features_done.add(key)
                    label = ("features", "features", -1 if rf == "linear" else seed)
                else:
                    label = (arm, family, seed)
                for row in menu_rows(menu_sets[arm_set], v1_sets[arm_set], labels_menu, clf, reg, seed):
                    probe_rows.append({"block": "menu", "arm": label[0], "family": label[1],
                                       "seed": label[2], "readout": readout, "readout_family": rf,
                                       "arm_set": arm_set, **row})

    probe = pd.DataFrame(probe_rows)
    probe["pool_cache"] = str(cache_dir)
    probe["subset_cache"] = str(subset_cache_dir)
    probe.to_csv(out_dir / "f1_probe.csv", index=False)
    log.info(f"wrote {out_dir / 'f1_probe.csv'} ({len(probe)} rows)")

    if "v1" in args.blocks and "mean" in args.readouts and "linear" in args.families:
        reproduction_check(probe[probe["readout_family"] == "linear"])

    summary = summarize(probe)
    summary.to_csv(out_dir / "f1_summary.csv", index=False)
    absolutes = absolute_rows(probe)
    absolutes.to_csv(out_dir / "f1_absolute.csv", index=False)
    log.info(f"wrote {out_dir / 'f1_summary.csv'} ({len(summary)} rows) and "
             f"f1_absolute.csv ({len(absolutes)} rows)")

    head = summary[(summary["contrast"] == "fusion_minus_features") & summary["reportable"]
                   & (summary["readout"] == "mean")]
    for rf in args.families:
        view = head[head["readout_family"] == rf]
        if view.empty:
            continue
        print(f"\nfusion delta (features_plus_mu - features_only), readout `mean`, "
              f"readout_family={rf}, 11 tasks:")
        print(view.pivot_table(index=["block", "task"], columns="family", values="delta_mean")
              .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
