"""S1 -- label-efficiency curves on cached mu. Roadmap 2026-08-25 stretch row S1 (provenance YM-W6).

THE QUESTION. W6 says a "semi-supervised" claim has to be paid for in label reduction: "如果claim是semi
supervised，那就是看能减少多少有label的数据的使用". F1 measured the fusion advantage
(`features (+) mu` over `features`) at ONE point on the label axis -- 100 % of the train split. S1 asks
whether that advantage GROWS as the label budget shrinks, which is the only version of the claim that
supports the semi-supervised framing as a figure. The paper's claim (D16/D19) does not change either
way; this is a supporting panel, and a flat or reversed curve prints exactly as honestly.

PRE-REGISTERED EXPECTATION, fixed before any score was read:
    S1-E1  the fusion delta at each task's SMALLEST admissible label budget exceeds its delta at full n
           by more than the seed-paired 2*SE, on >= 6 of the 11 tasks
           --> "the frozen representation is worth more when labels are scarce", the W6 panel works.
    S1-E2  otherwise --> report the widen/flat/narrow split verbatim. ML4PS welcomes nulls (tips T6).

WHAT THIS REUSES RATHER THAN RE-DERIVES. `analyze_exp10_fe_subsample.py` already established the
subsampling estimator (stratified star-level draws, test set never touched, encoder-seed spread and
draw spread reported side by side and never merged); this script generalises it from 3 tasks x 2 arms
to 11 x 5. The arm tables come from `analyze_f1_fusion_scorecard.pool`/`concat` verbatim, so the inputs
to every readout are bit-identical to F1's, and the FULL-BUDGET row of this script is therefore not a
new measurement at all -- it must reproduce `f1_absolute.csv` exactly. That is gate FOOTING-1 below.

THE LADDER (user decision 2026-09-01). The roadmap sketched fractions {1, 10, 100} %. A fraction means
20x different label counts across these tasks (1 % is 160 stars for `osc_giant` and 8 for
`rgb_vs_heb`), so the grid is the UNION of the fraction ladder and an absolute-count ladder:
    fractions  {1, 3, 10, 30, 100} % of the task's own eligible TRAIN set
    absolute   {50, 100, 300, 1000, 3000} training stars
deduped, capped at the task's n_train. The roadmap's pre-registered {1, 10, 100} % points are contained
exactly, and the figure plots on absolute n so the curves are comparable across tasks.

THE FLOOR (pre-registered, user decision 2026-09-01, fixed BEFORE any score was read):
    admissible cell  <-->  n_train >= 50  AND  (regression, or n_train_pos >= 10)
The n_train clause is about the readout, not the labels: the fusion arm has 25 + 128 = 153 columns, and
below ~50 rows the fit is degenerate whatever the prevalence. Inadmissible cells are NOT scored, and
the CSV records why. Measured consequence at 1 %: `transit` (6 pos), `ijspeert` (4), `rotation` (8),
`eb` (9), `rgb_vs_heb` (8 stars), `rotation_period` (7 stars) drop out; all 11 tasks are admissible at
3 % and above.

ARMS (5), one readout (`mean`), one readout family (`linear`) -- the F1 headline cell:
    features_only               the A1 engineered ceiling. Arm-independent and seedless.
    mu                          hann0p3_fbwd seeds 0-5.
    features_plus_mu            hann0p3_fbwd seeds 0-5. THE HEADLINE ARM.
    mu / features_plus_mu       under `untrained`, the capacity-matched dilution control at every budget.
The dyn-off arm is deliberately absent: `delta(fbwd) - delta(off)` is a second story and would double
the runtime of a one-panel figure.

POPULATION. S1 inherits F1's populations unchanged -- the v1 packed subset (9,428 / 2,021 stars) and
the new-task pool (16,002 / 3,429). Both are CASE-CONTROL, so every prevalence here is inflated
relative to a survey; R8 measured absolutes falling 60-71 % under `survey_matched`. Deltas keep their
sign under that reweighting and magnitudes do not (R8), so no absolute score in this table transfers to
a survey population. Carried on every row as `population_note`.

FLARE. Scored with F1's label set (`flare_ever` via `new_task_scorecard.label_frame`), NOT L1's
corrected 2x2. It stays UNPRINTABLE until L1's visual gate lands (STATUS 2026-08-26d, reporting rule
B), so its rows carry printable=False and it is excluded from the S1-E1 count.

ESTIMATOR CONVENTIONS THAT BIND:
  - subsample the TRAIN split only, at STAR level, stratified (binary label, or target quartile for the
    two regressions). The test set is byte-identical at every budget, so n_test / n_test_pos reproduce
    F1's published values at every point on the curve -- gate FOOTING-2.
  - the draw index is shared across ALL FIVE ARMS within a cell, so a delta is differenced on identical
    rows and the draw noise cancels.
  - two spreads, side by side, never merged: seed spread (average over draws first, then sd over the 6
    encoder seeds) is the headline error bar, matching every other table in this project; draw spread
    (average over seeds first, then sd over draws) is reported beside it.
  - the full-budget level has nothing to resample AND a deterministic readout, so it runs one draw and
    the CSV says so. Its draw spread is 0 by construction, not by measurement.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~25 min at --jobs 10):
    PYTHONUNBUFFERED=1 python experiments/analyze_s1_label_efficiency.py
    python experiments/analyze_s1_label_efficiency.py --tasks eb --draws 3 --seeds 0
    python experiments/analyze_s1_label_efficiency.py --summary-only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Pin BLAS to one thread BEFORE numpy is imported. This is not a performance knob: `lbfgs` on the
# 153-column fusion design matrix is ill-conditioned enough that the reduction order inside the BLAS
# dot products changes the fitted coefficients, and joblib already pins its workers to one thread. Left
# unpinned, this script returns different numbers at --jobs 1 than at --jobs 12. Measured on
# flare/untrained/features_plus_mu: 0.4808140093533278 single-threaded vs 0.4798055587265005
# multi-threaded, and the single-threaded value reproduces a joblib worker's to 16 digits.
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse F1's arm tables rather than fork them

from analyze_exp08_menu_channel import align_features, arm_parts, load_mu_cache, stacked  # noqa: E402
from analyze_f1_fusion_scorecard import concat, pool  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features  # noqa: E402
from swm.eval.new_task_scorecard import label_frame, score_regression  # noqa: E402
from swm.eval.readout_sweep import fit_readout_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("s1_label_efficiency")

source_home = repo_root / "experiments" / "exp08_menu_channel"
V1_TASKS = ("pulsating", "eb", "rotation", "transit")
TASK_ORDER = ("pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "flare",
              "rgb_vs_heb", "ijspeert", "numax_hon", "rotation_period")
# `subset` = the frozen v1 packed subset (9,428 / 2,021 stars); `pool` = the new-task pool
# (16,002 / 3,429). rotation_period and ijspeert ride the subset, not the pool, exactly as in F1.
TASK_POPULATION = {"pulsating": "subset", "eb": "subset", "rotation": "subset", "transit": "subset",
                   "osc_giant": "pool", "solar_like_osc": "pool", "flare": "pool",
                   "rgb_vs_heb": "pool", "ijspeert": "subset", "numax_hon": "pool",
                   "rotation_period": "subset"}
FRACTIONS = (0.01, 0.03, 0.10, 0.30, 1.00)   # roadmap S1 asks for {1, 10, 100}; 3 and 30 fill the gap
ABSOLUTE = (50, 100, 300, 1000, 3000)        # so a budget means the same thing across tasks
MIN_TRAIN = 50                               # 153 fusion columns: below this the fit is degenerate
MIN_TRAIN_POS = 10                           # detection/contrastive only; regression has no positives
N_STRATA_REGRESSION = 4                      # target quartiles, the regression analogue of class strata
SKIP_COLUMNS = ("task", "n_target", "n_actual", "n_train_pos", "frac", "provenance", "is_full",
                "admissible", "floor_reason", "population")
CELL_NAME = "hann0p3_fbwd"                   # D17-CLOSED 2026-08-26: hann0p3 ships
UNTRAINED_ARM = "untrained"
POPULATION_NOTE = ("F1 scorecard population, CASE-CONTROL: prevalence is inflated relative to a survey "
                   "(R8 measured absolutes falling 60-71% under survey_matched, deltas keeping sign). "
                   "No absolute score here transfers to a survey population.")
# flare rides F1's flare_ever labels and is unprintable until L1's visual gate lands (STATUS 2026-08-26d).
UNPRINTABLE = {"flare"}


# ------------------------------------------------------------------------------------- task registry
def v1_label_frame() -> pd.DataFrame:
    """The four v1 variability flags, keyed by tic, exactly the table F1's v1 block merges against."""
    frame = pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
    return frame[["tic_id", *V1_TASKS]].drop_duplicates("tic_id").set_index("tic_id")


def rotation_period_targets() -> pd.Series:
    """TARS rotation periods under the frozen 5 d scope cap, the target `score_rotation_period_from_mu` builds."""
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["rotation"] = pd.to_numeric(canon["rotation"], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    keep = canon.loc[(canon["rotation"] == 1) & canon["rotation_period"].notna()
                     & (canon["rotation_period"] <= 5), ["tic_id", "rotation_period"]]
    return keep.set_index("tic_id")["rotation_period"]


def ijspeert_positives() -> set[int]:
    """Ijspeert+2024 OBA eclipsing-binary TICs; membership IS the label, so every subset star is eligible."""
    cat = pd.read_csv(repo_root / "labels" / "external" / "ijspeert2024_bright.csv")
    return set(cat["TIC"].astype(int))


def task_specs(tics: dict[str, dict[str, list[int]]], tasks: list[str]) -> dict[str, dict]:
    """Per task: which population, which rows of it are eligible, and the target on those rows.

    The keep-masks here are RE-DERIVED from the same label sources the shipped scorers read, because a
    stratified draw has to know the eligible set and the scorers apply their masks internally. Getting
    that wrong would silently move the population, so it is checked twice rather than argued: the test
    masks must reproduce F1's published n_test / n_test_pos (FOOTING-2), and the full-budget scores must
    reproduce f1_absolute.csv (FOOTING-1). Both run before any curve is read.
    """
    menu = label_frame()
    v1 = v1_label_frame()
    ijspeert = ijspeert_positives()
    prot = rotation_period_targets()
    specs: dict[str, dict] = {}

    def add(task: str, shape: str, metric: str, selector, log_target: bool = False) -> None:
        if task not in tasks:
            return
        population = TASK_POPULATION[task]
        entry = {"task": task, "population": population, "shape": shape, "metric": metric,
                 "log_target": log_target}
        for split in ("train", "test"):
            keep, target = selector(tics[population][split])
            entry[f"{split}_index"] = np.flatnonzero(keep)
            entry[f"{split}_y"] = np.asarray(target, dtype=float)[np.flatnonzero(keep)]
        specs[task] = entry

    for task in V1_TASKS:
        def v1_selector(star_ids: list[int], column: str = task):
            values = pd.to_numeric(v1[column].reindex(star_ids), errors="coerce").fillna(0).astype(int)
            return np.ones(len(star_ids), dtype=bool), values.to_numpy()
        add(task, "detection", "pr_auc", v1_selector)

    for task in ("osc_giant", "solar_like_osc", "flare"):
        def menu_selector(star_ids: list[int], column: str = task):
            values = pd.to_numeric(menu[column].reindex(star_ids), errors="coerce").fillna(0).astype(int)
            return np.ones(len(star_ids), dtype=bool), values.to_numpy()
        add(task, "detection", "pr_auc", menu_selector)

    def rgb_selector(star_ids: list[int]):
        target = menu["rgb_vs_heb"].reindex(star_ids)
        return target.notna().to_numpy(), target.fillna(0).astype(int).to_numpy()
    add("rgb_vs_heb", "contrastive", "roc_auc", rgb_selector)

    def ijspeert_selector(star_ids: list[int]):
        values = []
        for star in star_ids:
            values.append(int(int(star) in ijspeert))
        return np.ones(len(star_ids), dtype=bool), np.asarray(values)
    add("ijspeert", "detection", "pr_auc", ijspeert_selector)

    def numax_selector(star_ids: list[int]):
        target = menu["numax_hon"].reindex(star_ids)
        return target.notna().to_numpy(), np.log10(target.fillna(1.0).to_numpy(dtype=float))
    add("numax_hon", "regression", "r2", numax_selector, log_target=True)

    def prot_selector(star_ids: list[int]):
        target = prot.reindex(star_ids)
        return target.notna().to_numpy(), target.fillna(0.0).to_numpy(dtype=float)
    add("rotation_period", "regression", "r2", prot_selector)
    return specs


# ------------------------------------------------------------------------------------------- ladder
def strata_for(spec: dict) -> np.ndarray:
    """Stratification key on the eligible TRAIN rows: the class label, or target quartile for regressions.

    Stratifying is what makes the small budgets about n rather than about luck -- at 50 stars an
    unstratified draw of a 6 % prevalence task can land with two positives, and the resulting spread
    would be the draw's, not the label budget's.
    """
    y = spec["train_y"]
    if spec["shape"] == "regression":
        edges = np.quantile(y, np.linspace(0, 1, N_STRATA_REGRESSION + 1)[1:-1])
        return np.digitize(y, edges)
    return y.astype(int)


def ladder(spec: dict) -> list[dict]:
    """The admissible label budgets for one task, with each level's provenance and its floor verdict."""
    n_train = len(spec["train_y"])
    strata = strata_for(spec)
    levels: dict[int, list[str]] = {}
    for fraction in FRACTIONS:
        size = int(round(fraction * n_train))
        levels.setdefault(min(size, n_train), []).append(f"{fraction:g}x")
    for size in ABSOLUTE:
        if size < n_train:
            levels.setdefault(size, []).append(f"n{size}")
    rows = []
    for size in sorted(levels):
        index = stratified_draw(strata, size, draw=0)
        n_pos = int(spec["train_y"][index].sum()) if spec["shape"] != "regression" else -1
        admissible = len(index) >= MIN_TRAIN and (spec["shape"] == "regression" or n_pos >= MIN_TRAIN_POS)
        reason = ""
        if len(index) < MIN_TRAIN:
            reason = f"n_train {len(index)} < {MIN_TRAIN}"
        elif spec["shape"] != "regression" and n_pos < MIN_TRAIN_POS:
            reason = f"n_train_pos {n_pos} < {MIN_TRAIN_POS}"
        rows.append({"n_target": size, "n_actual": int(len(index)), "n_train_pos": n_pos,
                     "frac": size / n_train, "provenance": "|".join(levels[size]),
                     "is_full": size == n_train, "admissible": admissible, "floor_reason": reason})
    return rows


def stratified_draw(strata: np.ndarray, n_target: int, draw: int) -> np.ndarray:
    """Row indices of one train subsample that preserves the task's stratum proportions.

    Each stratum keeps at least one row, so the returned size can exceed `n_target` by a few rows on a
    many-stratum task; the realised size is recorded per cell rather than assumed.
    """
    rng = np.random.default_rng(20260901 + draw)
    keep = []
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        take = int(round(n_target * len(members) / len(strata)))
        take = max(1, min(take, len(members)))
        keep.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(keep))


# ------------------------------------------------------------------------------------------ scoring
def score_cell(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
               shape: str) -> float:
    """One (arm, budget, draw) fit under the F1 headline readout: logistic for labels, RidgeCV for targets."""
    if shape == "regression":
        metrics, _ = score_regression(x_train, y_train, x_test, y_test, "ridge", 0)
        return float(metrics["r2"])
    scores = fit_readout_scores("logistic", x_train, y_train, x_test, 0)
    if shape == "contrastive":
        return float(roc_auc_score(y_test, scores))
    return float(average_precision_score(y_test, scores))


def run_cell(task: str, shape: str, metric: str, level: dict, draw: int, index: np.ndarray,
             arm_set: str, family: str, seed: int, x_train, y_train, x_test, y_test) -> dict:
    """One (task, budget, draw, arm) cell, at module level so joblib can ship it to a worker."""
    score = score_cell(x_train[index], y_train[index], x_test, y_test, shape)
    if level["is_full"]:
        draw_kind = "full budget: nothing to resample and the readout is deterministic"
    else:
        draw_kind = "stratified train resample"
    return {"task": task, "shape": shape, "metric": metric, "arm_set": arm_set, "family": family,
            "seed": seed, "n_target": level["n_target"], "frac": level["frac"],
            "provenance": level["provenance"], "is_full": level["is_full"], "draw": draw,
            "n_train": int(len(index)),
            "n_train_pos": int(y_train[index].sum()) if shape != "regression" else -1,
            "n_test": int(len(y_test)), "n_test_pos": int(y_test.sum()) if shape != "regression" else -1,
            "score": score, "draw_kind": draw_kind}


def arm_matrices(task_spec: dict, tables: dict[str, dict[str, dict]]) -> dict[tuple, tuple]:
    """Every arm's (train, test) design matrix restricted to one task's eligible rows.

    Keyed (arm_set, family, seed). `features_only` appears once: it carries no encoder, so scoring it
    per seed would report a fake six-fold replication of one deterministic fit.
    """
    population = task_spec["population"]
    train_index, test_index = task_spec["train_index"], task_spec["test_index"]
    out = {}
    for (arm_set, family, seed), table in tables[population].items():
        out[(arm_set, family, seed)] = (table["train"][train_index], table["test"][test_index])
    return out


def task_rows(task: str, spec: dict, tables: dict, draws: int,
              jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every admissible (budget, draw, arm) cell for one task, plus the budgets the floor rejected."""
    strata = strata_for(spec)
    matrices = arm_matrices(spec, tables)
    y_train, y_test = spec["train_y"], spec["test_y"]
    payload, skipped = [], []
    for level in ladder(spec):
        if not level["admissible"]:
            skipped.append({"task": task, **level, "population": spec["population"]})
            continue
        n_draws = 1 if level["is_full"] else draws
        for draw in range(n_draws):
            if level["is_full"]:
                index = np.arange(len(y_train))
            else:
                index = stratified_draw(strata, level["n_target"], draw)
            for (arm_set, family, seed), (x_train, x_test) in matrices.items():
                payload.append(delayed(run_cell)(task, spec["shape"], spec["metric"], level, draw,
                                                 index, arm_set, family, seed, x_train, y_train,
                                                 x_test, y_test))
    results = Parallel(n_jobs=jobs)(tqdm(payload, desc=f"cells[{task}]", total=len(payload)))
    frame = pd.DataFrame(results)
    frame["population"] = spec["population"]
    # named columns even when nothing was rejected: an empty frame otherwise writes a headerless file,
    # and "this task had no budget below the floor" then reads back as a corrupt shard
    return frame, pd.DataFrame(skipped, columns=SKIP_COLUMNS)


# ------------------------------------------------------------------------------------------ summary
def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Per (task, budget) fusion and mu deltas against the engineered arm, both spreads kept apart.

    The delta is differenced WITHIN a draw, so the draw noise cancels exactly and what is left is the
    encoder-seed spread the rest of this project reports. The draw spread is carried beside it and never
    pooled into the same error bar: one is representation noise, the other is which stars got labelled.
    """
    rows = []
    for (task, n_target), group in probe.groupby(["task", "n_target"], sort=False):
        shape = group["shape"].iloc[0]
        metric = group["metric"].iloc[0]
        base = group[group["arm_set"] == "features_only"].set_index("draw")["score"]
        meta = {"task": task, "shape": shape, "metric": metric,
                "population": group["population"].iloc[0], "n_target": n_target,
                "frac": float(group["frac"].iloc[0]), "provenance": group["provenance"].iloc[0],
                "is_full": bool(group["is_full"].iloc[0]),
                "n_train": int(group["n_train"].median()),
                "n_train_pos": int(group["n_train_pos"].median()),
                "n_test": int(group["n_test"].max()), "n_test_pos": int(group["n_test_pos"].max()),
                "n_draws": int(group["draw"].nunique()),
                "features_only": float(base.mean()),
                "printable": task not in UNPRINTABLE, "population_note": POPULATION_NOTE}
        for (arm_set, family), arm in group.groupby(["arm_set", "family"], sort=False):
            if arm_set == "features_only":
                continue
            arm = arm.copy()
            arm["delta"] = arm["score"].to_numpy() - base.reindex(arm["draw"]).to_numpy()
            per_seed = arm.groupby("seed")["delta"].mean()
            per_draw = arm.groupby("draw")["delta"].mean()
            seed_sd = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan
            draw_sd = float(per_draw.std(ddof=1)) if len(per_draw) > 1 else np.nan
            rows.append({**meta, "arm_set": arm_set, "family": family,
                         "n_seeds": int(len(per_seed)),
                         "score_mean": float(arm["score"].mean()),
                         "delta_mean": float(arm["delta"].mean()),
                         "seed_sd": seed_sd,
                         "seed_2se": 2 * seed_sd / np.sqrt(len(per_seed)) if len(per_seed) > 1 else np.nan,
                         "draw_sd": draw_sd,
                         "draw_2se": 2 * draw_sd / np.sqrt(len(per_draw)) if len(per_draw) > 1 else np.nan,
                         "frac_draws_positive": float((per_draw > 0).mean()),
                         "beats_seed_2se": bool(len(per_seed) > 1
                                                and arm["delta"].mean() > 2 * seed_sd / np.sqrt(len(per_seed)))})
    return pd.DataFrame(rows)


def growth_verdict(probe: pd.DataFrame) -> pd.DataFrame:
    """S1-E1 scored verbatim: fusion delta at the smallest admissible budget vs at the full budget.

    The difference is paired per encoder seed (the same six seeds appear at both budgets), so its 2*SE
    is the sd of the six per-seed differences -- not the two levels' error bars added, which would
    double-count the shared seed effect.
    """
    rows = []
    fusion = probe[(probe["arm_set"] == "features_plus_mu") & (probe["family"] == CELL_NAME)]
    for task, group in fusion.groupby("task", sort=False):
        base = group[group["arm_set"] == "features_plus_mu"]
        low_n = int(base["n_target"].min())
        full_n = int(base.loc[base["is_full"], "n_target"].max())
        if low_n == full_n:
            continue
        per_seed = {}
        for level, tag in ((low_n, "low"), (full_n, "full")):
            sel = base[base["n_target"] == level]
            deltas = sel.groupby("seed").apply(
                lambda block: block["score"].mean() - block["features_baseline"].mean(),
                include_groups=False)
            per_seed[tag] = deltas
        diff = (per_seed["low"] - per_seed["full"]).dropna()
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
        two_se = 2 * sd / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        if diff.mean() > two_se:
            call = "widens"
        elif diff.mean() < -two_se:
            call = "narrows"
        else:
            call = "flat"
        rows.append({"task": task, "n_low": low_n, "n_full": full_n,
                     "delta_low": float(per_seed["low"].mean()), "delta_full": float(per_seed["full"].mean()),
                     "growth": float(diff.mean()), "growth_2se": two_se, "call": call,
                     "printable": task not in UNPRINTABLE})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- footing gates
def footing_full_budget(summary: pd.DataFrame) -> pd.DataFrame:
    """FOOTING-1: the full-budget row is F1's published cell and must reproduce f1_absolute.csv.

    Nothing in this script's full-budget path is new -- same stars, same columns, same readout -- so a
    deviation means the population or the arm table drifted, and no curve below it would be measuring
    what its axis says. TWO clauses, because the arithmetic is exactly reproducible for one arm and not
    for the others:

    1a EXACT, |diff| < 1e-9, on the `features_only` arm. 25 well-conditioned columns and a
       deterministic logistic fit: this is the clause that actually tests the population, the keep
       masks and the feature table, and it has no numerical excuse available to it.
    1b BOUNDED, |diff| < 5e-3, on the mu-bearing arms. These carry 128 or 153 columns and the lbfgs
       solve is ill-conditioned enough that the BLAS thread count changes the fitted coefficients (see
       the pinning block at the top of this file). S1 runs single-threaded throughout; F1 did not, so a
       residual is EXPECTED here and its size is reported rather than assumed. Measured 2026-09-01: 29
       of 55 rows exact, 45 under 1e-4, max 1.0e-3 on untrained/features_plus_mu -- the most
       ill-conditioned cell in the table, which is the ordering the mechanism predicts. This is the
       same class of term F1 already recorded for its two mu caches (3.5e-4 from cuDNN nondeterminism,
       moving 6-seed PR-AUC by <= 1e-4).

    IT DOES NOT TOUCH ANY S1-INTERNAL COMPARISON. Every cell on every curve is fitted in the same
    single-threaded regime, so the term cancels exactly in each fusion delta; it survives only in this
    cross-artifact check against a differently-threaded run.
    """
    path = repo_root / "experiments" / "f1_fusion_scorecard" / "f1_absolute.csv"
    if not path.exists():
        log.warning("f1_absolute.csv absent; FOOTING-1 skipped and this run is NOT certified")
        return pd.DataFrame()
    ref = pd.read_csv(path)
    ref = ref[(ref["readout"] == "mean") & (ref["readout_family"] == "linear")]
    rows = []
    for _, cell in summary[summary["is_full"]].iterrows():
        for arm_set, family, value, n_seeds in (
                ("features_only", "features", cell["features_only"], 1),
                (cell["arm_set"], cell["family"], cell["score_mean"], cell["n_seeds"])):
            match = ref[(ref["task"] == cell["task"]) & (ref["arm_set"] == arm_set)
                        & (ref["family"] == family)]
            if match.empty:
                continue
            rows.append({"task": cell["task"], "arm_set": arm_set, "family": family,
                         "metric": cell["metric"], "s1": float(value),
                         "f1": float(match["score_mean"].iloc[0]),
                         "s1_n_seeds": int(n_seeds), "f1_n_seeds": int(match["n_seeds"].iloc[0])})
    out = pd.DataFrame(rows).drop_duplicates(["task", "arm_set", "family"])
    if out.empty:
        return out
    out["abs_diff"] = (out["s1"] - out["f1"]).abs()
    # F1's published number is a mean over ITS seed fan. A short fan here is a different quantity, not
    # a failed reproduction: asserting on it would fire on every smoke run and teach us to ignore the
    # gate (the exp09 REPRO_SKIPPED precedent). Partial rows are reported and excluded from the assert.
    out["comparable"] = out["s1_n_seeds"] == out["f1_n_seeds"]
    out["clause"] = np.where(out["arm_set"] == "features_only", "1a exact", "1b bounded")
    log.info(f"FOOTING-1 full-budget vs f1_absolute.csv: {int(out['comparable'].sum())} of {len(out)} "
             f"rows comparable, {int((out['abs_diff'] < 1e-9).sum())} exact, "
             f"{int((out['abs_diff'] < 1e-4).sum())} under 1e-4, max {out['abs_diff'].max():.2e}\n"
             + out.sort_values("abs_diff", ascending=False).head(5).round(8).to_string(index=False))
    strict = out[out["comparable"]]
    if len(strict) < len(out):
        log.warning(f"FOOTING-1 PARTIAL: {len(out) - len(strict)} rows ran a short seed fan and are "
                    "NOT certified against F1. Re-run the full 6-seed fan before reading any curve.")
    exact = strict[strict["clause"] == "1a exact"]
    bounded = strict[strict["clause"] == "1b bounded"]
    assert exact.empty or exact["abs_diff"].max() < 1e-9, \
        f"FOOTING-1a FAILED (population/feature drift), max |diff| {exact['abs_diff'].max()}"
    assert bounded.empty or bounded["abs_diff"].max() < 5e-3, \
        f"FOOTING-1b FAILED, max |diff| {bounded['abs_diff'].max()} exceeds the BLAS-threading term"
    return out


def footing_test_population(specs: dict[str, dict]) -> pd.DataFrame:
    """FOOTING-2: the re-derived test keep-masks must reproduce F1's published n_test / n_test_pos.

    The masks are re-derived here because a stratified draw needs the eligible set, and the shipped
    scorers apply their masks internally. This is the check that the re-derivation did not quietly move
    the population -- it runs on the TEST split, which S1 never subsamples.
    """
    path = repo_root / "experiments" / "f1_fusion_scorecard" / "f1_absolute.csv"
    if not path.exists():
        log.warning("f1_absolute.csv absent; FOOTING-2 skipped")
        return pd.DataFrame()
    ref = pd.read_csv(path)
    ref = ref[(ref["readout"] == "mean") & (ref["readout_family"] == "linear")]
    rows = []
    for task, spec in specs.items():
        match = ref[ref["task"] == task]
        if match.empty:
            continue
        n_pos = int(spec["test_y"].sum()) if spec["shape"] != "regression" else -1
        rows.append({"task": task, "s1_n_test": int(len(spec["test_y"])),
                     "f1_n_test": int(match["n_test"].max()),
                     "s1_n_test_pos": n_pos, "f1_n_test_pos": int(match["n_test_pos"].max())})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ok"] = (out["s1_n_test"] == out["f1_n_test"]) & (out["s1_n_test_pos"] == out["f1_n_test_pos"])
    log.info("FOOTING-2 test population vs F1\n" + out.to_string(index=False))
    assert out["ok"].all(), f"FOOTING-2 FAILED on {out.loc[~out['ok'], 'task'].tolist()}"
    return out


# ---------------------------------------------------------------------------------------- arm tables
def build_tables(arms: list[str], populations: list[str]) -> tuple[dict, dict]:
    """Every arm's design matrices per population, built through F1's own pooling and concatenation.

    Going through `analyze_f1_fusion_scorecard.pool` / `.concat` rather than re-deriving them is what
    makes FOOTING-1 a real check: the inputs to the readout are the same objects F1 fitted on, down to
    the float32 cast the fusion cell picks up from `as_blocks`.
    """
    feats = {}
    if "subset" in populations:
        feats["subset"] = cached_subset_features(
            repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    if "pool" in populations:
        log.info("loading pool feature table (~15 s, no output until it finishes)")
        feats["pool"] = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                             repo_root / "processed" / "sequences", None)
    cache_sub = {"subset": source_home / "subset_mu_cache", "pool": source_home / "mu_cache"}

    tables: dict[str, dict] = {}
    tics: dict[str, dict] = {}
    for population in populations:
        tables[population] = {}
        for arm in tqdm(arms, desc=f"arm tables[{population}]", total=len(arms)):
            mu = load_mu_cache(cache_sub[population] / f"{arm}.npz")
            aligned = align_features(feats[population], mu)
            pooled = pool(mu, "mean", ["train", "test"])
            fused = concat(aligned, pooled, ["train", "test"])
            if population not in tics:
                tics[population] = {"train": list(mu["train"][0]), "test": list(mu["test"][0])}
                tables[population][("features_only", "features", -1)] = {
                    "train": stacked(aligned, "train"), "test": stacked(aligned, "test")}
            family, seed = arm_parts(arm)
            tables[population][("mu", family, seed)] = {
                "train": stacked(pooled, "train"), "test": stacked(pooled, "test")}
            tables[population][("features_plus_mu", family, seed)] = {
                "train": stacked(fused, "train"), "test": stacked(fused, "test")}
    return tables, tics


def main() -> int:
    ap = argparse.ArgumentParser(description="S1: label-efficiency curves on cached mu (roadmap stretch S1).")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_ORDER), choices=list(TASK_ORDER))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--draws", type=int, default=10, help="stratified train resamples per budget")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out-dir", default="experiments/s1_label_efficiency")
    ap.add_argument("--force", action="store_true", help="rescore tasks whose shard already exists")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild the summary/verdict tables from existing shards (seconds, not minutes)")
    args = ap.parse_args()

    out_dir = repo_root / args.out_dir
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    arms = []
    for seed in args.seeds:
        arms.append(f"{CELL_NAME}_s{seed}")
    arms.append(UNTRAINED_ARM)

    if not args.summary_only:
        populations = []
        for task in args.tasks:
            if TASK_POPULATION[task] not in populations:
                populations.append(TASK_POPULATION[task])
        tables, tics = build_tables(arms, populations)
        specs = task_specs(tics, list(args.tasks))
        footing_test_population(specs)
        for task in args.tasks:
            shard = shard_dir / f"{task}.csv"
            skip_shard = shard_dir / f"{task}_skipped.csv"
            if shard.exists() and not args.force:
                log.info(f"shard exists, skipping {task} (use --force to rescore)")
                continue
            frame, skipped = task_rows(task, specs[task], tables, args.draws, args.jobs)
            frame.to_csv(shard, index=False)
            skipped.to_csv(skip_shard, index=False)
            log.info(f"{task}: {len(frame)} cells, {len(skipped)} budgets below the floor -> {shard}")
        del tables

    shards = sorted(shard_dir.glob("*.csv"))
    probe_frames, skip_frames = [], []
    for shard in shards:
        if not shard.read_text().strip():
            continue  # shards written before SKIP_COLUMNS existed: empty means "nothing below the floor"
        if shard.name.endswith("_skipped.csv"):
            skip_frames.append(pd.read_csv(shard))
        else:
            probe_frames.append(pd.read_csv(shard))
    probe = pd.concat(probe_frames, ignore_index=True)
    probe["population_note"] = POPULATION_NOTE
    probe["printable"] = ~probe["task"].isin(UNPRINTABLE)
    probe.to_csv(out_dir / "s1_probe.csv", index=False)
    log.info(f"wrote {out_dir / 's1_probe.csv'} ({len(probe)} rows)")
    if skip_frames:
        pd.concat(skip_frames, ignore_index=True).to_csv(out_dir / "s1_below_floor.csv", index=False)

    # the growth verdict differences the fusion arm against the engineered arm within a draw, so the
    # baseline has to ride along on every fusion row rather than be re-joined by position later
    base = probe[probe["arm_set"] == "features_only"][["task", "n_target", "draw", "score"]]
    base = base.rename(columns={"score": "features_baseline"})
    probe = probe.merge(base, on=["task", "n_target", "draw"], how="left")

    summary = summarize(probe)
    summary.to_csv(out_dir / "s1_summary.csv", index=False)
    checks = footing_full_budget(summary)
    if not checks.empty:
        checks.to_csv(out_dir / "s1_footing.csv", index=False)

    verdict = growth_verdict(probe)
    verdict.to_csv(out_dir / "s1_growth.csv", index=False)
    log.info(f"wrote {out_dir}/s1_{{summary,growth}}.csv")

    head = summary[(summary["arm_set"] == "features_plus_mu") & (summary["family"] == CELL_NAME)]
    print("\nS1 -- fusion delta (features (+) mu minus features), readout `mean`, linear, "
          f"{CELL_NAME} 6 seeds, by label budget:")
    print(head.pivot_table(index="task", columns="n_target", values="delta_mean").round(4).to_string())

    scored = verdict[verdict["printable"]]
    counts = scored["call"].value_counts()
    print(f"\nS1-E1 scored on {len(scored)} printable tasks "
          f"(`flare` excluded, unprintable until L1's visual gate): "
          f"widens {counts.get('widens', 0)} / flat {counts.get('flat', 0)} / "
          f"narrows {counts.get('narrows', 0)}")
    print(scored.round(4).to_string(index=False))
    if counts.get("widens", 0) >= 6:
        print("\nVERDICT: S1-E1 FIRES -- the fusion advantage grows as the label budget shrinks.")
    else:
        print("\nVERDICT: S1-E2 -- S1-E1 does not fire at its pre-registered bar of 6 of 11 tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
