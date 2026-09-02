"""T2 (Lane R, Q17 limb A) -- can the frozen encoder infer rotation periods it never sees a full cycle of?

The model's input baseline is seq_len * window_size * cadence = 16 * 256 * 2 min = 5.69 d. TARS labels
span 0.1-23.6 d over our corpus, so across roughly 5.7-23.6 d we hold labels for a quantity no tensor
the encoder sees contains one full cycle of. T2 asks whether a linear readout on cached mu recovers
them anyway, and -- the part that decides whether any positive result means anything -- whether it does
so better than a random encoder and better than activity amplitude alone.

WHAT T1 SETTLED AND THIS SCRIPT INHERITS (do not re-derive):
  - buckets `<=2 d / 2-5.7 d / 5.7-13 d / >=13 d`, edges (0, 2, 5.69, 13, inf], copied verbatim from
    `src/notebooks/t1_rotation_census.ipynb` so the two tables tile the same way.
  - the two beyond-baseline buckets are reported SEPARATELY (the pre-registered rule asked for >= 1000
    in 5.7-13 d; the count is 3,357).
  - the frozen v1 subset cannot carry this (158 beyond-baseline rotators, 25 in its test split). Its
    contrast row is CITED from F1's `rotation_period` rows, not recomputed here.

POPULATION (grilled 2026-08-27, deviation from the handoff, disclosed in the README):
`r8_added` = the 106,284 R8 `added` stars = corpus-with-windows minus probe train/val/test. T1 named
`r8_scoreable` = added + existing_test = 108,305, but existing_test's mu lives in a different cache
layout and contributes 25 of the 4,546 beyond-baseline rotators (0.55%). One reader, one population.
Two properties of `added` that matter and are stated in the write-up:
  - it holds 13,789 TARS rotators spanning 0.10-23.59 d, 4,521 of them beyond the baseline.
  - it holds ZERO transit/eb/pulsating positives -- the v1 subset absorbed all of them -- so these are
    catalogue-pure rotators and no cross-class contamination control is needed.

THE CAP, STATED SO NEITHER IS INHERITED SILENTLY (handoff section 4d). `new_task_scorecard` carries two
different caps in one file: `score_rotation_period_from_mu` filters P <= 5.0 (ADR-0004) and
`score_regression_task` filters prot_kounkel at 5.7. T2 applies NEITHER. Its population is every added
star with a TARS period, uncapped, bounded above by the data at 23.59 d. That is the whole point of the
task; T3 decides what to do about the resulting disagreement with ADR-0004.

THE DENOMINATOR, MEASURED BEFORE THE RUN. R2 is sliced from one global fit, so each bucket is scored
against its OWN target variance. In log10 those SDs are 0.365 / 0.131 / 0.092 / 0.068 dex against a
global 0.595, i.e. the `>=13 d` bucket has ~8.8x less variance to explain than the pooled sample. A
positive bucket-local R2 there demands RMSE < 0.068 dex (~17% in period). Expect negative absolutes and
do not read them as the finding: G-rot is a CONTRAST between arms sharing one denominator, so the gate
still reads. Every bucket row therefore carries `target_sd`, Spearman rho and RMSE beside R2, and the
monotonic-degradation sentence must cite rho and RMSE, never R2 alone.

G-rot -- PRE-REGISTERED in `docs/roadmap/2026-08-15-post-yue-ma-roadmap.md` section P1, quoted not
rewritten: in the 5.7-13 d and >=13 d buckets, trained mu must beat BOTH the untrained floor AND the
amplitude-only baseline by > 2*SE over 6 seeds, at `mean`. Partial pass (beats untrained, not
amplitude-only) is reported as "the beyond-baseline signal is amplitude/activity, not period inference"
-- a real, printable negative. Two things the gate text leaves open, pinned here before any number is
read (grilled 2026-08-27):
  - "trained mu" = `exp07_hann0p3_fbwd`, the shipped encoder (D17-CLOSED). `hann0p3_off` is reported
    beside it and never gates, so there is exactly one verdict. Selecting whichever arm scores higher
    would be the estimator-shopping the VOID rule forbids.
  - vs untrained: paired by index, delta_s = R2(fbwd_s) - R2(untrained_i<s>), SE = SD/sqrt(6), which is
    F1/R8's `paired_delta` verbatim. vs amplitude-only: ridge is deterministic so that baseline is a
    CONSTANT, and the SE is the trained series' own spread. Labelled as such rather than dressed up as
    a two-sided SE.
  - the gate pre-registers the `survey_matched` pool, which cannot carry a regression (its positives are
    the 2,021-star subset-test split, i.e. 25 beyond-baseline rotators). Substituted with `r8_added` and
    DISCLOSED; the old population stays recoverable through the cited F1 row.

TWO FOOTING GATES, BOTH ABORT THE RUN (every P0 task here carried one before reading anything new):
  1. estimator -- this script's ridge path, run in v1-compat mode (subset mu, P <= 5 d, linear target,
     `mean`, the subset's own split), must reproduce F1's `rotation_period` R2 = 0.680302 to <= 5e-4.
  2. features -- the re-extracted amplitude columns must equal the cached `features/*.parquet` to 0.0
     across all 106,284 stars. `analyze_r8_fullpool.py` already computed all 25 features per star and
     kept only 4, so this stage recovers the 21 that were discarded rather than computing anything new.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, no GPU, ~35 min cold / ~10 min warm):
    PYTHONUNBUFFERED=1 python experiments/analyze_t2_beyond_baseline_rotation.py
    python experiments/analyze_t2_beyond_baseline_rotation.py --stages features
    python experiments/analyze_t2_beyond_baseline_rotation.py --stages score gate --arms hann0p3_fbwd_s0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg") # write PNG from a headless script, no display backend
import matplotlib.pyplot as plt # noqa: E402

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent)) # reuse R1's mu reader rather than fork it

from analyze_exp08_menu_channel import load_mu_cache # noqa: E402
from swm.eval.features import FEATURE_NAMES, extract_features # noqa: E402
from swm.eval.new_task_extract import replay_first_segment # noqa: E402
from swm.eval.new_task_scorecard import score_regression, score_rotation_period_from_mu # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("t2_rotation")

r8_dir = repo_root / "experiments" / "r8_fullpool"
out_dir = repo_root / "experiments" / "t2_beyond_baseline"
features25_dir = r8_dir / "features25"
subset_cache_dir = repo_root / "experiments" / "exp08_menu_channel" / "subset_mu_cache"

window = 256 # the exp01_window256_seq16 geometry every cached arm was extracted under
shard_size = 5000 # matches analyze_r8_fullpool's shard bounds so the two feature tables align row-for-row

amp_cols = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"] # exp06 basis: scale + roughness, no periodicity
baseline_days = 16 * 256 * 2 / 60 / 24 # seq_len * window_size * 2-min cadence, in days: T1's value verbatim
bucket_edges = [0.0, 2.0, baseline_days, 13.0, np.inf]
bucket_names = ["<=2 d", "2-5.7 d", "5.7-13 d", ">=13 d"]
beyond_buckets = ["5.7-13 d", ">=13 d"] # the two G-rot reads, reported separately (T1's >= 1000 rule)

test_fraction = 0.30 # stratified by bucket, drawn ONCE at split_seed and reused across every arm/control
split_seed = 0

trained_cells = ["exp07_hann0p3_fbwd", "exp07_hann0p3_off"]
gate_cell = "exp07_hann0p3_fbwd" # D17-CLOSED: the shipped encoder is the one G-rot reads
n_seeds = 6
# F1's `rotation_period` cell for hann0p3_fbwd_s0, arm_set `mu`, readout `mean`, linear target, on the
# frozen v1 subset (experiments/f1_fusion_scorecard/f1_probe.csv). The estimator footing gate.
repro_target = {"arm": "hann0p3_fbwd_s0", "r2": 0.680302, "tol": 5e-4}


def arm_names() -> list[str]:
    """The 18 cached encoder arms T2 scores: 6 seeds of each trained cell plus 6 untrained inits."""
    arms = []
    for cell in trained_cells:
        for seed in range(n_seeds):
            arms.append(f"{cell}_s{seed}")
    for init in range(n_seeds):
        arms.append(f"untrained_i{init}")
    return arms


def arm_parts(arm: str) -> tuple[str, int]:
    """Split a cached arm directory name into (family, seed), where an untrained init counts as a seed."""
    if arm.startswith("untrained_i"):
        return "untrained", int(arm.split("_i")[1])
    cell, _, seed = arm.rpartition("_s")
    return cell, int(seed)


# ---------------------------------------------------------------------------------- stage: features
def shard_bounds(n: int) -> list[tuple[int, int]]:
    """Contiguous fixed-size shards over the added-star index; the unit of resume, as in R8."""
    bounds = []
    start = 0
    while start < n:
        bounds.append((start, min(start + shard_size, n)))
        start += shard_size
    return bounds


def run_features(index: pd.DataFrame) -> None:
    """
    Recover the full 25-feature basis for every added star, one parquet per shard, skipping finished work.
    R8 replayed the same flux and called the same extractor but persisted only the last 4 columns, so this
    stage is a recovery of 21 discarded columns rather than a new measurement. That is exactly why footing
    gate 2 can demand equality to 0.0 rather than a tolerance.
    """
    features25_dir.mkdir(parents=True, exist_ok=True)
    bounds = shard_bounds(len(index))
    for shard_idx, (lo, hi) in enumerate(tqdm(bounds, desc="feature shards", total=len(bounds))):
        path = features25_dir / f"shard_{shard_idx}.parquet"
        if path.exists():
            continue
        rows = []
        for entry in index.iloc[lo:hi].itertuples(index=False):
            block = replay_first_segment(Path(entry.path), window) # (n_win, window), absmax guard applied
            if block.shape[0] == 0:
                continue
            vector = extract_features(block.reshape(-1)) # first-segment concatenated flux, skyline protocol
            row = {"tic_id": int(entry.tic_id)}
            for name, value in zip(FEATURE_NAMES, vector):
                row[name] = float(value)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(path, index=False)


def load_features25() -> pd.DataFrame:
    """The 25-feature table for every added star, tic-indexed, after footing gate 2 has passed."""
    parts = []
    for path in sorted(features25_dir.glob("shard_*.parquet"), key=lambda p: int(p.stem.split("_")[1])):
        parts.append(pd.read_parquet(path))
    fresh = pd.concat(parts, ignore_index=True).set_index("tic_id")

    cached_parts = []
    for path in sorted((r8_dir / "features").glob("shard_*.parquet"), key=lambda p: int(p.stem.split("_")[1])):
        cached_parts.append(pd.read_parquet(path))
    cached = pd.concat(cached_parts, ignore_index=True).set_index("tic_id")

    assert set(fresh.index) == set(cached.index), "the 25-feature table covers a different star set than R8's"
    diff = np.abs(fresh.loc[cached.index, amp_cols].to_numpy(dtype=float)
                  - cached[amp_cols].to_numpy(dtype=float)).max()
    assert diff == 0.0, f"footing gate 2 FAILED: amplitude columns differ from R8's cache by {diff}"
    log.info(f"footing gate 2 PASS: {len(fresh)} stars, amplitude columns reproduce R8's cache to {diff}")
    return fresh


# ------------------------------------------------------------------------------- population and split
def rotator_frame() -> pd.DataFrame:
    """
    Every `added` star carrying a TARS rotation period, with its bucket, uncapped in period.
    `added` is the R8 role that excludes the frozen probe's train/val/test stars, so nothing here was
    ever seen by the v1 probe; it also happens to contain no transit/eb/pulsating positive at all.
    """
    pool = pd.read_parquet(r8_dir / "r8_pool.parquet")
    added = pool.loc[pool["role"] == "added", ["tic_id"]]
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["rotation"] = pd.to_numeric(canon["rotation"], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    keep = canon.loc[(canon["rotation"] == 1) & canon["rotation_period"].notna(),
                     ["tic_id", "rotation_period"]]
    frame = added.merge(keep, on="tic_id", how="inner")
    frame["bucket"] = pd.cut(frame["rotation_period"], bins=bucket_edges, labels=bucket_names)
    assert frame["bucket"].notna().all(), "a rotator fell outside the bucket edges"
    return frame.sort_values("tic_id").reset_index(drop=True)


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Draw the 70/30 train/test split ONCE, stratified by period bucket, and reuse it everywhere.
    R8's frozen-probe split does not transfer: `added` stars were never in any split, and a regression
    needs its own train set. Fixing the split across arms, seeds and controls is what makes the 6-seed
    spread pure encoder-seed spread, comparable to every other paired delta in the project.
    """
    rng = np.random.default_rng(split_seed)
    split = pd.Series("train", index=frame.index)
    for bucket in bucket_names:
        members = frame.index[frame["bucket"] == bucket].to_numpy()
        n_test = int(round(test_fraction * len(members)))
        chosen = rng.choice(members, size=n_test, replace=False)
        split.loc[chosen] = "test"
    out = frame.copy()
    out["split"] = split
    return out


# --------------------------------------------------------------------------------- design matrices
def load_arm_mu(arm: str) -> pd.DataFrame:
    """One cached arm's per-star pooled mu, tic-indexed, with the mean and std halves side by side.

    R8 pooled at extraction time, so `mean` and `std` are both already on disk: the `mean_std` readout
    is a column concatenation here, not a second encoder pass. `readout_sweep.pool_stars` has no
    `mean_std` mode and is deliberately not extended (handoff section 4a).
    """
    tics, means, stds = [], [], []
    parts = sorted((r8_dir / "mu_cache" / arm).glob("shard_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    assert parts, f"no cached mu shards for arm {arm}"
    for path in parts:
        with np.load(path) as data:
            tics.append(data["tics"])
            means.append(data["mean"])
            stds.append(data["std"])
    values = np.concatenate([np.concatenate(means, axis=0), np.concatenate(stds, axis=0)], axis=1)
    columns = []
    for half in ["mean", "std"]:
        for j in range(values.shape[1] // 2):
            columns.append(f"{half}{j}")
    return pd.DataFrame(values, index=pd.Index(np.concatenate(tics), name="tic_id"), columns=columns)


def readout_columns(mu: pd.DataFrame, readout: str) -> list[str]:
    """The mu columns one readout uses: `mean` is the published pooling, `mean_std` adds the std half."""
    if readout == "mean_std":
        return list(mu.columns)
    keep = []
    for name in mu.columns:
        if name.startswith("mean"):
            keep.append(name)
    return keep


# ------------------------------------------------------------------------------------------ scoring
def bucket_rows(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                global_metrics: dict, scope_label: str) -> list[dict]:
    """
    Slice one fitted arm's held-out predictions into per-bucket R2 / RMSE / Spearman, plus a pooled row.
    Each bucket is scored against its OWN target variance, so `target_sd` rides on every row: a narrow
    bucket can post a negative R2 while predicting perfectly well in absolute terms, and the reader must
    be able to see the denominator that produced it.
    """
    rows = []
    reported = list(bucket_names)
    if scope_label == "beyond":
        reported = list(beyond_buckets)
    for bucket in reported + ["all"]:
        if bucket == "all":
            mask = np.ones(len(y_true), dtype=bool)
        else:
            mask = (frame["bucket"] == bucket).to_numpy()
        n = int(mask.sum())
        if n < 3:
            continue
        truth = y_true[mask]
        pred = y_pred[mask]
        sd = float(truth.std(ddof=1))
        residual = truth - pred
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if sd > 0:
            r2 = float(1.0 - np.sum(residual ** 2) / np.sum((truth - truth.mean()) ** 2))
        else:
            r2 = np.nan
        rho = float(pd.Series(truth).corr(pd.Series(pred), method="spearman"))
        # bias and scatter are split out because R2 conflates them, and in the narrow beyond-baseline
        # buckets the arm ordering on R2 is driven by BIAS while the ordering on rho is driven by
        # scatter. A gate read on R2 alone cannot tell those apart; these two columns are how the
        # write-up demonstrates which one produced a given margin.
        bias = float(pred.mean() - truth.mean())
        rows.append({"bucket": bucket, "n_test": n, "target_sd": sd, "r2": r2, "rmse": rmse,
                     "spearman": rho, "bias": bias, "pred_mean": float(pred.mean()),
                     "truth_mean": float(truth.mean()), "pred_sd": float(pred.std(ddof=1))})
    for row in rows:
        if row["bucket"] == "all":
            # the pooled row must be the SHIPPED estimator's own number, not a re-derivation of it
            row["r2"] = global_metrics["r2"]
            row["rmse"] = global_metrics["rmse"]
            row["spearman"] = global_metrics["spearman"]
    return rows


def score_cell(x: pd.DataFrame, frame: pd.DataFrame, transform: str, scope: str,
               seed: int) -> list[dict]:
    """
    Fit one design matrix under one target transform and one fit scope, and return its per-bucket rows.
    `global` fits on every rotator and slices the held-out predictions by bucket, which is what
    architecture.md's pre-registered "monotonic degradation across buckets" reading means. `beyond`
    refits on the beyond-baseline rotators alone and is the labelled secondary: it asks whether the band
    carries signal at all, not whether the global map degrades across it.
    """
    rows = frame
    if scope == "beyond":
        rows = frame[frame["bucket"].isin(beyond_buckets)]
    target = rows["rotation_period"].to_numpy(dtype=float)
    if transform == "log10":
        target = np.log10(target)
    is_train = (rows["split"] == "train").to_numpy()
    features = x.loc[rows["tic_id"]].to_numpy(dtype=float)
    metrics, pred = score_regression(features[is_train], target[is_train], features[~is_train],
                                     target[~is_train], "ridge", seed)
    return bucket_rows(rows[~is_train], target[~is_train], pred, metrics, scope)


def footing_gate_estimator() -> None:
    """
    Reproduce F1's published `rotation_period` R2 through this script's own ridge path before anything
    new is read. The cell is the frozen v1 subset under ADR-0004's P <= 5 d cap and a linear target, so
    it shares nothing with T2's population or split; what it pins is that `score_regression` reaches the
    same estimator here that it reached in F1. A divergence means the readout drifted, and every T2
    number downstream would be measuring a different probe than the one the gates were decided on.
    """
    cache = subset_cache_dir / f"{repro_target['arm']}.npz"
    assert cache.exists(), f"the v1 subset mu cache is missing: {cache}"
    cell = score_rotation_period_from_mu(load_mu_cache(cache), regressor="ridge", random_state=0)[0]
    delta = abs(cell["r2"] - repro_target["r2"])
    assert delta <= repro_target["tol"], (
        f"footing gate 1 FAILED: rotation_period R2 {cell['r2']} vs F1's {repro_target['r2']} "
        f"(delta {delta} > {repro_target['tol']})")
    log.info(f"footing gate 1 PASS: rotation_period R2 {cell['r2']} vs F1's {repro_target['r2']}, "
             f"delta {delta}")


def run_score(frame: pd.DataFrame, feats: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """
    Score every arm x arm-set x readout x transform x fit-scope cell and stack the per-bucket rows.
    The two engineered arm sets carry no seed (ridge is deterministic and they do not depend on an
    encoder), so they are fitted once and emitted with seed -1 rather than copied per arm.
    """
    tics = frame["tic_id"]
    rows = []

    for arm_set, columns in [("amplitude_only", amp_cols), ("features_only", list(FEATURE_NAMES))]:
        design = feats.loc[tics, columns]
        for transform in ["log10", "linear"]:
            for scope in ["global", "beyond"]:
                for row in score_cell(design, frame, transform, scope, 0):
                    rows.append({"arm": arm_set, "family": arm_set, "seed": -1, "arm_set": arm_set,
                                 "readout": "-", "transform": transform, "fit_scope": scope, **row})

    jobs = []
    for arm in arms:
        for readout in ["mean", "mean_std"]:
            jobs.append((arm, readout))
    for arm, readout in tqdm(jobs, desc="arm x readout", total=len(jobs)):
        family, seed = arm_parts(arm)
        mu = load_arm_mu(arm)
        design_mu = mu.loc[tics, readout_columns(mu, readout)]
        design_fusion = pd.concat([feats.loc[tics, list(FEATURE_NAMES)], design_mu], axis=1)
        for arm_set, design in [("mu", design_mu), ("features_plus_mu", design_fusion)]:
            for transform in ["log10", "linear"]:
                for scope in ["global", "beyond"]:
                    for row in score_cell(design, frame, transform, scope, seed):
                        rows.append({"arm": arm, "family": family, "seed": seed, "arm_set": arm_set,
                                     "readout": readout, "transform": transform, "fit_scope": scope,
                                     **row})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- G-rot gate
def paired_delta(deltas: list[float]) -> dict:
    """Mean, SD and 2*SE of a per-seed delta series; F1/R8's `paired_delta`, unchanged."""
    values = np.array(deltas, dtype=float)
    sd = float(values.std(ddof=1))
    two_se = 2 * sd / np.sqrt(len(values))
    return {"n_seeds": len(values), "delta_mean": float(values.mean()), "delta_sd": sd,
            "delta_2se": two_se, "beats_2se": bool(values.mean() > two_se)}


def run_gate(probe: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate G-rot exactly as pre-registered, and return one row per (bucket, baseline) contrast.
    Fixed by the gate text: `mean` readout, mu arm set, the two beyond-baseline buckets. Fixed by the
    2026-08-27 grill: the log10 target, the global fit, and `hann0p3_fbwd` as the single gating arm.
    The untrained contrast is paired by seed index; the amplitude contrast is against a deterministic
    constant, so its spread is the trained arm's alone and the column says so.
    """
    cells = probe[(probe["arm_set"] == "mu") & (probe["readout"] == "mean")
                  & (probe["transform"] == "log10") & (probe["fit_scope"] == "global")]
    amp = probe[(probe["arm_set"] == "amplitude_only") & (probe["transform"] == "log10")
                & (probe["fit_scope"] == "global")]
    rows = []
    for bucket in beyond_buckets:
        for metric in ["r2", "spearman"]:
            trained = cells[(cells["family"] == gate_cell) & (cells["bucket"] == bucket)]
            untrained = cells[(cells["family"] == "untrained") & (cells["bucket"] == bucket)]
            trained = trained.set_index("seed")[metric].sort_index()
            untrained = untrained.set_index("seed")[metric].sort_index()
            assert list(trained.index) == list(untrained.index), "trained and untrained seeds do not pair"
            baseline_value = float(amp.loc[amp["bucket"] == bucket, metric].iloc[0])

            vs_untrained = paired_delta((trained - untrained).tolist())
            vs_untrained.update({"bucket": bucket, "metric": metric, "baseline": "untrained",
                                 "se_source": "paired", "trained_mean": float(trained.mean()),
                                 "baseline_mean": float(untrained.mean()),
                                 "is_pregistered_gate": metric == "r2"})
            rows.append(vs_untrained)

            vs_amp = paired_delta((trained - baseline_value).tolist())
            vs_amp.update({"bucket": bucket, "metric": metric, "baseline": "amplitude_only",
                           "se_source": "trained arm only", "trained_mean": float(trained.mean()),
                           "baseline_mean": baseline_value, "is_pregistered_gate": metric == "r2"})
            rows.append(vs_amp)
    return pd.DataFrame(rows)


def gate_verdict(gate: pd.DataFrame, metric: str = "r2") -> str:
    """
    Turn the two contrasts per bucket into G-rot's pre-registered verdict language.
    `metric` defaults to R2, which is what G-rot pre-registers and therefore what the VERDICT is. The
    same function is called a second time on Spearman rho purely to report, beside the verdict, whether
    the metric that answers the gate's stated QUESTION agrees with the metric the gate names. Rerunning
    the verdict on rho and quoting whichever reads better would be estimator-shopping; both are printed.
    """
    verdicts = []
    for bucket in beyond_buckets:
        rows = gate[(gate["bucket"] == bucket) & (gate["metric"] == metric)].set_index("baseline")
        beats_untrained = bool(rows.loc["untrained", "beats_2se"])
        beats_amp = bool(rows.loc["amplitude_only", "beats_2se"])
        if beats_untrained and beats_amp:
            verdicts.append(f"{bucket}: PASS")
        elif beats_untrained:
            verdicts.append(f"{bucket}: PARTIAL PASS -- the beyond-baseline signal is "
                            f"amplitude/activity, not period inference")
        else:
            verdicts.append(f"{bucket}: FAIL -- does not clear the untrained floor")
    return " | ".join(verdicts)


def saturation_table(probe: pd.DataFrame) -> pd.DataFrame:
    """
    Where each arm's predicted period stops rising, in days, per bucket. This is T2's headline finding
    and it is a level statistic, so it belongs in its own table rather than inside an R2 column.
    Read it against two reference numbers printed beside it: the model's 5.69 d input baseline, and the
    training set's unconditional geometric-mean period of 2.40 d. A ceiling AT the unconditional mean
    would mean the readout learned nothing and simply reverted; a ceiling well above it that stops near
    the input baseline is the pre-registered "encoder behavior matches the physical limit of the input"
    reading, made quantitative.
    """
    cell = probe[(probe["arm_set"] == "mu") & (probe["readout"] == "mean")
                 & (probe["transform"] == "log10") & (probe["fit_scope"] == "global")
                 & (probe["bucket"] != "all")]
    engineered = probe[(probe["arm_set"].isin(["amplitude_only", "features_only"]))
                       & (probe["transform"] == "log10") & (probe["fit_scope"] == "global")
                       & (probe["bucket"] != "all")]
    rows = pd.concat([cell, engineered], ignore_index=True)
    rows = rows.assign(pred_days=10 ** rows["pred_mean"], truth_days=10 ** rows["truth_mean"])
    summary = rows.groupby(["family", "bucket"], observed=True).agg(
        n_arms=("pred_days", "size"), pred_days_mean=("pred_days", "mean"),
        pred_days_sd=("pred_days", "std"), truth_days=("truth_days", "first"),
        spearman_mean=("spearman", "mean")).reset_index()
    summary["baseline_days"] = baseline_days
    return summary


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Per-arm absolute score, mean and 2*SE over seeds, for every arm set / readout / transform / bucket."""
    keys = ["family", "arm_set", "readout", "transform", "fit_scope", "bucket"]
    grouped = probe.groupby(keys, observed=True)
    summary = grouped.agg(n_seeds=("r2", "size"), n_test=("n_test", "first"),
                          target_sd=("target_sd", "first"), r2_mean=("r2", "mean"),
                          r2_sd=("r2", "std"), rmse_mean=("rmse", "mean"),
                          spearman_mean=("spearman", "mean")).reset_index()
    summary["r2_2se"] = 2 * summary["r2_sd"] / np.sqrt(summary["n_seeds"])
    return summary


def plot_degradation(summary: pd.DataFrame, path: Path) -> None:
    """
    The headline picture, on two panels because one axis cannot carry both statistics honestly.
    Left is within-bucket Spearman rho, which is denominator-free and therefore the readable degradation
    curve. Right is bucket-local R2 on a symlog axis: it spans roughly -60 to +0.7 purely because the
    narrow buckets have 8x less variance to explain, and plotting it linearly would compress every arm
    into one line at the top. Vertical distance between arms is the quantity G-rot reads; the absolute
    level on the right panel is not a quantity at all.
    """
    cell = summary[(summary["readout"].isin(["mean", "-"])) & (summary["transform"] == "log10")
                   & (summary["fit_scope"] == "global") & (summary["bucket"] != "all")]
    series = [("exp07_hann0p3_fbwd", "mu"), ("exp07_hann0p3_off", "mu"), ("untrained", "mu"),
              ("exp07_hann0p3_fbwd", "features_plus_mu"), ("features_only", "features_only"),
              ("amplitude_only", "amplitude_only")]
    positions = np.arange(len(bucket_names))
    panels = [("spearman_mean", None, "within-bucket Spearman rho", "linear"),
              ("r2_mean", "r2_2se", "R2 on log10(P_rot), bucket-local", "symlog")]
    plt.figure(figsize=(13, 5))
    for panel_idx, (value_col, error_col, ylabel, yscale) in enumerate(panels):
        plt.subplot(1, 2, panel_idx + 1)
        for family, arm_set in series:
            rows = cell[(cell["family"] == family) & (cell["arm_set"] == arm_set)]
            rows = rows.set_index("bucket").reindex(bucket_names)
            if error_col is None:
                errors = None
            else:
                errors = rows[error_col]
            plt.errorbar(positions, rows[value_col], yerr=errors, marker="o", capsize=3,
                         label=f"{family} / {arm_set}")
        plt.axvline(1.5, linestyle="--", color="gray") # the 5.69 d model input baseline sits on this edge
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.yscale(yscale) # symlog: linear near zero, logarithmic outside, so -59 and +0.7 share an axis
        plt.xticks(positions, bucket_names)
        plt.xlabel("TARS rotation period bucket")
        plt.ylabel(ylabel)
        if panel_idx == 0:
            plt.title("ordering: mu collapses past the baseline, amplitude rises")
            plt.legend(fontsize=8)
        else:
            plt.title("level: R2 orders the arms the opposite way (symlog axis)")
    plt.suptitle("T2: rotation-period recovery per bucket, global fit, log10 target, `mean` readout")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="T2: beyond-baseline rotation-period probe on cached mu.")
    ap.add_argument("--stages", nargs="+", default=["features", "score", "gate"],
                    choices=["features", "score", "gate"])
    ap.add_argument("--arms", nargs="+", default=None, help="Default: all 18 cached arms.")
    args = ap.parse_args()

    out_dir.mkdir(parents=True, exist_ok=True)
    arms = args.arms or arm_names()

    if "features" in args.stages:
        index = pd.read_parquet(r8_dir / "added_npz_index.parquet")
        log.info(f"recovering the 25-feature basis for {len(index)} added stars")
        run_features(index)

    if "score" in args.stages:
        footing_gate_estimator()
        feats = load_features25()
        frame = assign_split(rotator_frame())
        counts = frame.groupby(["bucket", "split"], observed=True).size().unstack(fill_value=0)
        log.info(f"population: {len(frame)} rotators, "
                 f"{frame['rotation_period'].min()}-{frame['rotation_period'].max()} d\n{counts}")
        frame.to_csv(out_dir / "t2_population.csv", index=False)
        probe = run_score(frame, feats, arms)
        probe.to_csv(out_dir / "t2_bucket.csv", index=False)
        log.info(f"wrote {out_dir / 't2_bucket.csv'} ({len(probe)} rows)")

    if "gate" in args.stages:
        probe = pd.read_csv(out_dir / "t2_bucket.csv")
        summary = summarize(probe)
        summary.to_csv(out_dir / "t2_summary.csv", index=False)
        gate = run_gate(probe)
        gate.to_csv(out_dir / "t2_gate.csv", index=False)
        saturation = saturation_table(probe)
        saturation.to_csv(out_dir / "t2_saturation.csv", index=False)
        plot_degradation(summary, out_dir / "t2_degradation.png")
        log.info(f"wrote {out_dir / 't2_summary.csv'} ({len(summary)} rows), t2_gate.csv, "
                 f"t2_saturation.csv and t2_degradation.png")
        log.info(f"G-rot VERDICT as pre-registered on R2 ({gate_cell}, mu, mean, log10, global fit, "
                 f"population r8_added substituted for survey_matched): {gate_verdict(gate, 'r2')}")
        log.info(f"same contrast read on Spearman rho, REPORTED NOT GATED: "
                 f"{gate_verdict(gate, 'spearman')}")
        ceiling = saturation[(saturation["family"] == gate_cell)
                             & (saturation["bucket"].isin(beyond_buckets))]
        for row in ceiling.itertuples(index=False):
            log.info(f"saturation {row.bucket}: predicted {row.pred_days_mean} d "
                     f"(seed sd {row.pred_days_sd}) against a true {row.truth_days} d, "
                     f"input baseline {baseline_days} d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
