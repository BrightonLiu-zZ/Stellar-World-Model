"""Ceiling A1/A2 for the new-task probe menu (plan 2026-07-25, decision 9).

Every downstream number so far says "the trained encoder beats an untrained random encoder". That is a
weak bar. A1/A2 add the bar a reviewer actually asks about -- can the learned representation beat 25
hand-engineered light-curve features (periodogram peaks, log-spaced PSD bands, spectral entropy,
moments, robust scatter; `swm.eval.features`)?

  A1  the SAME linear readout the headline uses, on engineered features   -> "beats classic features"
  A2  gradient-boosted trees on the same features                         -> separates "the features
      lack the information" from "a linear probe cannot exploit it"

The B1 supervised ceiling is deliberately absent (it needs GPU training runs); so "beats engineered
features" is claimable from this module and "closes the supervised gap" is not.

Populations are NOT re-derived here. Each star's feature vector is packed into a one-row block and fed
through `new_task_scorecard`'s own scorers, so every keep-mask (numax floor, prot 5.7 d cap, rgb State
filter, Villanova exclusion) is byte-identical to the frozen-mu scorecard by construction.

Run (swm env, from repo root, PYTHONPATH=src; CPU-only, no checkpoints needed):
    PYTHONUNBUFFERED=1 python -m swm.eval.new_task_ceiling --out experiments/new_task_exp05/ceiling_A1A2.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from swm.eval.features import FEATURE_NAMES, extract_features
from swm.eval.new_task_extract import index_pool_npz, replay_first_segment
from swm.eval.new_task_scorecard import (DETECTION, REGRESSION, label_frame, score_contrastive,
                                         score_detection, score_ijspeert_from_mu, score_regression_task,
                                         score_rotation_period_from_mu)
from swm.eval.readout_sweep import load_first_segment_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("new_task_ceiling")

repo_root = Path(__file__).resolve().parents[3]
WINDOW = 256  # exp01 geometry, shared by every arm the scorecard compares against

# (arm label written to the CSV, detection/contrastive readout, regressor)
CEILINGS = [("A1_feats", "logistic", "ridge"), ("A2_feats", "gbm", "gbm")]


def _features_as_mu(tics: list[int], blocks: list[np.ndarray], desc: str) -> tuple[list[int], list[np.ndarray]]:
    """Reduce each star's first-segment flux to one engineered feature vector, shaped as a 1-window mu block.

    Concatenating the star's windows into a contiguous series before reducing matches skyline's Ceiling-A
    protocol exactly (`extract_features(block.reshape(-1))`). Returning (1, n_features) blocks means
    `pool_stars(..., "mean")` inside the scorecard is the identity, so the scorers need no special-casing.
    """
    out = []
    for block in tqdm(blocks, desc=desc, total=len(blocks)):
        out.append(extract_features(block.reshape(-1)).astype(np.float32).reshape(1, -1))
    return tics, out


def _guard(task: str, shape: str, arm: str, score) -> list[dict]:
    """Run one task's scorer, turning an empty train/test population into a visible NaN row, not a crash.

    A `--limit` smoke can legitimately leave a narrow task (rgb_vs_heb, prot_kounkel) with no labelled
    stars. Emitting the row with n_test=0 and a `skipped` reason keeps that fact in the CSV rather than
    silently dropping the task -- an empty population in a FULL run is a real anomaly and must stay visible.
    """
    try:
        return [{"arm": arm, "task": task, "shape": shape, **cell} for cell in score()]
    except ValueError as err:
        log.warning(f"{arm}/{task}: empty population, emitting NaN row ({err})")
        return [{"arm": arm, "task": task, "shape": shape, "pooling": "mean", "n_test": 0,
                 "skipped": "empty population"}]


def pool_feature_mu(pool_path: Path, seq_dir: Path, limit: int | None) -> dict:
    """Engineered-feature 'mu' over the new-task pool, replayed from npz exactly as new_task_extract does."""
    pool = pd.read_parquet(pool_path)
    if limit:
        pool = pd.concat([pool[pool["split"] == s].sort_values("tic_id").head(limit)
                          for s in ["train", "val", "test"]], ignore_index=True)
    npz_index = index_pool_npz(seq_dir, set(pool["tic_id"].astype(int)))
    log.info(f"indexed first-segment npz for {len(npz_index)} pool TICs")
    result = {}
    for split in ["train", "val", "test"]:
        want = pool.loc[pool["split"] == split, "tic_id"].astype(int).sort_values().tolist()
        tics, blocks = [], []
        for tic in tqdm(want, desc=f"read npz[{split}]", total=len(want)):
            entry = npz_index.get(tic)
            assert entry is not None, f"pool TIC {tic} has no npz under {seq_dir}"
            block = replay_first_segment(entry[2], WINDOW)
            if block.shape[0] == 0:
                continue
            tics.append(tic)
            blocks.append(block)
        result[split] = _features_as_mu(tics, blocks, f"feats[{split}]")
    return result


def subset_feature_mu(packed_dir: Path) -> dict:
    """Engineered-feature 'mu' over the frozen v1 packed subset (rotation_period + ijspeert ride this)."""
    result = {}
    for split in ["train", "test"]:
        tics, blocks = load_first_segment_blocks(packed_dir, split, WINDOW)
        result[split] = _features_as_mu(tics, blocks, f"feats_subset[{split}]")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="A1/A2 engineered-feature ceilings for the new-task menu.")
    ap.add_argument("--limit", type=int, default=None, help="First N pool TICs per split (smoke).")
    ap.add_argument("--pool", default=None, help="Default: processed/subset/new_task_pool.parquet")
    ap.add_argument("--with-subset", action="store_true",
                    help="Also compute the v1-subset ceilings (rotation_period, ijspeert).")
    ap.add_argument("--out", default=None, help="Default: experiments/new_task_exp05/ceiling_A1A2.csv")
    args = ap.parse_args()

    pool_path = Path(args.pool) if args.pool else repo_root / "processed" / "subset" / "new_task_pool.parquet"
    seq_dir = repo_root / "processed" / "sequences"
    labels = label_frame()

    log.info("building pool feature table (one FFT + moments per star)")
    feat_mu = pool_feature_mu(pool_path, seq_dir, args.limit)

    subset_mu_feats = None
    if args.with_subset:
        log.info("building v1-subset feature table")
        subset_mu_feats = subset_feature_mu(repo_root / "experiments" / "exp01_window256_seq16" / "packed")

    rows = []
    for arm, readout, regressor in CEILINGS:
        for name, column in DETECTION:
            # mean pooling only: with one feature row per star, window_score(MIL) is the same fit.
            rows += _guard(name, "detection", arm,
                           lambda: score_detection(feat_mu, labels, column, readout, poolings=("mean",)))
        rows += _guard("rgb_vs_heb", "contrastive", arm, lambda: score_contrastive(feat_mu, labels, readout))
        for name, column, log_target in REGRESSION:
            rows += _guard(name, "regression", arm,
                           lambda: score_regression_task(feat_mu, labels, column, log_target, regressor))
        if subset_mu_feats is not None:
            rows.extend(_subset_rows(arm, readout, regressor, subset_mu_feats))
        log.info(f"scored ceiling {arm}")

    result = pd.DataFrame(rows)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    result["n_features"] = len(FEATURE_NAMES)
    out_path = Path(args.out) if args.out else repo_root / "experiments" / "new_task_exp05" / "ceiling_A1A2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(rows)} new rows)")
    return 0


def _subset_rows(arm: str, readout: str, regressor: str, feats: dict) -> list[dict]:
    """v1-subset ceilings, scored through the same helpers the frozen-mu scorecard uses."""
    rows = []
    for cell in score_rotation_period_from_mu(feats, regressor):
        rows.append({"arm": arm, "task": "rotation_period", "shape": "regression", **cell})
    for cell in score_ijspeert_from_mu(feats, readout):
        rows.append({"arm": arm, **cell})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
