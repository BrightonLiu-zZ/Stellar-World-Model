"""Stage 1 prep: build the enriched local-POC subset and the frozen star-disjoint split.

Subset = all transit/eb/pulsating positives that have at least one packed window, plus a
random sample of quiet stars (matched in no variability catalog) as negatives. The split is
70/15/15 train/val/test, stratified so each rare positive class is proportionally present in
all three folds, and keyed by TIC so no star ever crosses a fold boundary.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.data.subset
    python -m swm.data.subset data.n_quiet=2000 seed=0        # smaller negative pool
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from swm.data.sequence_files import scan_sequence_files

log = logging.getLogger(__name__)

positive_cols = ["transit", "eb", "pulsating"]
all_label_cols = positive_cols + ["rotation", "flare_ever"]


def load_labels(labels_csv: str) -> pd.DataFrame:
    """
    Load the Stage 0d variability label table and coerce the binary label columns to int.
    Fails loud if the file or any required column is missing, since the whole subset depends on it.
    """
    path = Path(labels_csv)
    assert path.exists(), f"labels csv not found: {path}"
    df = pd.read_csv(path)
    assert "tic_id" in df.columns, "labels csv missing tic_id"
    for col in all_label_cols:
        assert col in df.columns, f"labels csv missing column: {col}"
        df[col] = df[col].fillna(0).astype(int)
    return df


def assign_stratum(row: pd.Series) -> str:
    """
    Map a star to one stratification bucket, rarest positive class first.
    Rarest-first (transit < eb < pulsating) lets the scarce classes drive the stratified split;
    multi-positive overlap is tiny (transit and eb co-occur on 59 stars) so single-bucket is fine.
    """
    if row["transit"] == 1:
        return "transit"
    if row["eb"] == 1:
        return "eb"
    if row["pulsating"] == 1:
        return "pulsating"
    return "quiet"


def split_within_stratum(
    tics: np.ndarray, rng: np.random.Generator, val_frac: float, test_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle one stratum's TICs and slice them into train/val/test.
    Done per stratum so every fold gets the same class proportions as the whole subset.
    """
    shuffled = tics.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rng = np.random.default_rng(cfg.seed)

    files = scan_sequence_files(cfg.paths.sequences_dir)
    present_tics = set(files["tic_id"].unique().tolist()) # TICs with >=1 packed segment
    log.info(f"corpus: {len(files)} segment files across {len(present_tics)} TICs")

    labels = load_labels(cfg.paths.labels_csv)
    labels["has_windows"] = labels["tic_id"].isin(present_tics)

    is_positive = labels[positive_cols].sum(axis=1) > 0
    is_quiet = labels[all_label_cols].sum(axis=1) == 0 # matched in no catalog

    positives = labels[is_positive & labels["has_windows"]].copy()
    quiet_pool = labels[is_quiet & labels["has_windows"]].copy()

    n_quiet = min(int(cfg.data.n_quiet), len(quiet_pool))
    quiet_idx = rng.choice(quiet_pool.index.to_numpy(), size=n_quiet, replace=False)
    quiet_sample = quiet_pool.loc[quiet_idx].copy()

    subset = pd.concat([positives, quiet_sample], ignore_index=True)
    subset["stratum"] = subset.apply(assign_stratum, axis=1)

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    for stratum_name in ["transit", "eb", "pulsating", "quiet"]:
        stratum_tics = subset.loc[subset["stratum"] == stratum_name, "tic_id"].to_numpy()
        tr, va, te = split_within_stratum(stratum_tics, rng, cfg.data.val_frac, cfg.data.test_frac)
        train_ids.extend(tr.tolist())
        val_ids.extend(va.tolist())
        test_ids.extend(te.tolist())

    split_of: dict[int, str] = {}
    for tic in train_ids:
        split_of[tic] = "train"
    for tic in val_ids:
        split_of[tic] = "val"
    for tic in test_ids:
        split_of[tic] = "test"
    subset["split"] = subset["tic_id"].map(split_of)

    out_dir = Path(cfg.paths.subset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = ["tic_id", "split", "stratum"] + positive_cols + ["rotation", "flare_ever"]
    subset[keep_cols].to_parquet(out_dir / "subset_tics.parquet", index=False)
    subset[["tic_id", "split"]].to_parquet(out_dir / "split.parquet", index=False)

    # Visibility: what the cuts kept vs dropped, and the per-fold class balance.
    availability = pd.DataFrame(
        {
            "labeled_total": [int((labels[c] == 1).sum()) for c in positive_cols],
            "with_windows": [int(((labels[c] == 1) & labels["has_windows"]).sum()) for c in positive_cols],
        },
        index=positive_cols,
    )
    log.info(f"positive availability (dropped = labeled_total - with_windows):\n{availability}")
    log.info(f"quiet pool with windows: {len(quiet_pool)}, sampled into subset: {n_quiet}")

    balance = subset.pivot_table(index="stratum", columns="split", values="tic_id", aggfunc="count", fill_value=0)
    balance = balance.reindex(columns=["train", "val", "test"], fill_value=0)
    log.info(f"per-fold star counts by stratum:\n{balance}")
    log.info(f"subset total stars: {len(subset)} --> {out_dir/'subset_tics.parquet'}")


if __name__ == "__main__":
    main()
