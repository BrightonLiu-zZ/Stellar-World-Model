"""Pool 3 (rotation eval pool) invariants: the properties the design's argument rests on."""
from pathlib import Path

import pandas as pd
import pytest

from swm.eval.rotation_pool import PERIOD_CAP_DAYS, assign_stratum, load_canonical

repo_root = Path(__file__).resolve().parents[3]
pool_path = repo_root / "processed" / "subset" / "rotation_pool.parquet"
needs_pool = pytest.mark.skipif(not pool_path.exists(), reason="pool 3 not built on this machine")


def test_stratum_rarest_first() -> None:
    assert assign_stratum(1, PERIOD_CAP_DAYS) == "rot_le5"
    assert assign_stratum(1, PERIOD_CAP_DAYS + 0.01) == "rot_gt5"
    assert assign_stratum(0, float("nan")) == "neg"


@needs_pool
def test_disjoint_from_pool1_and_split_once() -> None:
    pool = pd.read_parquet(pool_path)
    pool1 = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")
    assert pool["tic_id"].is_unique
    assert not set(pool["tic_id"]) & set(pool1["tic_id"])
    assert set(pool["split"]) == {"train", "val", "test"}


@needs_pool
def test_no_second_v1_label_and_strata_match_labels() -> None:
    pool = pd.read_parquet(pool_path)
    labels = load_canonical(repo_root / "labels" / "variability_labels_star.csv")
    merged = pool.merge(labels, on="tic_id", how="left")
    assert merged[["transit", "eb", "pulsating"]].to_numpy().sum() == 0
    assert ((merged["stratum"] != "neg") == (merged["rotation"] == 1)).all()
    le5 = merged["stratum"] == "rot_le5"
    assert (merged.loc[le5, "rotation_period"] <= PERIOD_CAP_DAYS).all()
    # 70/15/15 inside every stratum, to rounding
    for _, group in merged.groupby("stratum"):
        share = group["split"].value_counts(normalize=True)
        assert abs(share["train"] - 0.70) < 0.001 and abs(share["test"] - 0.15) < 0.001


@needs_pool
def test_sensitivity_flags_are_subsets_with_both_classes() -> None:
    pool = pd.read_parquet(pool_path)
    assert (pool["strict"] <= pool["no_flare"]).all()  # strict also drops flare stars
    for flag in ["no_flare", "strict"]:
        kept = pool[pool[flag]]
        assert set(kept["stratum"]) == {"rot_le5", "rot_gt5", "neg"}
