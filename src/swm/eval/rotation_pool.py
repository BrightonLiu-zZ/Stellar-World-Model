"""Build pool 3, the rotation eval pool (design: docs/plans/2026-09-20-rotation-pool-design.md).

Pool 1 admits positives only through transit/eb/pulsating and defines quiet as "matched in no
catalogue", so every one of its 1,138 rotation positives carries a second label and none of its quiet
stars rotates. Pool 3 is the complement: every corpus star with windows that is NOT in pool 1. That
universe holds no transit/eb/pulsating positive at all (pool 1 took them), no star seen in
pre-training, and its prevalence is the survey's (0.130 against a corpus 0.125), so nothing is sampled.

Splits are TIC-disjoint 70/15/15, seed 0, stratified rarest first on
    rot_le5   rotation = 1 and P <= 5 d   (the rotation_period population, ADR-0004)
    rot_gt5   rotation = 1 and P  > 5 d
    neg       rotation = 0
Two sensitivity sub-populations ride along as boolean columns; the scorer refits on them:
    no_flare  flare_ever = 0, both classes (rotators flare 20x more often than non-rotators)
    strict    positives whose only catalogue memberships are rotation ones, negatives with none at all

Writes `processed/subset/rotation_pool.parquet` (tic_id, split, stratum, no_flare, strict); label
values are joined at score time, as pools 1 and 2 do.

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.rotation_pool
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from swm.eval.new_task_pool import corpus_tics, split_stratum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("rotation_pool")

repo_root = Path(__file__).resolve().parents[3]
STRATUM_PRIORITY = ["rot_le5", "rot_gt5", "neg"]
PERIOD_CAP_DAYS = 5.0  # ADR-0004; one model input sequence spans 5.69 d
V1_COLS = ["transit", "eb", "pulsating", "rotation", "flare_ever"]
# pool-2 catalogues that are NOT rotation catalogues; prot_kounkel is one, so a Kounkel rotator stays
# a legitimate strict positive but is barred from the strict negatives (TARS-null, Kounkel-positive)
NON_ROTATION_EXT_COLS = ["osc_giant", "solar_like_osc", "rgb_vs_heb"]


def load_canonical(path: Path) -> pd.DataFrame:
    """The Stage 0d label table with the five binary columns as int and the TARS period as float."""
    assert path.exists(), f"labels csv not found: {path}"
    canon = pd.read_csv(path)
    canon["tic_id"] = canon["tic_id"].astype(int)
    for col in V1_COLS:
        canon[col] = pd.to_numeric(canon[col], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    return canon[["tic_id", *V1_COLS, "rotation_period"]]


def other_catalogue_flags(tics: pd.Series) -> pd.DataFrame:
    """Per star: is it in a non-rotation pool-2 catalogue, and is it in ANY pool-2 catalogue."""
    ext = pd.read_csv(repo_root / "labels" / "new_task_labels_star.csv")
    ext["tic_id"] = ext["tic_id"].astype(int)
    non_rot = (ext["osc_giant"].fillna(0) == 1) | (ext["solar_like_osc"].fillna(0) == 1) | ext["rgb_vs_heb"].notna()
    ijspeert = set(pd.read_csv(repo_root / "labels" / "external" / "ijspeert2024_bright.csv")["TIC"].astype(int))
    in_non_rot = set(ext.loc[non_rot, "tic_id"]) | ijspeert
    in_any = set(ext["tic_id"]) | ijspeert
    return pd.DataFrame({"in_non_rotation_ext": tics.isin(in_non_rot).to_numpy(),
                         "in_any_ext": tics.isin(in_any).to_numpy()}, index=tics.index)


def assign_stratum(rotation: int, period: float) -> str:
    """Single split stratum for one star, rarest first."""
    if rotation == 1 and period <= PERIOD_CAP_DAYS:
        return "rot_le5"
    if rotation == 1:
        return "rot_gt5"
    return "neg"


def build_pool(seed: int) -> pd.DataFrame:
    """Pool 3 membership, strata, splits and the two sensitivity flags."""
    corpus = corpus_tics(repo_root / "processed" / "sequences")
    pool1 = set(pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")["tic_id"].astype(int))
    canon = load_canonical(repo_root / "labels" / "variability_labels_star.csv")
    pool = canon[canon["tic_id"].isin(corpus - pool1)].sort_values("tic_id").reset_index(drop=True)
    log.info(f"corpus {len(corpus)} TICs, pool 1 {len(pool1)}, pool 3 universe {len(pool)}")

    leaked = int(pool[["transit", "eb", "pulsating"]].to_numpy().sum())
    assert leaked == 0, f"{leaked} transit/eb/pulsating positives outside pool 1: the design's premise is broken"
    assert pool.loc[pool["rotation"] == 1, "rotation_period"].notna().all(), "a TARS rotator has no period"

    strata = []
    for rotation, period in zip(pool["rotation"], pool["rotation_period"]):
        strata.append(assign_stratum(int(rotation), float(period)))
    pool["stratum"] = strata

    rng = np.random.default_rng(seed)
    split_of: dict[int, str] = {}
    for stratum in STRATUM_PRIORITY:
        split_of.update(split_stratum(pool.loc[pool["stratum"] == stratum, "tic_id"].to_numpy(), rng))
    pool["split"] = pool["tic_id"].map(split_of)

    ext = other_catalogue_flags(pool["tic_id"])
    is_rot = pool["rotation"] == 1
    pool["no_flare"] = pool["flare_ever"] == 0
    strict_pos = is_rot & pool["no_flare"] & ~ext["in_non_rotation_ext"]
    strict_neg = ~is_rot & pool["no_flare"] & ~ext["in_any_ext"]
    pool["strict"] = strict_pos | strict_neg
    return pool


def log_composition(pool: pd.DataFrame) -> None:
    """Stratum x split table, plus what each sensitivity cut keeps and drops (decision visibility)."""
    table = pool.groupby(["stratum", "split"]).size().unstack(fill_value=0).reindex(
        index=STRATUM_PRIORITY, columns=["train", "val", "test"])
    for line in table.to_string().splitlines():
        log.info(line)
    is_rot = pool["rotation"] == 1
    log.info(f"headline : {int(is_rot.sum())} pos / {int((~is_rot).sum())} neg, prevalence {is_rot.mean():.4f}")
    for flag in ["no_flare", "strict"]:
        kept = pool[flag]
        log.info(f"{flag:9s}: {int((is_rot & kept).sum())} pos / {int((~is_rot & kept).sum())} neg, "
                 f"prevalence {is_rot[kept].mean():.4f} | dropped {int((is_rot & ~kept).sum())} pos, "
                 f"{int((~is_rot & ~kept).sum())} neg")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build pool 3, the rotation eval pool.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the splits (an eval split: fixed).")
    ap.add_argument("--out", default=None, help="Default: processed/subset/rotation_pool.parquet")
    args = ap.parse_args()

    pool = build_pool(args.seed)
    log_composition(pool)
    out_path = Path(args.out) if args.out else repo_root / "processed" / "subset" / "rotation_pool.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool[["tic_id", "split", "stratum", "no_flare", "strict"]].to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(pool)} TICs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
