"""Task 2 (plan 2026-07-22, D1/D8) — build the separate new-task eval pool + split.

The frozen v1 subset (13,470 TICs) barely overlaps the external new-task catalogs, so the new probes
get their own eval population: every new-task positive in the npz corpus (the 4 external catalogs plus
resurrected flare) union a fresh 10k quiet draw, split TIC-disjoint 70/15/15 stratified by a single
rarest-task stratum (seed 0, an eval split so a fixed seed is correct). rotation_period is NOT here: it
rides the existing v1 caches (enough rotation stars are already in the frozen subset).

Writes `processed/subset/new_task_pool.parquet` (tic_id, split, stratum) — membership + split only;
the scorecard joins the actual label values from new_task_labels_star.csv + the canonical CSV at score
time, exactly as the v1 eval joins labels onto its frozen membership.

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.new_task_pool
    python -m swm.eval.new_task_pool --quiet 10000 --seed 0
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("new_task_pool")

repo_root = Path(__file__).resolve().parents[3]
_NPZ_RE = re.compile(r"^TIC(\d+)_s\d+_seg\d+_run\d+\.npz$")

# rarest task first: small strata are assigned before large ones so their 70/15/15 splits stay balanced
STRATUM_PRIORITY = ["rgb_vs_heb", "prot_kounkel", "flare", "solar_like_osc", "osc_giant", "quiet"]


def corpus_tics(seq_dir: Path) -> set[int]:
    """Set of TIC IDs with at least one packed-window npz (the encodable corpus)."""
    tics: set[int] = set()
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_RE.match(entry.name)
            if m:
                tics.add(int(m.group(1)))
    return tics


def assign_stratum(row: pd.Series) -> str:
    """Single split-stratum for one pool star: the rarest new-task membership it satisfies, else quiet."""
    if row.get("rgb_vs_heb_present", False):
        return "rgb_vs_heb"
    if row.get("prot_present", False):
        return "prot_kounkel"
    if row.get("flare", 0) == 1:
        return "flare"
    if row.get("solar_like_osc", 0) == 1:
        return "solar_like_osc"
    if row.get("osc_giant", 0) == 1:
        return "osc_giant"
    return "quiet"


def split_stratum(tics: np.ndarray, rng: np.random.Generator) -> dict[int, str]:
    """Shuffle one stratum's TICs and cut 70/15/15 into train/val/test, returning tic --> split."""
    order = rng.permutation(len(tics))
    n_train = int(round(0.70 * len(tics)))
    n_val = int(round(0.15 * len(tics)))
    assignment: dict[int, str] = {}
    for rank, idx in enumerate(order):
        if rank < n_train:
            assignment[int(tics[idx])] = "train"
        elif rank < n_train + n_val:
            assignment[int(tics[idx])] = "val"
        else:
            assignment[int(tics[idx])] = "test"
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the new-task eval pool + stratified split.")
    ap.add_argument("--quiet", type=int, default=10000, help="Fresh quiet-star draw size (detection negatives).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the quiet draw and the splits.")
    ap.add_argument("--out", default=None, help="Default: processed/subset/new_task_pool.parquet")
    args = ap.parse_args()

    seq_dir = repo_root / "processed" / "sequences"
    corpus = corpus_tics(seq_dir)
    log.info(f"npz corpus: {len(corpus)} TICs")

    ext = pd.read_csv(repo_root / "labels" / "new_task_labels_star.csv")
    ext["tic_id"] = ext["tic_id"].astype(int)
    ext["rgb_vs_heb_present"] = ext["rgb_vs_heb"].notna()
    ext["prot_present"] = ext["prot_kounkel"].notna()

    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["flare"] = pd.to_numeric(canon["flare_ever"], errors="coerce").fillna(0).astype(int)
    flare_pos = set(canon.loc[canon["flare"] == 1, "tic_id"]) & corpus

    members = pd.DataFrame({"tic_id": sorted(set(ext["tic_id"]) | flare_pos)})
    members = members.merge(
        ext[["tic_id", "osc_giant", "solar_like_osc", "rgb_vs_heb_present", "prot_present"]],
        on="tic_id", how="left")
    members = members.merge(canon[["tic_id", "flare"]], on="tic_id", how="left")
    for col in ["osc_giant", "solar_like_osc", "flare"]:
        members[col] = members[col].fillna(0).astype(int)
    for col in ["rgb_vs_heb_present", "prot_present"]:
        members[col] = members[col].fillna(False).astype(bool)

    rng = np.random.default_rng(args.seed)
    v1pos = (canon["transit"] + canon["eb"] + canon["pulsating"] + canon["rotation"]) > 0
    quiet_universe = set(canon.loc[(~v1pos) & (canon["flare"] == 0), "tic_id"]) & corpus
    quiet_universe -= set(members["tic_id"])
    quiet_sorted = np.array(sorted(quiet_universe))
    quiet_draw = rng.choice(quiet_sorted, size=min(args.quiet, len(quiet_sorted)), replace=False)
    quiet = pd.DataFrame({"tic_id": quiet_draw})
    for col in ["osc_giant", "solar_like_osc", "flare"]:
        quiet[col] = 0
    for col in ["rgb_vs_heb_present", "prot_present"]:
        quiet[col] = False

    pool = pd.concat([members, quiet], ignore_index=True)
    strata = []
    for _, row in pool.iterrows():
        strata.append(assign_stratum(row))
    pool["stratum"] = strata

    split_of: dict[int, str] = {}
    for stratum in STRATUM_PRIORITY:
        tics = pool.loc[pool["stratum"] == stratum, "tic_id"].to_numpy()
        split_of.update(split_stratum(tics, rng))
    pool["split"] = pool["tic_id"].map(split_of)

    out_path = Path(args.out) if args.out else repo_root / "processed" / "subset" / "new_task_pool.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool[["tic_id", "split", "stratum"]].sort_values("tic_id").to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(pool)} TICs)")

    log.info("=" * 60)
    log.info("New-task pool composition (stratum x split)")
    log.info("=" * 60)
    table = pool.groupby(["stratum", "split"]).size().unstack(fill_value=0)
    for line in table.to_string().splitlines():
        log.info(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
