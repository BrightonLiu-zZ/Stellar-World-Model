"""Task 2 (plan 2026-07-22, D7 Level B) — true per-window flare labels on the new-task pool.

Flares are transient (timescale ~7 min inside a 512-min window), so the star-level flare_ever broadcast
is mostly noise at window granularity, exactly the transit dilution problem. This folds every flatwrm2
Table-3 flare `peak_time` onto each pool star's first-segment window time span, emitting one label per
window in the SAME order new_task_extract encodes them (identical absmax-guard keep-mask + 1024->256
subdivision), so the labels align row-for-row with the pool mu cache:

  label =  1  a flare peak falls inside the window's [t0, t_last]
  label = -1  quarantine: no peak inside, but a peak within `timescale` of an edge (rise/decay tail leaks in)
  label =  0  clean negative

Only flaring pool stars (flare_ever=1) are materialized; every window of a non-flaring pool star is a
clean 0 by construction and is not written.

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.flare_window_labels --limit 40   # smoke
    python -m swm.eval.flare_window_labels
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from swm.eval.new_task_extract import MAX_ABSMAX, index_pool_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("flare_window_labels")

repo_root = Path(__file__).resolve().parents[3]
MIN_BUFFER_DAYS = 5.0 / 1440.0  # floor on the edge quarantine band, 5 min, absorbs peak-time rounding


def replay_times(npz_path: Path, window: int) -> np.ndarray:
    """First-segment window times (n_win, window) BTJD, using the exact keep-mask new_task_extract encodes with."""
    with np.load(npz_path) as data:
        windows = data["windows"]  # (N, native, 1) float32
        times = data["times"]      # (N, native) float32 BTJD
    native = windows.shape[1]
    absmax = np.abs(windows).max(axis=(1, 2))  # guard computed on flux, mirrors replay_first_segment
    keep = absmax <= MAX_ABSMAX
    return times[keep].reshape(-1, window).astype(np.float64)


def label_segment(times: np.ndarray, peaks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Per-window 1/0/-1 for one segment: peak inside window --> 1, peak within timescale of an edge --> -1."""
    t0 = times[:, 0]      # (n_win,)
    tL = times[:, -1]     # (n_win,)
    n_win = times.shape[0]
    inside = np.zeros(n_win, dtype=bool)
    near = np.zeros(n_win, dtype=bool)
    for peak, scale in zip(peaks, scales):
        buf = max(scale, MIN_BUFFER_DAYS)
        inside |= (peak >= t0) & (peak <= tL)
        near |= (peak >= t0 - buf) & (peak <= tL + buf)
    label = np.zeros(n_win, dtype=np.int8)
    label[near & ~inside] = -1
    label[inside] = 1
    return label


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-window flare labels aligned to the new-task pool mu cache.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N flaring pool TICs (smoke).")
    ap.add_argument("--pool", default=None, help="Default: processed/subset/new_task_pool.parquet")
    ap.add_argument("--flare-catalog", default=None, help="Default: data/Table3_flare_catalog.csv")
    ap.add_argument("--out", default=None, help="Default: labels/qc/flare_window_labels_pool.parquet")
    args = ap.parse_args()

    seq_dir = repo_root / "processed" / "sequences"
    pool_path = Path(args.pool) if args.pool else repo_root / "processed" / "subset" / "new_task_pool.parquet"
    cat_path = Path(args.flare_catalog) if args.flare_catalog else repo_root / "data" / "Table3_flare_catalog.csv"
    out_path = Path(args.out) if args.out else repo_root / "labels" / "qc" / "flare_window_labels_pool.parquet"

    pool = pd.read_parquet(pool_path)
    pool["tic_id"] = pool["tic_id"].astype(int)
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    flare_tics = set(canon.loc[pd.to_numeric(canon["flare_ever"], errors="coerce") == 1, "tic_id"])
    flaring = pool[pool["tic_id"].isin(flare_tics)].sort_values("tic_id").reset_index(drop=True)
    if args.limit:
        flaring = flaring.head(args.limit)
    log.info(f"flaring pool stars: {len(flaring)}")

    cat = pd.read_csv(cat_path)
    cat["TIC"] = cat["TIC"].astype(int)
    by_tic_sector = cat.groupby(["TIC", "sector"])

    npz_index = index_pool_npz(seq_dir, set(flaring["tic_id"]))
    records = []
    for row in tqdm(flaring.itertuples(index=False), desc="flare-window-labels", total=len(flaring)):
        tic = int(row.tic_id)
        entry = npz_index.get(tic)
        if entry is None:
            continue
        sector, _, npz_path = entry
        times = replay_times(npz_path, 256)
        if (tic, sector) in by_tic_sector.groups:
            sub = by_tic_sector.get_group((tic, sector))
            peaks = sub["peak_time"].to_numpy(dtype=float)
            scales = (sub["timescale"].to_numpy(dtype=float)) / 1440.0  # minutes --> days
        else:
            peaks = np.array([])
            scales = np.array([])
        label = label_segment(times, peaks, scales)
        for j in range(len(label)):
            records.append({"tic_id": tic, "split": row.split, "sector": sector,
                            "win_idx": j, "label": int(label[j])})

    df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(df)} window rows)")

    log.info("=" * 60)
    log.info("Flare window-label counters (first segment, flaring pool stars)")
    log.info("=" * 60)
    for val, tag in ((1, "in-flare"), (0, "clean"), (-1, "quarantine")):
        log.info(f"label {val} {tag}: {int((df['label'] == val).sum())}")
    pos_stars = df.loc[df["label"] == 1, "tic_id"].nunique()
    log.info(f"flaring stars with >=1 in-flare window in first segment: {pos_stars} of {len(flaring)}")
    for split in ["train", "val", "test"]:
        d = df[df["split"] == split]
        log.info(f"{split}: {int((d['label'] == 1).sum())} in-flare / {len(d)} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
