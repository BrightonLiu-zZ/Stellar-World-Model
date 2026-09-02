"""L1 route (g) -- reconstruct which of our stars flatwrm2 actually SEARCHED for flares.

Seli et al. (2025), "Stellar flare morphology with TESS across the main sequence" (A&A,
arXiv:2412.12989), did not search every TESS light curve. Their Sect. 2.1, Eq. (1) smooths each
2-min PDCSAP curve with a 31-point (one hour) running median and KEEPS it only if

    sigma_ratio = STD(smoothed) / STD(original) > 0.4

"indicating that astrophysical variation dominates short-timescale random noise". Their Table 1
gives the funnel: 1,258,154 light curves through sector 69 --> 444,963 searched (35.4%).

WHY THIS MATTERS FOR THE LABEL. `flare_ever = 0` therefore mixes two populations that carry opposite
amounts of information: stars flatwrm2 searched and found clean, and stars it never looked at. Worse
for the probe, the cut runs the SAME WAY as the positive class: every flare positive necessarily
cleared 0.4, so a `flare` probe scored against a negative class containing unsearched quiet stars can
win by detecting variability that the catalog's own pre-cut wrote into the label.

The cut is fully specified in the paper and sigma_ratio is SCALE-INVARIANT (a ratio of two standard
deviations of the same series), so our per-segment MAD normalization does not perturb it and the
selection is reconstructible from the packed windows alone. No network, no second catalog.

TWO SPANS ARE EMITTED, NOT ONE. Seli computed the statistic per light curve, i.e. per sector; our
pipeline only preserves segments. `sigma_ratio_segment` is the max over a sector's segments (each
internally MAD-consistent, but shorter than a sector, which biases the smoothed STD down);
`sigma_ratio_sector` concatenates a sector's segments in time order (full span, but each piece
carries its own MAD scale). Neither is exactly the published quantity, so both are written and the
caller's validation -- every `flare_ever = 1` star was necessarily searched, so it must clear 0.4 --
decides which reconstruction is trustworthy.

Output `labels/qc/flare_search_universe.parquet`, one row per (tic_id, sector). Star-level rollup is
deliberately left to the caller: "searched" is a max over sectors, and which sectors are admissible
is a question about the CALLER's population, not about this file.

Run (repo root, swm env, PYTHONPATH=src; CPU-only):
    PYTHONUNBUFFERED=1 python -m swm.eval.flare_search_universe
    python -m swm.eval.flare_search_universe --limit 200
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("flare_search_universe")

repo_root = Path(__file__).resolve().parents[3]
_NPZ_RE = re.compile(r"^TIC(\d+)_s(\d+)_seg\d+_run\d+\.npz$")

smoothing_points = 31 # 31 x 2 min = 62 min, Seli Sect. 2.1's "one hour" running median
sigma_ratio_threshold = 0.4 # Seli Eq. (1); above it the curve was searched
flatwrm2_last_sector = 69 # the catalog's coverage; our corpus runs to 101


def sigma_ratio(flux: np.ndarray) -> float:
    """
    Seli Eq. (1) on one flux series: how much scatter survives an hour-wide running median.
    Near 1 the star's variation is astrophysical and slower than an hour; near 0 the series is
    dominated by point-to-point noise, which is what flatwrm2 declined to search.
    Returns NaN for a series too short to smooth or with zero scatter.
    """
    if flux.size < smoothing_points:
        return float("nan")
    original = float(flux.std())
    if original == 0.0:
        return float("nan")
    smoothed = median_filter(flux, size=smoothing_points, mode="nearest") # running median, edges held
    return float(smoothed.std() / original)


def segment_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one packed segment as a flat flux series ordered in time, with its cadence times.
    Class-B/C windows were discarded at pack time, so consecutive windows are not necessarily
    contiguous; the times are returned so the caller can order and report the real span.
    """
    payload = np.load(path)
    flux = payload["windows"].reshape(-1) # (n_win, 1024, 1) --> flat cadence series
    times = payload["times"].reshape(-1)
    order = np.argsort(times)
    return flux[order], times[order]


def collect_paths(seq_dir: Path, keep_tics: set[int] | None, max_sector: int) -> dict[tuple[int, int], list[Path]]:
    """Group the packed npz files by (tic_id, sector), restricted to the caller's stars and to the sectors flatwrm2 covered."""
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    with os.scandir(seq_dir) as entries:
        for entry in entries:
            match = _NPZ_RE.match(entry.name)
            if match is None:
                continue
            tic = int(match.group(1))
            sector = int(match.group(2))
            if sector > max_sector:
                continue
            if keep_tics is not None and tic not in keep_tics:
                continue
            grouped[(tic, sector)].append(Path(entry.path))
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct flatwrm2's searched-light-curve universe.")
    ap.add_argument("--pool", default="processed/subset/new_task_pool.parquet",
                    help="restrict to these stars; pass 'all' to scan the whole corpus")
    ap.add_argument("--sequences-dir", default="processed/sequences")
    ap.add_argument("--out", default="labels/qc/flare_search_universe.parquet")
    ap.add_argument("--max-sector", type=int, default=flatwrm2_last_sector)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N (tic, sector) groups")
    args = ap.parse_args()

    seq_dir = repo_root / args.sequences_dir
    assert seq_dir.is_dir(), f"sequences dir not found: {seq_dir}"

    keep_tics = None
    if args.pool != "all":
        pool_path = repo_root / args.pool
        assert pool_path.exists(), f"pool parquet not found: {pool_path}"
        keep_tics = set(pd.read_parquet(pool_path)["tic_id"].astype(int))
        log.info(f"restricting to {len(keep_tics)} pool stars")

    grouped = collect_paths(seq_dir, keep_tics, args.max_sector)
    keys = sorted(grouped)
    if args.limit is not None:
        keys = keys[: args.limit]
    log.info(f"{len(keys)} (tic, sector) groups, {sum(len(grouped[k]) for k in keys)} npz files, "
             f"sectors <= {args.max_sector}")

    rows = []
    for tic, sector in tqdm(keys, desc="sigma_ratio per (tic, sector)", total=len(keys)):
        per_segment = []
        chunks = []
        starts = []
        for path in grouped[(tic, sector)]:
            flux, times = segment_series(path)
            per_segment.append(sigma_ratio(flux))
            chunks.append(flux)
            starts.append(times[0])
        order = np.argsort(starts)
        concatenated = np.concatenate([chunks[i] for i in order])
        segment_values = np.array(per_segment, dtype=float)
        rows.append({"tic_id": tic, "sector": sector, "n_segments": len(chunks),
                     "n_cadences": int(concatenated.size),
                     "sigma_ratio_segment": float(np.nanmax(segment_values)) if np.isfinite(segment_values).any() else float("nan"),
                     "sigma_ratio_sector": sigma_ratio(concatenated)})

    frame = pd.DataFrame(rows)
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    log.info(f"wrote {out_path} ({len(frame)} rows)")

    for column in ["sigma_ratio_segment", "sigma_ratio_sector"]:
        passing = (frame[column] > sigma_ratio_threshold).mean()
        log.info(f"{column}: {frame[column].notna().sum()} finite, "
                 f"{passing} of (tic, sector) rows above {sigma_ratio_threshold} "
                 f"(Seli's published light-curve rate is 0.354)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
