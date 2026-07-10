"""build_sequences_bulk.py — Stage 0b (bulk-download variant).

Replaces the per-star lightkurve.search_lightcurve() approach in build_sequences.py
with MAST pre-built curl-script downloads.  For each sector in spoc_sector_map.csv:

  1. Fetch (and cache) the MAST curl script for that sector.
  2. Parse it to extract (TIC ID, URL) pairs.
  3. Filter to target TICs not already done.
  4. Download FITS files in parallel (ThreadPoolExecutor, max 16 workers).
  5. Read each FITS directly with astropy (PDCSAP_FLUX, mask QUALITY != 0).
  6. Run the same find_segments → mad_normalize → slide_windows pipeline.
  7. Save .npz files to processed/sequences/ (identical format to build_sequences.py).
  8. Delete each FITS immediately after processing.
  9. Checkpoint progress to processed/build_sequences_bulk_progress.csv.

Usage:
    python src/build_sequences_bulk.py
    python src/build_sequences_bulk.py --resume
    python src/build_sequences_bulk.py --limit 5 --sectors 1        # smoke test
    python src/build_sequences_bulk.py --sectors 1-10 --resume
"""
from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import socket

import numpy as np
import pandas as pd
import requests
import requests.exceptions
import urllib3.exceptions
from astropy.io import fits

SHORT_NAN_RUN = 10
MAX_FLUX_ABSMAX: float = 20.0
_TIME_GAP_MULTIPLIER = 5  # cadences with gap > 5x median diff get flux=NaN, splitting segments at real observing breaks (e.g. mid-sector downlink). See docs/adr/0003-segment-on-time-gap.md.

_TRANSIENT_EXCEPTIONS: tuple = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.NewConnectionError,
    urllib3.exceptions.MaxRetryError,
    urllib3.exceptions.ReadTimeoutError,
    socket.gaierror,
    socket.timeout,
    ConnectionError,
    TimeoutError,
)

_BACKOFF_SCHEDULE: tuple = (5, 15, 45)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 0b (bulk): download + segment + window TESS light curves via MAST curl scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sector-map", default="processed/spoc_sector_map.csv",
                    help="CSV with columns tic_id, tmag, sector")
    ap.add_argument("--out-dir", default="processed/sequences",
                    help="Where to save per-segment .npz files")
    ap.add_argument("--progress-csv",
                    default="processed/build_sequences_bulk_progress.csv",
                    help="Per-(TIC,sector) checkpoint")
    ap.add_argument("--curl-dir", default="data/curl_scripts",
                    help="Directory for cached per-sector curl scripts")
    ap.add_argument("--fits-dir", default="data/fits_temp",
                    help="Temporary directory for downloaded FITS files")
    ap.add_argument("--log-file", default="build_sequences_bulk.log")
    ap.add_argument("--seq-len", type=int, default=4,
                    help="Min Class-A windows for a segment to be saved")
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--gap-threshold", type=int, default=1,
                    help="Min consecutive NaNs that define a segment break")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel FITS download threads per sector")
    ap.add_argument("--sectors", default=None,
                    help="Restrict to sectors, e.g. '1' or '1-10' or '1,3,7'")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total unique TICs processed (smoke-test helper)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (tic_id, sector) pairs already 'done' in progress CSV")
    ap.add_argument("--refresh-curl", action="store_true",
                    help="Re-download curl scripts even if cached on disk")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="Flush progress CSV every N completed (TIC,sector) pairs")
    ap.add_argument("--max-consecutive-errors", type=int, default=10,
                    help="Abort sector after this many consecutive errors (0=disable)")
    ap.add_argument("--max-flux-absmax", type=float, default=MAX_FLUX_ABSMAX,
                    help="Reject a run if any window's abs-max exceeds this threshold "
                         "(MAD-normalised units). Set <= 0 to disable. Default: %(default)s")
    return ap.parse_args()


def _parse_sectors(spec: str | None, available: list[int]) -> list[int]:
    """Parse --sectors argument into a sorted list of sector numbers."""
    if spec is None:
        return sorted(available)
    sectors: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            sectors.update(range(int(lo), int(hi) + 1))
        else:
            sectors.add(int(part))
    return sorted(s for s in sectors if s in set(available))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_sequences_bulk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    pairs_done: int = 0
    pairs_error: int = 0
    segments_total: int = 0
    segments_saved: int = 0
    segments_too_short: int = 0
    windows_class_a: int = 0
    windows_class_b: int = 0
    windows_class_c: int = 0
    segments_outlier: int = 0


# ---------------------------------------------------------------------------
# Core algorithm — copied verbatim from build_sequences.py
# ---------------------------------------------------------------------------

def find_segments(flux: np.ndarray, gap_threshold: int) -> list[tuple[int, int]]:
    n = len(flux)
    if n == 0:
        return []
    nan = np.isnan(flux)
    big_runs: list[tuple[int, int]] = []
    in_nan = bool(nan[0])
    run_start = 0
    for i in range(1, n):
        if bool(nan[i]) != in_nan:
            if in_nan and (i - run_start) >= gap_threshold:
                big_runs.append((run_start, i))
            in_nan = bool(nan[i])
            run_start = i
    if in_nan and (n - run_start) >= gap_threshold:
        big_runs.append((run_start, n))
    segments: list[tuple[int, int]] = []
    cursor = 0
    for rs, re in big_runs:
        if rs > cursor:
            segments.append((cursor, rs))
        cursor = re
    if cursor < n:
        segments.append((cursor, n))
    return [(s, e) for (s, e) in segments if np.any(~np.isnan(flux[s:e]))]


def mad_normalize(seg: np.ndarray) -> np.ndarray:
    med = np.nanmedian(seg)
    mad = np.nanmedian(np.abs(seg - med))
    if not np.isfinite(mad) or mad == 0:
        return seg - med
    return (seg - med) / (1.4826 * mad)


def longest_nan_run(window: np.ndarray) -> int:
    nan = np.isnan(window)
    if not nan.any():
        return 0
    longest = 0
    current = 0
    for v in nan:
        if v:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def classify_window(window: np.ndarray) -> str:
    if not np.isnan(window).any():
        return "A"
    return "B" if longest_nan_run(window) <= SHORT_NAN_RUN else "C"


def slide_windows(seg: np.ndarray, T: int, stride: int) -> Iterable[np.ndarray]:
    n = len(seg)
    start = 0
    while start + T <= n:
        yield seg[start : start + T]
        start += stride


def _one_line(msg: str) -> str:
    return msg.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Network retry
# ---------------------------------------------------------------------------

def _call_with_retry(func, label: str, logger: logging.Logger):
    max_attempts = len(_BACKOFF_SCHEDULE) + 1
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except _TRANSIENT_EXCEPTIONS as e:
            last_exc = e
            if attempt >= max_attempts:
                break
            wait = _BACKOFF_SCHEDULE[attempt - 1]
            logger.warning(
                f"{label}: transient {type(e).__name__}: {e} "
                f"(attempt {attempt}/{max_attempts}); retrying in {wait}s"
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Curl script fetch + parse
# ---------------------------------------------------------------------------

_CURL_BASE = (
    "https://archive.stsci.edu/missions/tess/download_scripts/sector/"
    "tesscurl_sector_{sector}_lc.sh"
)
_TIC_RE = re.compile(r"(\d{16})-\d{4}-s_lc\.fits")


def fetch_curl_script(
    sector: int,
    curl_dir: Path,
    refresh: bool,
    logger: logging.Logger,
) -> list[tuple[int, str]]:
    """Return list of (tic_id, url) for every entry in the sector curl script."""
    cache_path = curl_dir / f"tesscurl_sector_{sector}_lc.sh"
    if cache_path.exists() and not refresh:
        logger.debug(f"Sector {sector}: using cached curl script {cache_path}")
        text = cache_path.read_text(encoding="utf-8")
    else:
        url = _CURL_BASE.format(sector=sector)
        logger.info(f"Sector {sector}: downloading curl script from MAST")
        resp = _call_with_retry(
            lambda: requests.get(url, timeout=60),
            label=f"curl_script sector {sector}",
            logger=logger,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Sector {sector}: curl script HTTP {resp.status_code} from {url}"
            )
        text = resp.text
        curl_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        logger.info(f"Sector {sector}: curl script cached to {cache_path}")

    pairs: list[tuple[int, str]] = []
    # Each curl line (unquoted): curl -C - -L -o <filename> <url>
    for line in text.splitlines():
        if not line.startswith("curl"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        url = parts[-1]
        m_tic = _TIC_RE.search(line)
        if m_tic and url.startswith("http"):
            pairs.append((int(m_tic.group(1)), url))
    logger.info(f"Sector {sector}: parsed {len(pairs)} entries from curl script")
    return pairs


# ---------------------------------------------------------------------------
# FITS download + read
# ---------------------------------------------------------------------------

def download_fits(url: str, dest_path: Path, logger: logging.Logger) -> None:
    def _get():
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code != 200:
            raise ConnectionError(f"HTTP {r.status_code} for {url}")
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)

    _call_with_retry(_get, label=f"download {dest_path.name}", logger=logger)


def read_fits(fits_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TIME (TBJD) and PDCSAP_FLUX from a TESS SPOC FITS, applying:
      1. drop cadences where TIME is NaN (matches lightkurve internal filter)
      2. inject NaN flux at post-gap cadences (gap > _TIME_GAP_MULTIPLIER x median
         diff) so find_segments splits at real observing breaks like the mid-sector
         downlink, instead of silently stitching across them
      3. mask QUALITY != 0 cadences to NaN

    Step 2 is the intentional divergence from build_sequences.py / lightkurve;
    see docs/adr/0003-segment-on-time-gap.md.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[1].data
        time_arr = np.asarray(data["TIME"], dtype=np.float32)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float32)
        quality = np.asarray(data["QUALITY"])
    valid = ~np.isnan(time_arr)
    time_arr = time_arr[valid]
    flux = flux[valid]
    quality = quality[valid]
    if len(time_arr) > 1:
        diffs = np.diff(time_arr)
        median_diff = np.median(diffs)
        if np.isfinite(median_diff) and median_diff > 0:
            gap_mask = np.concatenate(
                [[False], diffs > _TIME_GAP_MULTIPLIER * median_diff]
            )
            flux[gap_mask] = np.nan
    flux[quality != 0] = np.nan
    return time_arr, flux


# ---------------------------------------------------------------------------
# Per-(TIC,sector) result
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    tic_id: int
    sector: int
    status: str          # 'done' | 'error'
    n_segments_saved: int
    error_msg: str
    d_segs_total: int = 0
    d_segs_saved: int = 0
    d_segs_short: int = 0
    d_win_a: int = 0
    d_win_b: int = 0
    d_win_c: int = 0
    d_segs_outlier: int = 0


# ---------------------------------------------------------------------------
# Core per-(TIC,sector) processor
# ---------------------------------------------------------------------------

def process_one(
    tic_id: int,
    sector: int,
    url: str,
    out_dir: Path,
    fits_dir: Path,
    seq_len: int,
    window_size: int,
    stride: int,
    gap_threshold: int,
    max_flux_absmax: float,
    logger: logging.Logger,
) -> PairResult:
    res = PairResult(tic_id=tic_id, sector=sector,
                     status="error", n_segments_saved=0, error_msg="")
    dest_path = fits_dir / f"TIC{tic_id:016d}_s{sector:02d}_lc.fits"
    try:
        download_fits(url, dest_path, logger)
        time_arr, flux = read_fits(dest_path)
    except Exception as e:
        logger.error(f"TIC {tic_id} s{sector:02d}: download/read failed: {e}")
        res.error_msg = _one_line(f"download: {type(e).__name__}: {e}")
        _try_unlink(dest_path)
        return res

    try:
        for seg_idx, (s, e) in enumerate(find_segments(flux, gap_threshold)):
            res.d_segs_total += 1
            seg = mad_normalize(flux[s:e].copy())
            seg_time = time_arr[s:e]

            current_run: list[np.ndarray] = []
            current_run_times: list[np.ndarray] = []
            runs: list[list[np.ndarray]] = []
            runs_times: list[list[np.ndarray]] = []

            for w, t in zip(slide_windows(seg, window_size, stride),
                            slide_windows(seg_time, window_size, stride)):
                klass = classify_window(w)
                if klass == "A":
                    res.d_win_a += 1
                    current_run.append(w.copy())
                    current_run_times.append(t.copy())
                else:
                    if klass == "B":
                        res.d_win_b += 1
                    else:
                        res.d_win_c += 1
                    if current_run:
                        runs.append(current_run)
                        runs_times.append(current_run_times)
                        current_run = []
                        current_run_times = []

            if current_run:
                runs.append(current_run)
                runs_times.append(current_run_times)

            for run_idx, (run, run_times) in enumerate(zip(runs, runs_times)):
                if len(run) >= seq_len:
                    arr = np.stack(run, axis=0).astype(np.float32)
                    arr = arr.reshape(arr.shape[0], window_size, 1)
                    if max_flux_absmax > 0 and np.abs(arr).max() > max_flux_absmax:
                        res.d_segs_outlier += 1
                        continue
                    times_stacked = np.stack(run_times, axis=0).astype(np.float32)
                    out_path = out_dir / (
                        f"TIC{tic_id:010d}_s{sector:02d}"
                        f"_seg{seg_idx:02d}_run{run_idx:02d}.npz"
                    )
                    np.savez(
                        out_path,
                        windows=arr,
                        times=times_stacked,
                        tic_id=np.int64(tic_id),
                        sector=np.int64(sector),
                        seg_idx=np.int64(seg_idx),
                        run_idx=np.int64(run_idx),
                        n_windows=np.int64(arr.shape[0]),
                    )
                    res.d_segs_saved += 1
                    res.n_segments_saved += 1
                else:
                    res.d_segs_short += 1

        res.status = "done"
    except Exception as e:
        logger.error(f"TIC {tic_id} s{sector:02d}: processing failed: {e}")
        res.error_msg = _one_line(f"process: {type(e).__name__}: {e}")
    finally:
        _try_unlink(dest_path)

    return res


def _try_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Progress CSV
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["tic_id", "sector", "status", "n_segments_saved", "error_msg"]


def load_progress(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROGRESS_COLS)
    try:
        df = pd.read_csv(path, dtype={"tic_id": int, "sector": int})
    except Exception:
        return pd.DataFrame(columns=PROGRESS_COLS)
    for col in PROGRESS_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[PROGRESS_COLS]


def save_progress(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Resume: scan existing .npz files for already-done (tic_id, sector) pairs
# ---------------------------------------------------------------------------

_NPZ_RE = re.compile(r"TIC(\d+)_s(\d+)_")


def scan_existing_npz(sequences_dir: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not sequences_dir.exists():
        return done
    for p in sequences_dir.glob("TIC*_s*_*.npz"):
        m = _NPZ_RE.match(p.name)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_INTERRUPTED = False


def _sigint_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True


def main() -> int:
    args = parse_args()

    sector_map_path = Path(args.sector_map)
    out_dir = Path(args.out_dir)
    progress_csv = Path(args.progress_csv)
    curl_dir = Path(args.curl_dir)
    fits_dir = Path(args.fits_dir)
    log_file = Path(args.log_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    curl_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    # Clean up any leftover FITS from a prior interrupted run.
    stale = list(fits_dir.glob("*.fits"))
    if stale:
        logger.warning(f"Deleting {len(stale)} leftover FITS from prior run in {fits_dir}")
        for p in stale:
            _try_unlink(p)

    if not sector_map_path.exists():
        logger.error(f"Sector map not found: {sector_map_path}")
        return 1

    df_map = pd.read_csv(sector_map_path, dtype={"tic_id": int, "sector": int})
    available_sectors = sorted(df_map["sector"].unique().tolist())
    sectors = _parse_sectors(args.sectors, available_sectors)
    if not sectors:
        logger.error(f"No valid sectors matched --sectors={args.sectors!r}. "
                     f"Available: {available_sectors[:10]}…")
        return 1

    # Target TIC set (all TICs in sector map, optionally capped by --limit).
    all_tics: set[int] = set(df_map["tic_id"].unique())
    if args.limit is not None:
        all_tics = set(sorted(all_tics)[: args.limit])
        logger.info(f"--limit {args.limit}: capped to {len(all_tics)} unique TICs")

    # Build done_set from progress CSV + existing .npz files.
    progress = load_progress(progress_csv)
    done_set: set[tuple[int, int]] = set(
        zip(
            progress.loc[progress["status"] == "done", "tic_id"].astype(int),
            progress.loc[progress["status"] == "done", "sector"].astype(int),
        )
    )
    if args.resume:
        npz_done = scan_existing_npz(out_dir)
        before = len(done_set)
        done_set |= npz_done
        logger.info(
            f"Resume: {before} pairs from progress CSV + "
            f"{len(npz_done)} (TIC,sector) pairs inferred from existing .npz files "
            f"= {len(done_set)} total already-done."
        )

    progress_records: list[dict] = progress.to_dict("records")
    progress_index: dict[tuple[int, int], int] = {
        (int(r["tic_id"]), int(r["sector"])): i
        for i, r in enumerate(progress_records)
    }

    stats = Stats()
    completed = 0
    signal.signal(signal.SIGINT, _sigint_handler)

    def _apply_result(res: PairResult) -> None:
        nonlocal completed
        completed += 1
        stats.segments_total    += res.d_segs_total
        stats.segments_saved    += res.d_segs_saved
        stats.segments_too_short += res.d_segs_short
        stats.windows_class_a   += res.d_win_a
        stats.windows_class_b   += res.d_win_b
        stats.windows_class_c   += res.d_win_c
        stats.segments_outlier  += res.d_segs_outlier
        if res.status == "done":
            stats.pairs_done += 1
        else:
            stats.pairs_error += 1
        rec = {
            "tic_id": res.tic_id, "sector": res.sector,
            "status": res.status, "n_segments_saved": res.n_segments_saved,
            "error_msg": res.error_msg,
        }
        key = (res.tic_id, res.sector)
        if key in progress_index:
            progress_records[progress_index[key]] = rec
        else:
            progress_index[key] = len(progress_records)
            progress_records.append(rec)

    t0 = time.time()
    logger.info(
        f"Starting build_sequences_bulk: {len(sectors)} sectors, "
        f"{len(all_tics)} target TICs, "
        f"window_size={args.window_size}, stride={args.stride}, "
        f"gap_threshold={args.gap_threshold}, seq_len={args.seq_len}, "
        f"workers={args.workers}"
    )

    for sector in sectors:
        if _INTERRUPTED:
            break

        try:
            pairs = fetch_curl_script(sector, curl_dir, args.refresh_curl, logger)
        except Exception as e:
            logger.error(f"Sector {sector}: failed to fetch curl script: {e}")
            continue

        in_target = [(t, u) for t, u in pairs if t in all_tics]
        to_process = [(t, u) for t, u in in_target if (t, sector) not in done_set]
        n_not_target = len(pairs) - len(in_target)
        n_already_done = len(in_target) - len(to_process)
        logger.info(
            f"Sector {sector}: {len(to_process)} pairs to process  "
            f"(out_of_target={n_not_target}, already_done={n_already_done})"
        )

        if not to_process:
            continue

        consecutive_errors = 0
        aborted = False

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map: dict[Future[PairResult], tuple[int, int]] = {
                pool.submit(
                    process_one,
                    tic_id, sector, url,
                    out_dir, fits_dir,
                    args.seq_len, args.window_size, args.stride, args.gap_threshold,
                    args.max_flux_absmax,
                    logger,
                ): (tic_id, sector)
                for tic_id, url in to_process
            }

            for f in as_completed(future_map):
                if _INTERRUPTED:
                    break
                tic_id, sec = future_map[f]
                try:
                    res = f.result()
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"TIC {tic_id} s{sec:02d}: unhandled:\n{tb}")
                    res = PairResult(
                        tic_id=tic_id, sector=sec, status="error",
                        n_segments_saved=0,
                        error_msg=_one_line(f"{type(e).__name__}: {e}"),
                    )

                _apply_result(res)
                consecutive_errors = (
                    0 if res.status == "done" else consecutive_errors + 1
                )

                if completed % args.checkpoint_every == 0:
                    save_progress(
                        pd.DataFrame(progress_records, columns=PROGRESS_COLS),
                        progress_csv,
                    )
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"  checkpoint: completed={completed} "
                        f"done={stats.pairs_done} err={stats.pairs_error} "
                        f"segs_saved={stats.segments_saved} "
                        f"rate={rate:.1f} pairs/s"
                    )

                if (args.max_consecutive_errors > 0
                        and consecutive_errors >= args.max_consecutive_errors):
                    logger.error(
                        f"Sector {sector}: {consecutive_errors} consecutive errors "
                        f"— aborting sector, will retry on --resume"
                    )
                    aborted = True
                    break

        # End-of-sector flush.
        save_progress(
            pd.DataFrame(progress_records, columns=PROGRESS_COLS),
            progress_csv,
        )
        if not aborted:
            logger.info(f"Sector {sector}: done.")

    # Final checkpoint.
    save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), progress_csv)

    elapsed = time.time() - t0
    total_class = stats.windows_class_a + stats.windows_class_b + stats.windows_class_c
    pct = lambda x: (100.0 * x / total_class) if total_class > 0 else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sequence Build (Bulk) Summary")
    logger.info("=" * 60)
    logger.info(f"(TIC,sector) pairs done:          {stats.pairs_done}")
    logger.info(f"(TIC,sector) pairs errored:       {stats.pairs_error}")
    logger.info(f"Total segments produced:          {stats.segments_total}")
    logger.info(f"Segments saved (>= seq_len):      {stats.segments_saved}")
    logger.info(f"Segments discarded (too short):   {stats.segments_too_short}")
    logger.info(f"Segments rejected (flux outlier): {stats.segments_outlier}")
    logger.info(f"Total candidate windows:          {total_class}")
    logger.info(f"  Class A (NaN-free, kept):       {stats.windows_class_a}  ({pct(stats.windows_class_a):.1f}%)")
    logger.info(f"  Class B (NaN runs all <= 10):   {stats.windows_class_b}  ({pct(stats.windows_class_b):.1f}%)")
    logger.info(f"  Class C (has NaN run > 10):     {stats.windows_class_c}  ({pct(stats.windows_class_c):.1f}%)")
    logger.info(f"Elapsed:                          {elapsed/60:.1f} min")
    logger.info("=" * 60)

    return 130 if _INTERRUPTED else 0


if __name__ == "__main__":
    raise SystemExit(main())
