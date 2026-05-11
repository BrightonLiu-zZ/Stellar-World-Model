"""build_sequences.py — Stage 0b of the stellar world model pipeline.

For each TIC ID in processed/df_final.csv:
  1. Download all SPOC 2-min light curves via lightkurve
  2. Mask QUALITY != 0 cadences to NaN
  3. Split each sector into segments at NaN runs >= gap_threshold cadences
  4. NaN-aware MAD-normalize each segment (NO interpolation at any stage)
  5. Slide T=1024 / stride=512 windows starting at index 0 of each segment
  6. Classify each candidate window:
       Class A — zero NaN          (KEPT)
       Class B — has NaN, all NaN runs <= 10 cadences  (discarded, counted)
       Class C — has NaN, at least one run > 10 cadences (discarded, counted)
  7. If a segment has >= seq_len Class-A windows, save them to
        processed/sequences/TIC<id:010d>_s<sector:02d>_seg<idx:02d>.npz

Per-star checkpointing to processed/build_sequences_progress.csv enables
--resume after interruption (Ctrl+C is caught and the checkpoint is flushed).

Usage examples:
    python build_sequences.py
    python build_sequences.py --resume
    python build_sequences.py --limit 5            # quick smoke test on 5 stars
    python build_sequences.py --seq-len 8 --window-size 1024 --stride 512
"""
from __future__ import annotations

import argparse
import logging
import re
import signal
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np
import pandas as pd

import requests.exceptions
import urllib3.exceptions

import lightkurve as lk

SHORT_NAN_RUN = 10  # NaN runs <= this length classify a window as B; longer → C

# Transient network exceptions that warrant retry. lightkurve calls into
# astroquery -> requests -> urllib3 -> socket; any layer can surface a blip.
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

# Wait (seconds) before retries 2, 3, 4, 5 → 5 attempts total per network call.
_BACKOFF_SCHEDULE: tuple = (5, 15, 45, 120)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 0b: download + segment + window TESS light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-csv", default="processed/df_final.csv",
                    help="CSV with TIC IDs in column 'ID'")
    ap.add_argument("--out-dir", default="processed/sequences",
                    help="Where to save per-segment .npz files")
    ap.add_argument("--progress-csv", default="processed/build_sequences_progress.csv",
                    help="Per-star checkpoint")
    ap.add_argument("--log-file", default="build_sequences.log",
                    help="Log file path")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="Min Class-A windows for a segment to be saved")
    ap.add_argument("--window-size", type=int, default=1024, help="Window length T")
    ap.add_argument("--stride", type=int, default=512, help="Stride between windows")
    ap.add_argument("--gap-threshold", type=int, default=100,
                    help="Min consecutive NaNs that define a segment break")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Flush progress CSV every N stars")
    ap.add_argument("--resume", action="store_true",
                    help="Skip TIC IDs whose status is 'done' or 'no_data' in the progress CSV")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N stars (for testing)")
    ap.add_argument("--max-consecutive-errors", type=int, default=5,
                    help="Abort the run after this many consecutive errored stars "
                         "(likely outage). 0 disables the safety valve.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_sequences")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Run-scoped statistics
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    stars_processed: int = 0
    stars_no_data: int = 0
    stars_error: int = 0
    sectors_examined: int = 0
    segments_total: int = 0
    segments_saved: int = 0
    segments_too_short: int = 0
    windows_class_a: int = 0
    windows_class_b: int = 0
    windows_class_c: int = 0


# ---------------------------------------------------------------------------
# Core algorithm — segmentation, normalization, classification
# ---------------------------------------------------------------------------

def find_segments(flux: np.ndarray, gap_threshold: int) -> list[tuple[int, int]]:
    """Split `flux` at runs of >= gap_threshold consecutive NaNs.

    Returns half-open (start, end) slice indices. Shorter NaN runs survive
    inside segments and are handled later by the per-window NaN classifier.
    Segments that are entirely NaN are dropped.
    """
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
    """NaN-aware MAD-normalize: (x - nanmedian) / (1.4826 * MAD).

    Falls back to a centered (median-subtracted) result for degenerate
    constant-flux segments where MAD is zero. NaNs are preserved in either case.
    """
    med = np.nanmedian(seg)
    mad = np.nanmedian(np.abs(seg - med))
    if not np.isfinite(mad) or mad == 0:
        return seg - med
    return (seg - med) / (1.4826 * mad)


def longest_nan_run(window: np.ndarray) -> int:
    """Length of the longest consecutive-NaN run in `window` (0 if no NaNs)."""
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
    """Return 'A' (no NaN), 'B' (all NaN runs <= SHORT_NAN_RUN), or 'C' (has long run)."""
    if not np.isnan(window).any():
        return "A"
    return "B" if longest_nan_run(window) <= SHORT_NAN_RUN else "C"


def slide_windows(seg: np.ndarray, T: int, stride: int) -> Iterable[np.ndarray]:
    """Yield non-overlapping starts at [0, stride, 2*stride, ...] while a full T fits."""
    n = len(seg)
    start = 0
    while start + T <= n:
        yield seg[start : start + T]
        start += stride


# ---------------------------------------------------------------------------
# Lightkurve column access — case-insensitive (per astro-api-queries skill)
# ---------------------------------------------------------------------------

def _get_column(lc, *candidates: str):
    """Case-insensitive column lookup on a LightCurve TimeSeries. Returns None if absent."""
    cols_lower = {c.lower(): c for c in lc.colnames}
    for cand in candidates:
        real = cols_lower.get(cand.lower())
        if real is not None:
            return lc[real]
    return None


# ---------------------------------------------------------------------------
# Network retry — transient errors only, exponential backoff
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _make_download_fn(sr, logger: logging.Logger, tic_id: int) -> Callable[[], object]:
    """Wrap sr.download_all to auto-delete corrupt cached FITS files.

    If lightkurve raises LightkurveError saying a file is corrupt (caused by an
    earlier interrupted download leaving a truncated file on disk), we delete the
    bad cache file and re-raise as ConnectionError so _call_with_retry treats it
    as a transient error and retries a fresh download.
    """
    def _download():
        try:
            return sr.download_all(quality_bitmask="none")
        except lk.LightkurveError as e:
            msg = str(e)
            if "may be corrupt" in msg or "interrupted download" in msg:
                m = re.search(r"Data product (.+?) of type", msg)
                if m:
                    corrupt = Path(m.group(1).strip())
                    if corrupt.exists():
                        corrupt.unlink()
                        logger.warning(
                            f"TIC {tic_id}: deleted corrupt cache file "
                            f"'{corrupt.name}'; download will be retried"
                        )
                raise ConnectionError(f"corrupt cache deleted, retrying: {e}") from e
            raise  # non-corrupt LightkurveError propagates immediately
    return _download


def _one_line(msg: str) -> str:
    """Collapse newlines in an error message to spaces so it stays on one CSV line."""
    return msg.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()


def _call_with_retry(
    func: Callable[[], T],
    label: str,
    tic_id: int,
    logger: logging.Logger,
) -> T:
    """Invoke `func()` with exponential-backoff retry on transient network errors.

    Non-transient exceptions propagate immediately. After all attempts fail,
    the last transient exception is re-raised so the caller can record it.
    """
    max_attempts = len(_BACKOFF_SCHEDULE) + 1
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except _TRANSIENT_EXCEPTIONS as e:
            last_exc = e
            if attempt >= max_attempts:
                break
            wait = _BACKOFF_SCHEDULE[attempt - 1]
            logger.warning(
                f"TIC {tic_id}: {label} transient {type(e).__name__}: {e} "
                f"(attempt {attempt}/{max_attempts}); retrying in {wait}s"
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Per-star processing
# ---------------------------------------------------------------------------

def process_star(
    tic_id: int,
    out_dir: Path,
    seq_len: int,
    window_size: int,
    stride: int,
    gap_threshold: int,
    stats: Stats,
    logger: logging.Logger,
) -> tuple[str, int, str]:
    """Download, process, and save sequences for one TIC.

    Returns (status, n_segments_saved_this_star, error_msg).
    status in {'done', 'no_data', 'error'}.
    """
    try:
        sr = _call_with_retry(
            lambda: lk.search_lightcurve(
                f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120
            ),
            label="search_lightcurve",
            tic_id=tic_id,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"TIC {tic_id}: search failed: {type(e).__name__}: {e}")
        return ("error", 0, _one_line(f"search: {type(e).__name__}: {e}"))

    if len(sr) == 0:
        logger.info(f"TIC {tic_id}: no SPOC 2-min data")
        return ("no_data", 0, "")

    try:
        lcs = _call_with_retry(
            _make_download_fn(sr, logger, tic_id),
            label="download_all",
            tic_id=tic_id,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"TIC {tic_id}: download_all failed: {type(e).__name__}: {e}")
        return ("error", 0, _one_line(f"download: {type(e).__name__}: {e}"))

    if lcs is None or len(lcs) == 0:
        logger.info(f"TIC {tic_id}: download returned empty collection")
        return ("no_data", 0, "")

    n_segments_saved = 0
    seen_sectors: set[int] = set()

    for lc in lcs:
        try:
            sector = int(lc.sector)
        except Exception:
            sector = int(lc.meta.get("SECTOR", -1)) if hasattr(lc, "meta") else -1

        if sector in seen_sectors:
            continue  # MAST occasionally returns duplicate (sector, pipeline) pairs
        seen_sectors.add(sector)
        stats.sectors_examined += 1

        flux_col = _get_column(lc, "PDCSAP_FLUX", "pdcsap_flux")
        if flux_col is None:
            logger.warning(f"TIC {tic_id} sector {sector}: PDCSAP_FLUX column absent — skipping sector")
            continue
        flux = np.asarray(flux_col.value, dtype=np.float32)

        quality_col = _get_column(lc, "QUALITY", "quality")
        if quality_col is not None:
            quality = np.asarray(quality_col.value)
            flux[quality != 0] = np.nan
        else:
            logger.warning(f"TIC {tic_id} sector {sector}: QUALITY column absent — proceeding without quality mask")

        for seg_idx, (s, e) in enumerate(find_segments(flux, gap_threshold)):
            stats.segments_total += 1
            seg = mad_normalize(flux[s:e].copy())

            class_a: list[np.ndarray] = []
            for w in slide_windows(seg, window_size, stride):
                klass = classify_window(w)
                if klass == "A":
                    stats.windows_class_a += 1
                    class_a.append(w.copy())
                elif klass == "B":
                    stats.windows_class_b += 1
                else:
                    stats.windows_class_c += 1

            if len(class_a) >= seq_len:
                arr = np.stack(class_a, axis=0).astype(np.float32)
                arr = arr.reshape(arr.shape[0], window_size, 1)
                out_path = out_dir / f"TIC{tic_id:010d}_s{sector:02d}_seg{seg_idx:02d}.npz"
                np.savez(
                    out_path,
                    windows=arr,
                    tic_id=np.int64(tic_id),
                    sector=np.int64(sector),
                    seg_idx=np.int64(seg_idx),
                    n_windows=np.int64(arr.shape[0]),
                )
                stats.segments_saved += 1
                n_segments_saved += 1
            else:
                stats.segments_too_short += 1

    return ("done", n_segments_saved, "")


# ---------------------------------------------------------------------------
# Progress CSV
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["tic_id", "status", "n_segments_saved", "error_msg"]


def load_progress(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROGRESS_COLS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=PROGRESS_COLS)
    for col in PROGRESS_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[PROGRESS_COLS]


def save_progress(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _prioritize_errors(tic_ids: list[int], progress: pd.DataFrame) -> list[int]:
    """Move previously-errored stars to the front of the queue.

    On --resume, stars whose last run ended in status='error' are retried before
    stars that have never been attempted.  This lets you confirm a transient failure
    (e.g. network outage) is resolved without waiting through hundreds of new stars.
    """
    error_set: set[int] = set(
        progress.loc[progress["status"] == "error", "tic_id"].astype(int)
    )
    if not error_set:
        return tic_ids
    return (
        [t for t in tic_ids if t in error_set]
        + [t for t in tic_ids if t not in error_set]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_INTERRUPTED = False


def _sigint_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True


def directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 ** 3)


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    progress_csv = Path(args.progress_csv)
    log_file = Path(args.log_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    if not input_csv.exists():
        logger.error(f"Input CSV not found: {input_csv}")
        logger.error("Run the export cell at the bottom of "
                     "src/notebooks/charaterize_data copy.ipynb to produce it.")
        return 1

    df_in = pd.read_csv(input_csv)
    if "ID" not in df_in.columns:
        logger.error(f"Input CSV missing 'ID' column. Found: {list(df_in.columns)}")
        return 1
    tic_ids = df_in["ID"].astype(int).tolist()

    progress = load_progress(progress_csv)
    if args.resume and len(progress) > 0:
        skip = set(
            progress.loc[progress["status"].isin(["done", "no_data"]), "tic_id"].astype(int)
        )
        before = len(tic_ids)
        tic_ids = [t for t in tic_ids if t not in skip]
        logger.info(f"Resume: skipping {before - len(tic_ids)} already-processed stars; "
                    f"{len(tic_ids)} remaining.")
        tic_ids = _prioritize_errors(tic_ids, progress)
        n_errors = sum(1 for t in tic_ids
                       if t in set(progress.loc[progress["status"] == "error", "tic_id"].astype(int)))
        if n_errors:
            logger.info(f"Resume: {n_errors} previously-errored stars queued first.")

    if args.limit is not None:
        tic_ids = tic_ids[: args.limit]
        logger.info(f"--limit set: processing only first {len(tic_ids)} stars.")

    signal.signal(signal.SIGINT, _sigint_handler)

    stats = Stats()
    progress_records: list[dict] = progress.to_dict("records")
    progress_index = {int(r["tic_id"]): i for i, r in enumerate(progress_records)}

    logger.info(
        f"Starting build_sequences: {len(tic_ids)} stars to process, "
        f"window_size={args.window_size}, stride={args.stride}, "
        f"gap_threshold={args.gap_threshold}, seq_len={args.seq_len}"
    )
    t0 = time.time()
    consecutive_errors = 0
    aborted_for_outage = False

    for i, tic_id in enumerate(tic_ids, start=1):
        if _INTERRUPTED:
            logger.warning("Ctrl+C received — flushing progress and exiting cleanly.")
            break

        try:
            status, n_saved, err = process_star(
                tic_id=tic_id,
                out_dir=out_dir,
                seq_len=args.seq_len,
                window_size=args.window_size,
                stride=args.stride,
                gap_threshold=args.gap_threshold,
                stats=stats,
                logger=logger,
            )
        except KeyboardInterrupt:
            _sigint_handler(None, None)
            break
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"TIC {tic_id}: unhandled exception:\n{tb}")
            status, n_saved, err = ("error", 0, _one_line(f"{type(e).__name__}: {e}"))

        if status == "done":
            stats.stars_processed += 1
            consecutive_errors = 0
        elif status == "no_data":
            stats.stars_no_data += 1
            consecutive_errors = 0
        else:
            stats.stars_error += 1
            consecutive_errors += 1

        rec = {"tic_id": int(tic_id), "status": status,
               "n_segments_saved": int(n_saved), "error_msg": err}
        if int(tic_id) in progress_index:
            progress_records[progress_index[int(tic_id)]] = rec
        else:
            progress_index[int(tic_id)] = len(progress_records)
            progress_records.append(rec)

        if (
            args.max_consecutive_errors > 0
            and consecutive_errors >= args.max_consecutive_errors
        ):
            logger.error(
                f"Aborting: {consecutive_errors} consecutive errored stars "
                f"(>= --max-consecutive-errors={args.max_consecutive_errors}). "
                f"Likely a network outage — flushing progress and exiting so "
                f"--resume can pick up later without burning through the queue."
            )
            aborted_for_outage = True
            break

        if i % args.checkpoint_every == 0 or i == len(tic_ids):
            save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), progress_csv)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(tic_ids) - i
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            logger.info(
                f"Progress {i}/{len(tic_ids)}  "
                f"done={stats.stars_processed} no_data={stats.stars_no_data} "
                f"err={stats.stars_error}  "
                f"segs_saved={stats.segments_saved}  "
                f"rate={rate:.2f} stars/s  ETA={eta_min:.1f} min"
            )

    save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), progress_csv)

    size_gb = directory_size_gb(out_dir)
    total_class = stats.windows_class_a + stats.windows_class_b + stats.windows_class_c
    pct = lambda x: (100.0 * x / total_class) if total_class > 0 else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sequence Build Summary")
    logger.info("=" * 60)
    logger.info(f"Stars processed (status=done):    {stats.stars_processed}")
    logger.info(f"Stars with no SPOC data:          {stats.stars_no_data}")
    logger.info(f"Stars with errors:                {stats.stars_error}")
    logger.info(f"Total sectors examined:           {stats.sectors_examined}")
    logger.info(f"Total segments produced:          {stats.segments_total}")
    logger.info(f"Segments saved (>= seq_len):      {stats.segments_saved}")
    logger.info(f"Segments discarded (too short):   {stats.segments_too_short}")
    logger.info(f"Total candidate windows:          {total_class}")
    logger.info(f"  Class A (NaN-free, kept):       {stats.windows_class_a}  ({pct(stats.windows_class_a):.1f}%)")
    logger.info(f"  Class B (NaN runs all <= 10):   {stats.windows_class_b}  ({pct(stats.windows_class_b):.1f}%)")
    logger.info(f"  Class C (has NaN run > 10):     {stats.windows_class_c}  ({pct(stats.windows_class_c):.1f}%)")
    logger.info(f"Output directory size:            {size_gb:.2f} GB")
    logger.info("=" * 60)

    if _INTERRUPTED:
        return 130
    if aborted_for_outage:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
