"""compute_flare_window_labels.py — Stage 0d-b: window-level flare labels.

Catalog column discovery (run once on 2026-05-11):
  flatwrm2 Table 3 (Table3_flare_catalog.csv):
    TIC column     : 'TIC'
    peak_time      : TBJD (float)
    timescale      : minutes (float) — half-duration derived as timescale/2 minutes

  Flare interval derived as:
    flare_begin = peak_time - timescale / (2 * 1440)  [days]
    flare_end   = peak_time + timescale / (2 * 1440)  [days]

  Overlap condition (standard interval):
    flare_begin < window_t_end AND flare_end > window_t_start

Input .npz files must contain a 'times' key (shape [N, 1024], TBJD float32).
Files without 'times' are skipped with a warning (re-run build_sequences.py first).

Output: labels/flare_window_labels.csv
Columns: tic_id, sector, seg_idx, run_idx, window_idx, t_start, t_end, flare_in_window

Usage:
    python compute_flare_window_labels.py
    python compute_flare_window_labels.py --resume
    python compute_flare_window_labels.py --limit 10    # process 10 npz files
"""
from __future__ import annotations

import argparse
import glob
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk up from CWD until CLAUDE.md is found."""
    p = Path.cwd()
    for _ in range(10):
        if (p / "CLAUDE.md").exists():
            return p
        p = p.parent
    raise FileNotFoundError("CLAUDE.md not found — cannot determine project root")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 0d-b: per-window flare overlap labels from flatwrm2 Table 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sequences-dir", default=None,
                    help="Directory of .npz files (default: processed/sequences)")
    ap.add_argument("--out-csv", default=None,
                    help="Output CSV path (default: labels/flare_window_labels.csv)")
    ap.add_argument("--progress-csv", default=None,
                    help="Per-file checkpoint CSV (default: labels/compute_flare_window_labels_progress.csv)")
    ap.add_argument("--flare-catalog", default=None,
                    help="Path to flatwrm2 Table 3 CSV (default: data/Table3_flare_catalog.csv)")
    ap.add_argument("--log-file", default=None,
                    help="Log file path (default: compute_flare_window_labels.log)")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="Flush progress CSV every N npz files")
    ap.add_argument("--resume", action="store_true",
                    help="Skip npz files already marked 'done' in progress CSV; retry 'error'")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N npz files (for smoke testing)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("compute_flare_window_labels")
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
# Flare catalog loading
# ---------------------------------------------------------------------------

def load_flare_index(path: Path, logger: logging.Logger) -> dict[int, list[tuple[float, float]]]:
    """Load flatwrm2 Table 3. Returns {tic_id: [(begin_tbjd, end_tbjd), ...]}."""
    logger.info(f"Loading flatwrm2 flare catalog from {path}")
    df = pd.read_csv(path)
    logger.info(f"  flatwrm2 rows: {len(df)}")

    tic_col = "TIC"
    peak_col = "peak_time"
    scale_col = "timescale"

    for col in (tic_col, peak_col, scale_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in flatwrm2. Found: {df.columns.tolist()}")

    # timescale is in minutes; convert half-duration to days
    flare_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for _, row in df.iterrows():
        tic_id = int(row[tic_col])
        peak = float(row[peak_col])
        half_dur_days = float(row[scale_col]) / (2.0 * 1440.0)
        flare_index[tic_id].append((peak - half_dur_days, peak + half_dur_days))

    logger.info(f"  Unique TIC IDs with flares: {len(flare_index)}")
    return dict(flare_index)


# ---------------------------------------------------------------------------
# Overlap check
# ---------------------------------------------------------------------------

def has_flare_overlap(
    t_start: float,
    t_end: float,
    flare_intervals: list[tuple[float, float]],
) -> bool:
    """True if any flare interval overlaps [t_start, t_end]."""
    for fb, fe in flare_intervals:
        if fb < t_end and fe > t_start:
            return True
    return False


# ---------------------------------------------------------------------------
# Progress CSV
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["npz_file", "status", "error_msg"]
OUTPUT_COLS = ["tic_id", "sector", "seg_idx", "run_idx", "window_idx",
               "t_start", "t_end", "flare_in_window"]


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


def flush_rows_to_csv(rows: list[dict], out_csv: Path, write_header: bool) -> None:
    """Append rows to the output CSV. Writes header only on first flush."""
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df.to_csv(out_csv, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_npz(
    npz_path: Path,
    flare_index: dict[int, list[tuple[float, float]]],
    logger: logging.Logger,
) -> tuple[str, list[dict], str]:
    """Process one .npz file. Returns (status, rows, error_msg)."""
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return ("error", [], f"load failed: {type(e).__name__}: {e}")

    if "times" not in data:
        logger.warning(f"  {npz_path.name}: missing 'times' key — skipping (re-run build_sequences.py first)")
        return ("error", [], "missing 'times' key in npz")

    try:
        tic_id = int(data["tic_id"])
        sector = int(data["sector"])
        seg_idx = int(data["seg_idx"])
        run_idx = int(data["run_idx"]) if "run_idx" in data else 0
        times = data["times"]   # [N, 1024] float32, TBJD
        n_windows = times.shape[0]
    except Exception as e:
        return ("error", [], f"metadata extraction failed: {type(e).__name__}: {e}")

    flare_intervals = flare_index.get(tic_id, [])
    rows: list[dict] = []

    for window_idx in range(n_windows):
        t_start = float(times[window_idx, 0])
        t_end = float(times[window_idx, -1])
        flare_in = 1 if (flare_intervals and has_flare_overlap(t_start, t_end, flare_intervals)) else 0
        rows.append({
            "tic_id": tic_id,
            "sector": sector,
            "seg_idx": seg_idx,
            "run_idx": run_idx,
            "window_idx": window_idx,
            "t_start": t_start,
            "t_end": t_end,
            "flare_in_window": flare_in,
        })

    return ("done", rows, "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_INTERRUPTED = False


def _sigint_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True


def main() -> int:
    args = parse_args()

    root = find_project_root()

    sequences_dir = Path(args.sequences_dir) if args.sequences_dir else root / "processed" / "sequences"
    out_csv       = Path(args.out_csv)       if args.out_csv       else root / "labels" / "flare_window_labels.csv"
    prog_csv      = Path(args.progress_csv)  if args.progress_csv  else root / "labels" / "compute_flare_window_labels_progress.csv"
    flare_path    = Path(args.flare_catalog) if args.flare_catalog else root / "data" / "Table3_flare_catalog.csv"
    log_file      = Path(args.log_file)      if args.log_file      else root / "compute_flare_window_labels.log"

    logger = setup_logging(log_file)

    if not sequences_dir.exists():
        logger.error(f"Sequences directory not found: {sequences_dir}")
        return 1
    if not flare_path.exists():
        logger.error(f"flatwrm2 catalog not found: {flare_path}")
        return 1

    try:
        flare_index = load_flare_index(flare_path, logger)
    except Exception as e:
        logger.error(f"Failed to load flare catalog: {e}")
        return 1

    npz_files = sorted(glob.glob(str(sequences_dir / "*.npz")))
    logger.info(f"Found {len(npz_files)} .npz files in {sequences_dir}")

    # Resume logic
    progress = load_progress(prog_csv)
    if args.resume and len(progress) > 0:
        skip = set(progress.loc[progress["status"] == "done", "npz_file"])
        before = len(npz_files)
        npz_files = [f for f in npz_files if Path(f).name not in skip]
        logger.info(f"Resume: skipping {before - len(npz_files)} done files; {len(npz_files)} remaining.")

    if args.limit is not None:
        npz_files = npz_files[: args.limit]
        logger.info(f"--limit {args.limit}: processing {len(npz_files)} files.")

    signal.signal(signal.SIGINT, _sigint_handler)

    progress_records: list[dict] = progress.to_dict("records")
    progress_index = {r["npz_file"]: i for i, r in enumerate(progress_records)}

    # Truncate output CSV on fresh run (not resume), write header once
    if not args.resume:
        if out_csv.exists():
            out_csv.unlink()
    write_header = not out_csv.exists()

    n_done = 0
    n_error = 0
    n_skipped = 0
    total_windows = 0
    total_flare_windows = 0
    pending_rows: list[dict] = []
    t0 = time.time()

    logger.info(f"Starting: {len(npz_files)} .npz files to process")

    for i, npz_path_str in enumerate(npz_files, start=1):
        if _INTERRUPTED:
            logger.warning("Ctrl+C received — flushing progress and exiting cleanly.")
            break

        npz_path = Path(npz_path_str)
        status, rows, err = process_npz(npz_path, flare_index, logger)

        if status == "done":
            n_done += 1
            total_windows += len(rows)
            total_flare_windows += sum(r["flare_in_window"] for r in rows)
            pending_rows.extend(rows)
        else:
            n_error += 1
            logger.error(f"{npz_path.name}: {err}")

        prec = {"npz_file": npz_path.name, "status": status, "error_msg": err}
        if npz_path.name in progress_index:
            progress_records[progress_index[npz_path.name]] = prec
        else:
            progress_index[npz_path.name] = len(progress_records)
            progress_records.append(prec)

        # Flush output rows in chunks to avoid holding all in memory
        if len(pending_rows) >= 10_000:
            flush_rows_to_csv(pending_rows, out_csv, write_header)
            write_header = False
            pending_rows = []

        if i % args.checkpoint_every == 0 or i == len(npz_files):
            if pending_rows:
                flush_rows_to_csv(pending_rows, out_csv, write_header)
                write_header = False
                pending_rows = []
            save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), prog_csv)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(npz_files) - i
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            logger.info(
                f"Progress {i}/{len(npz_files)}  done={n_done} err={n_error}  "
                f"windows={total_windows} flare_windows={total_flare_windows}  "
                f"rate={rate:.1f} files/s  ETA={eta_min:.1f} min"
            )

    # Final flush
    if pending_rows:
        flush_rows_to_csv(pending_rows, out_csv, write_header)
    save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), prog_csv)

    flare_pct = 100.0 * total_flare_windows / total_windows if total_windows > 0 else 0.0
    logger.info("")
    logger.info("=" * 60)
    logger.info("Flare Window Label Summary")
    logger.info("=" * 60)
    logger.info(f"Total .npz files processed:   {n_done}")
    logger.info(f"Files with errors/skipped:    {n_error}")
    logger.info(f"Total windows labeled:        {total_windows}")
    logger.info(f"Windows with flare_in_window: {total_flare_windows}  ({flare_pct:.2f}%)")
    logger.info(f"Output: {out_csv}")
    logger.info("=" * 60)

    if _INTERRUPTED:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
