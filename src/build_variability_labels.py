"""build_variability_labels.py — Stage 0d of the stellar world model pipeline.

Catalog column discovery (run once on 2026-05-11):
  TARS (tars_table_2.feather):
    TIC ID column  : 'TICID'
    Period column  : 'adopted_period'
  flatwrm2 Table 3 (Table3_flare_catalog.csv):
    TIC column     : 'TIC'
    (no begin/end time columns — only peak_time and timescale)

For each TIC ID in processed/df_final.csv:
  1. TARS lookup  → rotation=1, rotation_period=<value> if found, else 0/NaN
  2. flatwrm2 lookup → flare_ever=1 if TIC has any flare event, else 0
  3. TOI lookup   → transit=1, toi_id=<string> if found, else 0/NaN

Output: labels/variability_labels_star.csv
Columns: tic_id, rotation, rotation_period, flare_ever, transit, toi_id

Usage:
    python build_variability_labels.py
    python build_variability_labels.py --resume
    python build_variability_labels.py --limit 5        # smoke test
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests.exceptions

# Tenacity for TOI network retry
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

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
        description="Stage 0d: multi-label variability classification labels (rotation/flare/transit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-csv", default=None,
                    help="CSV with TIC IDs in column 'ID' (default: processed/df_final.csv)")
    ap.add_argument("--out-csv", default=None,
                    help="Output CSV path (default: labels/variability_labels_star.csv)")
    ap.add_argument("--progress-csv", default=None,
                    help="Per-star checkpoint CSV (default: labels/build_variability_labels_progress.csv)")
    ap.add_argument("--tars-catalog", default=None,
                    help="Path to tars_table_2.feather (default: data/tars_table_2.feather)")
    ap.add_argument("--flare-catalog", default=None,
                    help="Path to flatwrm2 Table 3 CSV (default: data/Table3_flare_catalog.csv)")
    ap.add_argument("--log-file", default=None,
                    help="Log file path (default: build_variability_labels.log)")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Flush progress + output CSVs every N stars")
    ap.add_argument("--resume", action="store_true",
                    help="Skip TIC IDs already marked 'done' in progress CSV; retry 'error' rows")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N stars (for smoke testing)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_variability_labels")
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
# TOI fetch with tenacity retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                   requests.exceptions.Timeout,
                                   requests.exceptions.ChunkedEncodingError,
                                   ConnectionError,
                                   TimeoutError,
                                   Exception)),
    reraise=False,
)
def _fetch_toi_attempt() -> pd.DataFrame:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    toi = NasaExoplanetArchive.query_criteria(
        table="toi",
        select="tid,toi,tfopwg_disp",
    ).to_pandas()
    toi = toi[toi["tfopwg_disp"] != "FP"].copy()
    return toi


def fetch_toi_table(logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Fetch TOI table with 3-attempt tenacity retry. Returns None on failure."""
    try:
        toi = _fetch_toi_attempt()
        logger.info(f"TOI table fetched: {len(toi)} rows after FP filter")
        return toi
    except RetryError as e:
        logger.warning(f"TOI fetch failed after 3 retries: {e}. transit labels will be NaN.")
        return None
    except Exception as e:
        logger.warning(f"TOI fetch failed: {type(e).__name__}: {e}. transit labels will be NaN.")
        return None


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_tars(path: Path, logger: logging.Logger) -> tuple[set[int], dict[int, float]]:
    """Load TARS feather. Returns (tic_set, {tic_id: period})."""
    logger.info(f"Loading TARS catalog from {path}")
    df = pd.read_feather(path)
    logger.info(f"  TARS rows: {len(df)}")
    # TICID column confirmed from inspection
    tic_col = "TICID"
    period_col = "adopted_period"
    if tic_col not in df.columns:
        raise KeyError(f"Expected column '{tic_col}' not found in TARS. Found: {df.columns.tolist()}")
    if period_col not in df.columns:
        raise KeyError(f"Expected column '{period_col}' not found in TARS. Found: {df.columns.tolist()}")
    tars_dict: dict[int, float] = dict(
        zip(df[tic_col].astype(int), df[period_col].astype(float))
    )
    logger.info(f"  TARS unique TIC IDs: {len(tars_dict)}")
    return set(tars_dict.keys()), tars_dict


def load_flares(path: Path, logger: logging.Logger) -> set[int]:
    """Load flatwrm2 Table 3. Returns set of TIC IDs that have any flare."""
    logger.info(f"Loading flatwrm2 flare catalog from {path}")
    df = pd.read_csv(path)
    logger.info(f"  flatwrm2 rows: {len(df)}")
    # TIC column confirmed from inspection
    tic_col = "TIC"
    if tic_col not in df.columns:
        raise KeyError(f"Expected column '{tic_col}' not found in flatwrm2. Found: {df.columns.tolist()}")
    flare_tic_set: set[int] = set(df[tic_col].astype(int).unique())
    logger.info(f"  flatwrm2 unique TIC IDs with flares: {len(flare_tic_set)}")
    return flare_tic_set


# ---------------------------------------------------------------------------
# Progress CSV
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["tic_id", "status", "error_msg"]
OUTPUT_COLS = ["tic_id", "rotation", "rotation_period", "flare_ever", "transit", "toi_id"]


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


def load_existing_output(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return df.to_dict("records")


def save_output(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLS) if rows else pd.DataFrame(columns=OUTPUT_COLS)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Per-star lookup (pure in-memory — no network calls)
# ---------------------------------------------------------------------------

def process_star(
    tic_id: int,
    tars_dict: dict[int, float],
    flare_tic_set: set[int],
    toi_tic_dict: Optional[dict[int, str]],
) -> tuple[str, dict, str]:
    """Look up one TIC in all three catalogs. Returns (status, row_dict, error_msg)."""
    try:
        # Rotation
        if tic_id in tars_dict:
            rotation = 1
            rotation_period = tars_dict[tic_id]
            if not np.isfinite(rotation_period):
                rotation_period = float("nan")
        else:
            rotation = 0
            rotation_period = float("nan")

        # Flare (star-level, supplementary)
        flare_ever = 1 if tic_id in flare_tic_set else 0

        # Transit
        if toi_tic_dict is None:
            transit = float("nan")
            toi_id = float("nan")
        elif tic_id in toi_tic_dict:
            transit = 1
            toi_id = toi_tic_dict[tic_id]
        else:
            transit = 0
            toi_id = float("nan")

        row = {
            "tic_id": int(tic_id),
            "rotation": rotation,
            "rotation_period": rotation_period,
            "flare_ever": flare_ever,
            "transit": transit,
            "toi_id": toi_id,
        }
        return ("done", row, "")
    except Exception as e:
        return ("error", {}, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict], total_input: int, n_processed: int, logger: logging.Logger) -> None:
    if not rows:
        logger.info("No output rows to summarize.")
        return

    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df["rotation"] = pd.to_numeric(df["rotation"], errors="coerce").fillna(0).astype(int)
    df["flare_ever"] = pd.to_numeric(df["flare_ever"], errors="coerce").fillna(0).astype(int)
    df["transit"] = pd.to_numeric(df["transit"], errors="coerce")

    r = df["rotation"] == 1
    f = df["flare_ever"] == 1
    t = df["transit"] == 1

    quiet = (~r) & (~f) & (~t)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Variability Label Build Summary")
    logger.info("=" * 60)
    logger.info(f"Input TICs (in df_final.csv):      {total_input}")
    logger.info(f"TICs processed this run:           {n_processed}")
    logger.info(f"Output rows (cumulative on disk):  {len(rows)}")
    logger.info("")
    logger.info("Per-label counts (in output rows):")
    logger.info(f"  rotation=1:    {r.sum()}")
    logger.info(f"  flare_ever=1:  {f.sum()}")
    logger.info(f"  transit=1:     {t.sum()}")
    logger.info("")
    logger.info("Per-combination counts:")
    logger.info(f"  rotation only:         {(r & ~f & ~t).sum()}")
    logger.info(f"  flare only:            {(~r & f & ~t).sum()}")
    logger.info(f"  transit only:          {(~r & ~f & t).sum()}")
    logger.info(f"  rotation + flare:      {(r & f & ~t).sum()}")
    logger.info(f"  rotation + transit:    {(r & ~f & t).sum()}")
    logger.info(f"  flare + transit:       {(~r & f & t).sum()}")
    logger.info(f"  all three:             {(r & f & t).sum()}")
    logger.info(f"  quiet (none):          {quiet.sum()}")
    logger.info("=" * 60)


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

    input_csv  = Path(args.input_csv)  if args.input_csv  else root / "processed" / "df_final.csv"
    out_csv    = Path(args.out_csv)    if args.out_csv    else root / "labels" / "variability_labels_star.csv"
    prog_csv   = Path(args.progress_csv) if args.progress_csv else root / "labels" / "build_variability_labels_progress.csv"
    tars_path  = Path(args.tars_catalog) if args.tars_catalog else root / "data" / "tars_table_2.feather"
    flare_path = Path(args.flare_catalog) if args.flare_catalog else root / "data" / "Table3_flare_catalog.csv"
    log_file   = Path(args.log_file)  if args.log_file  else root / "build_variability_labels.log"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    # Validate inputs
    if not input_csv.exists():
        logger.error(f"Input CSV not found: {input_csv}")
        return 1
    if not tars_path.exists():
        logger.error(f"TARS catalog not found: {tars_path}")
        logger.error("Download from Zenodo 10.5281/zenodo.18342591 and place at data/tars_table_2.feather")
        return 1
    if not flare_path.exists():
        logger.error(f"flatwrm2 catalog not found: {flare_path}")
        logger.error("Download Table3_flare_catalog.csv from Zenodo 10.5281/zenodo.14179313 and place in data/")
        return 1

    # Load all catalogs into memory
    try:
        tars_tic_set, tars_dict = load_tars(tars_path, logger)
    except Exception as e:
        logger.error(f"Failed to load TARS: {e}")
        return 1

    try:
        flare_tic_set = load_flares(flare_path, logger)
    except Exception as e:
        logger.error(f"Failed to load flatwrm2: {e}")
        return 1

    logger.info("Fetching NASA Exoplanet Archive TOI table...")
    toi_df = fetch_toi_table(logger)
    if toi_df is not None:
        # Build O(1) lookup: tid → toi string (take the first TOI per TIC if multiple)
        toi_tic_dict: Optional[dict[int, str]] = {}
        for _, row in toi_df.iterrows():
            tid = int(row["tid"])
            if tid not in toi_tic_dict:
                toi_tic_dict[tid] = str(row["toi"])
        logger.info(f"TOI lookup table: {len(toi_tic_dict)} unique TIC IDs")
    else:
        toi_tic_dict = None

    # Load star list
    df_in = pd.read_csv(input_csv)
    if "ID" not in df_in.columns:
        logger.error(f"Input CSV missing 'ID' column. Found: {list(df_in.columns)}")
        return 1
    tic_ids = df_in["ID"].astype(int).tolist()
    total_input = len(tic_ids)

    # Resume logic
    progress = load_progress(prog_csv)
    output_rows = load_existing_output(out_csv)
    output_index = {int(r["tic_id"]): i for i, r in enumerate(output_rows)}

    if args.resume and len(progress) > 0:
        skip = set(progress.loc[progress["status"] == "done", "tic_id"].astype(int))
        before = len(tic_ids)
        tic_ids = [t for t in tic_ids if t not in skip]
        logger.info(f"Resume: skipping {before - len(tic_ids)} done stars; {len(tic_ids)} remaining.")

    if args.limit is not None:
        tic_ids = tic_ids[: args.limit]
        logger.info(f"--limit {args.limit}: processing {len(tic_ids)} stars.")

    signal.signal(signal.SIGINT, _sigint_handler)

    progress_records: list[dict] = progress.to_dict("records")
    progress_index = {int(r["tic_id"]): i for i, r in enumerate(progress_records)}

    n_done = 0
    n_error = 0
    t0 = time.time()

    logger.info(f"Starting: {len(tic_ids)} TICs to process")

    for i, tic_id in enumerate(tic_ids, start=1):
        if _INTERRUPTED:
            logger.warning("Ctrl+C received — flushing progress and exiting cleanly.")
            break

        status, row, err = process_star(tic_id, tars_dict, flare_tic_set, toi_tic_dict)

        if status == "done":
            n_done += 1
            if int(tic_id) in output_index:
                output_rows[output_index[int(tic_id)]] = row
            else:
                output_index[int(tic_id)] = len(output_rows)
                output_rows.append(row)
        else:
            n_error += 1
            logger.error(f"TIC {tic_id}: {err}")

        prec = {"tic_id": int(tic_id), "status": status, "error_msg": err}
        if int(tic_id) in progress_index:
            progress_records[progress_index[int(tic_id)]] = prec
        else:
            progress_index[int(tic_id)] = len(progress_records)
            progress_records.append(prec)

        if i % args.checkpoint_every == 0 or i == len(tic_ids):
            save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), prog_csv)
            save_output(output_rows, out_csv)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(tic_ids) - i
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            logger.info(
                f"Progress {i}/{len(tic_ids)}  done={n_done} err={n_error}  "
                f"rate={rate:.0f} stars/s  ETA={eta_min:.1f} min"
            )

    save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), prog_csv)
    save_output(output_rows, out_csv)

    print_summary(output_rows, total_input, len(tic_ids), logger)

    if _INTERRUPTED:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
