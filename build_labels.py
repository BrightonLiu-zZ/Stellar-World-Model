"""build_labels.py — Stage 0c of the stellar world model pipeline.

For each TIC ID in processed/df_final.csv:
  1. Look up RA/Dec (and Gaia DR2 ID for traceability) from TIC v8.2 via
     astroquery.mast.Catalogs.query_criteria(catalog="TIC", ID=tic_id).
  2. APOGEE DR17 cone search at (ra, dec) within 1" via
     Vizier "III/286/catalog". If multiple rows, keep highest-SNR.
  3. Gaia DR3 GSP-Spec fallback (Recio-Blanco+2023) via Vizier "I/355/paramp",
     reading the '-S' suffixed columns (Teff-S, logg-S, [Fe/H]-S) with strict
     quality cut: first 13 chars of `Flags` (flags_gspspec) must all be '0'.
     Lower/upper CIs (b_*, B_*) are converted to symmetric err = (B-b)/2.
  4. LAMOST DR11 LRS fallback via Vizier "V/162/dr11sl" (DR8 substitute —
     DR8 is not mirrored on VizieR; DR11 is its strict superset). Cut: snrg
     >= 10. Tiebreak on highest snrg.
  5. If none match, skip the star (no row written).
  6. Save row to in-memory results; flush to CSV every --checkpoint-every stars.

Acceptance rule (Option 1, status quo): a row is accepted only if all three of
(teff, logg, feh) are present and unmasked. Stars where teff+logg are valid
but feh is missing fall through silently. The summary reports a what-if count
for the looser Option 3 (require teff+logg only; feh may be NaN) so the cost of
the strict rule is visible.

Output: labels/stellar_params.csv with columns
  tic_id, gaia_source_id, teff, teff_err, logg, logg_err, feh, feh_err, source
where source ∈ {"APOGEE", "GSP-Spec", "LAMOST"}.

Per-star checkpointing to labels/build_labels_progress.csv enables --resume
(skips apogee/gsp_spec/lamost/no_label rows; retries error rows).

Usage examples:
    python build_labels.py
    python build_labels.py --resume
    python build_labels.py --limit 5            # quick smoke test on 5 stars
"""
from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd

import requests.exceptions
import urllib3.exceptions

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier

# Vizier catalog identifiers — see CLAUDE.md "Labels Source" table.
APOGEE_CATALOG = "III/286/catalog"
# Gaia DR3 supplementary astrophysical parameters — GSP-Spec fields are the
# '-S' suffixed columns (Teff-S, logg-S, [Fe/H]-S, b_/B_ for lower/upper CI).
GSP_SPEC_CATALOG = "I/355/paramp"
# DR8 is not on VizieR. DR11 is a strict superset (same LASP pipeline, more
# observations) and serves as the in-pipeline substitute.
LAMOST_CATALOG = "V/162/dr11sl"

# GSP-Spec strict quality cut (Recio-Blanco+2023): the first 13 characters
# of `flags_gspspec` ('Flags' on VizieR) must all be '0'.
GSP_SPEC_FLAG_PREFIX_LEN = 13

# LAMOST quality cut: keep rows with snrg above this threshold.
LAMOST_SNRG_MIN = 10
# Reporting-only: how many of the kept rows fall below this stricter threshold,
# so we can see the cost of switching the production cut later.
LAMOST_SNRG_STRICT = 20

# Be polite to Vizier — sleep this long between catalog calls.
VIZIER_THROTTLE_S = 0.3

# Cone-search radius for all label catalogs.
CONE_RADIUS = 1 * u.arcsec

# MAST and Vizier sometimes drop large connections; bump the client timeout.
Catalogs.TIMEOUT = 300

# Transient network exceptions that warrant retry. astroquery -> requests
# -> urllib3 -> socket; any layer can surface a blip.
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
        description="Stage 0c: cross-match TICs to APOGEE DR17 → Gaia DR3 GSP-Spec → LAMOST DR11 labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-csv", default="processed/df_final.csv",
                    help="CSV with TIC IDs in column 'ID'")
    ap.add_argument("--out-csv", default="labels/stellar_params.csv",
                    help="Output CSV with cross-matched labels")
    ap.add_argument("--progress-csv", default="labels/build_labels_progress.csv",
                    help="Per-star checkpoint")
    ap.add_argument("--log-file", default="build_labels.log",
                    help="Log file path")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="Flush progress + output CSVs every N stars")
    ap.add_argument("--resume", action="store_true",
                    help="Skip TICs whose status is apogee/gsp_spec/lamost/no_label in progress CSV")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N stars (for testing)")
    ap.add_argument("--max-consecutive-errors", type=int, default=10,
                    help="Abort after this many consecutive errored stars (likely outage). "
                         "0 disables the safety valve.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_labels")
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
    apogee_matches: int = 0
    gsp_spec_matches: int = 0
    lamost_matches: int = 0
    no_label: int = 0
    errors: int = 0

    # What-if reporting (Option 3 — require teff+logg only, allow feh=NaN).
    # We still apply Option 1 (status quo: reject if any of teff/logg/feh
    # is missing). These counters track how many of the no_label stars
    # WOULD have been labeled under Option 3, broken down by which source
    # would have first matched in fallback order. Sum = Option 3 row gain.
    option3_gain_apogee: int = 0
    option3_gain_gsp_spec: int = 0
    option3_gain_lamost: int = 0

    # What-if reporting for LAMOST: how many of the rows accepted at
    # snrg >= LAMOST_SNRG_MIN have their best-row snrg in [MIN, STRICT)
    # — i.e., would be dropped if the cut were tightened to STRICT.
    lamost_kept_below_snr_strict: int = 0


# ---------------------------------------------------------------------------
# Network retry — transient errors only, exponential backoff
# ---------------------------------------------------------------------------

T = TypeVar("T")


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
# Astropy-table column access — case-insensitive, bracket-tolerant
# ---------------------------------------------------------------------------

def _find_col(table, *candidates: str, case_sensitive: bool = False) -> Optional[str]:
    """Case-insensitive column-name lookup on an astropy Table.

    Vizier mangles bracketed names (e.g. '[Fe/H]' → '__Fe_H_'); this helper also
    matches by stripping non-alphanumerics so 'feh' matches '__Fe_H_', 'Fe_H', etc.
    Returns the actual column name, or None if no candidate matches.

    Pass case_sensitive=True for columns that differ only in case — e.g. Gaia
    GSP-Spec has both `b_Teff-S` (lower CI) and `B_Teff-S` (upper CI) and the
    case-insensitive collapse would conflate them.
    """
    cols = list(table.colnames)
    if case_sensitive:
        for cand in candidates:
            if cand in cols:
                return cand
        return None
    cols_lower = {c.lower(): c for c in cols}
    cols_alnum = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in cols}
    for cand in candidates:
        real = cols_lower.get(cand.lower())
        if real is not None:
            return real
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        real = cols_alnum.get(key)
        if real is not None:
            return real
    return None


def _to_float(value: Any) -> Optional[float]:
    """Convert a possibly-masked Astropy scalar to a Python float, or None if missing."""
    try:
        if value is None:
            return None
        if hasattr(value, "mask") and bool(value.mask):
            return None
        v = float(value)
        if not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str:
    """Convert a possibly-masked Astropy scalar to a string, or '' if missing."""
    try:
        if value is None:
            return ""
        if hasattr(value, "mask") and bool(value.mask):
            return ""
        s = str(value).strip()
        if s.lower() in {"--", "nan", "none", "n/a"}:
            return ""
        return s
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# TIC v8.2 position lookup
# ---------------------------------------------------------------------------

def get_tic_position(tic_id: int, logger: logging.Logger) -> tuple[float, float, str]:
    """Return (ra_deg, dec_deg, gaia_dr2_id) for a TIC ID.

    Raises ValueError if the TIC row can't be found or has no usable RA/Dec.
    gaia_dr2_id is '' if absent in TIC.
    """
    result = _call_with_retry(
        lambda: Catalogs.query_criteria(catalog="TIC", ID=tic_id),
        label="TIC lookup",
        tic_id=tic_id,
        logger=logger,
    )
    if result is None or len(result) == 0:
        raise ValueError(f"TIC {tic_id} not found in TIC v8.2 catalog")

    ra_col = _find_col(result, "ra", "RAJ2000", "RA")
    dec_col = _find_col(result, "dec", "DEJ2000", "Dec")
    gaia_col = _find_col(result, "GAIA", "gaia", "Gaia")

    if ra_col is None or dec_col is None:
        raise ValueError(
            f"TIC {tic_id}: RA/Dec columns absent (cols={list(result.colnames)[:10]}...)"
        )

    ra = _to_float(result[ra_col][0])
    dec = _to_float(result[dec_col][0])
    if ra is None or dec is None:
        raise ValueError(f"TIC {tic_id}: RA/Dec masked or non-finite")

    gaia_dr2 = _to_str(result[gaia_col][0]) if gaia_col is not None else ""
    return ra, dec, gaia_dr2


# ---------------------------------------------------------------------------
# Vizier cone search
# ---------------------------------------------------------------------------

def _vizier_cone_search(
    catalog: str,
    ra: float,
    dec: float,
    tic_id: int,
    logger: logging.Logger,
):
    """Run a 1" cone search against `catalog`. Returns the first non-empty Table or None."""
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    v = Vizier(columns=["**"], catalog=catalog)
    v.ROW_LIMIT = -1

    def _go():
        return v.query_region(coord, radius=CONE_RADIUS)

    result = _call_with_retry(_go, label=f"Vizier {catalog}",
                              tic_id=tic_id, logger=logger)
    if result is None or len(result) == 0:
        return None
    # Vizier returns a TableList. Usually one table, but multi-table catalogs
    # (like GALAH) can have several. Pick the first non-empty one.
    for tbl in result:
        if tbl is not None and len(tbl) > 0:
            return tbl
    return None


def _highest_snr_row(table) -> int:
    """Return the index of the row with highest SNR, or 0 if no SNR column."""
    snr_col = _find_col(table, "SNR", "S_N", "snr_c2_iraf", "snr", "SN")
    if snr_col is None:
        return 0
    snr_vals = np.array(
        [_to_float(table[snr_col][i]) or -np.inf for i in range(len(table))]
    )
    return int(np.argmax(snr_vals))


def _closest_row(table) -> int:
    """Return the row index closest to the cone center.

    Vizier prefixes cone-search results with a `_r` column (arcsec). Falls back
    to row 0 when the column is absent (e.g. single-row result).
    """
    r_col = _find_col(table, "_r")
    if r_col is None:
        return 0
    rs = np.array(
        [_to_float(table[r_col][i]) if _to_float(table[r_col][i]) is not None else np.inf
         for i in range(len(table))]
    )
    return int(np.argmin(rs))


def _ci_err(row, lo_col: Optional[str], hi_col: Optional[str]) -> Optional[float]:
    """Symmetric error from lower/upper confidence bounds: (B - b) / 2.

    Returns None if either bound column is missing or its value is masked.
    """
    if lo_col is None or hi_col is None:
        return None
    lo = _to_float(row[lo_col])
    hi = _to_float(row[hi_col])
    if lo is None or hi is None:
        return None
    return (hi - lo) / 2.0


def _gsp_spec_flags_pass_strict(flags_str: str) -> bool:
    """Recio-Blanco+2023 strict cut on flags_gspspec.

    The flag is a 41-character string of single-digit codes. The first
    GSP_SPEC_FLAG_PREFIX_LEN (=13) chars cover the most important parameter
    quality indicators; require all of them to be '0' (best quality).
    """
    if not flags_str or len(flags_str) < GSP_SPEC_FLAG_PREFIX_LEN:
        return False
    return all(c == "0" for c in flags_str[:GSP_SPEC_FLAG_PREFIX_LEN])


# ---------------------------------------------------------------------------
# APOGEE DR17 query
# ---------------------------------------------------------------------------

def query_apogee(
    ra: float, dec: float, tic_id: int, logger: logging.Logger
) -> tuple[Optional[dict], bool]:
    """1" cone of APOGEE DR17. Returns (strict_row, has_partial_teff_logg).

    strict_row is the output dict only when all of (teff, logg, feh) are
    present and unmasked (Option 1). has_partial_teff_logg is True whenever
    teff and logg are both present on the chosen row, regardless of feh; it
    feeds the Option-3 what-if counter in the run summary.
    """
    table = _vizier_cone_search(APOGEE_CATALOG, ra, dec, tic_id, logger)
    if table is None or len(table) == 0:
        return None, False

    teff_col = _find_col(table, "Teff")
    e_teff_col = _find_col(table, "e_Teff")
    logg_col = _find_col(table, "logg")
    e_logg_col = _find_col(table, "e_logg")
    feh_col = _find_col(table, "[Fe/H]", "__Fe_H_", "Fe_H", "FeH", "feh")
    e_feh_col = _find_col(table, "e_[Fe/H]", "e__Fe_H_", "e_Fe_H", "e_FeH", "e_feh")

    if teff_col is None or logg_col is None:
        logger.warning(
            f"TIC {tic_id}: APOGEE row found but Teff/logg columns missing "
            f"(cols={list(table.colnames)[:15]}...)"
        )
        return None, False

    idx = _highest_snr_row(table)
    row = table[idx]

    teff = _to_float(row[teff_col])
    logg = _to_float(row[logg_col])
    feh = _to_float(row[feh_col]) if feh_col is not None else None

    has_partial = teff is not None and logg is not None
    if not has_partial:
        return None, False
    if feh is None:
        # Option 3 would accept; Option 1 rejects because feh is missing/masked.
        return None, True

    return {
        "teff": teff,
        "teff_err": _to_float(row[e_teff_col]) if e_teff_col else None,
        "logg": logg,
        "logg_err": _to_float(row[e_logg_col]) if e_logg_col else None,
        "feh": feh,
        "feh_err": _to_float(row[e_feh_col]) if e_feh_col else None,
        "source": "APOGEE",
    }, True


# ---------------------------------------------------------------------------
# Gaia DR3 GSP-Spec query
# ---------------------------------------------------------------------------

def query_gsp_spec(
    ra: float, dec: float, tic_id: int, logger: logging.Logger
) -> tuple[Optional[dict], bool]:
    """1" cone of Gaia DR3 GSP-Spec (I/355/paramp, '-S' suffixed columns).

    Strict quality cut: rows pass only if the first GSP_SPEC_FLAG_PREFIX_LEN
    characters of `Flags` (flags_gspspec) are all '0'. Errors derive from
    asymmetric lower/upper confidence bounds: err = (B - b) / 2.

    Returns (strict_row, has_partial_teff_logg) — see query_apogee.
    """
    table = _vizier_cone_search(GSP_SPEC_CATALOG, ra, dec, tic_id, logger)
    if table is None or len(table) == 0:
        return None, False

    teff_col = _find_col(table, "Teff-S", "TeffS", "Teff_S")
    teff_lo  = _find_col(table, "b_Teff-S", case_sensitive=True)
    teff_hi  = _find_col(table, "B_Teff-S", case_sensitive=True)
    logg_col = _find_col(table, "logg-S", "loggS", "logg_S")
    logg_lo  = _find_col(table, "b_logg-S", case_sensitive=True)
    logg_hi  = _find_col(table, "B_logg-S", case_sensitive=True)
    feh_col  = _find_col(table, "[Fe/H]-S", "__Fe_H__S", "Fe_H_S", "FeHS")
    feh_lo   = _find_col(table, "b_[Fe/H]-S", case_sensitive=True)
    feh_hi   = _find_col(table, "B_[Fe/H]-S", case_sensitive=True)
    flags_col = _find_col(table, "Flags")

    if teff_col is None or logg_col is None:
        logger.warning(
            f"TIC {tic_id}: GSP-Spec row found but Teff-S/logg-S columns missing "
            f"(cols={list(table.colnames)[:20]}...)"
        )
        return None, False

    if flags_col is None:
        # Without the quality string we cannot apply the strict cut and we
        # refuse to silently downgrade to "no filter" for a Tier-1 source.
        logger.warning(
            f"TIC {tic_id}: GSP-Spec Flags column absent — strict filter cannot be applied; rejecting"
        )
        return None, False

    # Strict flag filter — keep only rows whose first 13 flag chars are all '0'.
    keep = np.zeros(len(table), dtype=bool)
    for i in range(len(table)):
        if _gsp_spec_flags_pass_strict(_to_str(table[flags_col][i])):
            keep[i] = True
    if not keep.any():
        return None, False
    surviving = table[keep]

    # No SNR-equivalent in this table; tiebreak on closest cone-distance.
    idx = _closest_row(surviving)
    row = surviving[idx]

    teff = _to_float(row[teff_col])
    logg = _to_float(row[logg_col])
    feh = _to_float(row[feh_col]) if feh_col is not None else None

    has_partial = teff is not None and logg is not None
    if not has_partial:
        return None, False
    if feh is None:
        return None, True

    return {
        "teff": teff,
        "teff_err": _ci_err(row, teff_lo, teff_hi),
        "logg": logg,
        "logg_err": _ci_err(row, logg_lo, logg_hi),
        "feh": feh,
        "feh_err": _ci_err(row, feh_lo, feh_hi),
        "source": "GSP-Spec",
    }, True


# ---------------------------------------------------------------------------
# LAMOST DR11 LRS query  (DR8 substitute — DR8 not on VizieR)
# ---------------------------------------------------------------------------

def query_lamost(
    ra: float, dec: float, tic_id: int, logger: logging.Logger
) -> tuple[Optional[dict], bool, Optional[float]]:
    """1" cone of LAMOST DR11 LRS single-epoch parameters (V/162/dr11sl).

    Quality cut: snrg >= LAMOST_SNRG_MIN. Tiebreak on highest snrg.

    Returns (strict_row, has_partial_teff_logg, snrg_of_chosen_row). The third
    element lets the caller report how many accepted rows have snrg below the
    'strict' threshold (LAMOST_SNRG_STRICT) — a what-if counter for tightening
    the cut later.
    """
    table = _vizier_cone_search(LAMOST_CATALOG, ra, dec, tic_id, logger)
    if table is None or len(table) == 0:
        return None, False, None

    teff_col = _find_col(table, "Teff")
    e_teff_col = _find_col(table, "e_Teff")
    logg_col = _find_col(table, "logg")
    e_logg_col = _find_col(table, "e_logg")
    feh_col = _find_col(table, "[Fe/H]", "__Fe_H_", "Fe_H", "FeH", "feh")
    e_feh_col = _find_col(table, "e_[Fe/H]", "e__Fe_H_", "e_Fe_H", "e_FeH", "e_feh")
    snrg_col = _find_col(table, "snrg")

    if teff_col is None or logg_col is None:
        logger.warning(
            f"TIC {tic_id}: LAMOST row found but Teff/logg columns missing "
            f"(cols={list(table.colnames)[:15]}...)"
        )
        return None, False, None
    if snrg_col is None:
        logger.warning(
            f"TIC {tic_id}: LAMOST snrg column absent — cannot apply quality cut; rejecting"
        )
        return None, False, None

    # Apply snrg cut.
    keep = np.zeros(len(table), dtype=bool)
    for i in range(len(table)):
        s = _to_float(table[snrg_col][i])
        if s is not None and s >= LAMOST_SNRG_MIN:
            keep[i] = True
    if not keep.any():
        return None, False, None
    surviving = table[keep]

    # Pick highest-snrg row.
    snrg_vals = np.array(
        [_to_float(surviving[snrg_col][i]) if _to_float(surviving[snrg_col][i]) is not None else -np.inf
         for i in range(len(surviving))]
    )
    idx = int(np.argmax(snrg_vals))
    row = surviving[idx]
    snrg_best = float(snrg_vals[idx]) if np.isfinite(snrg_vals[idx]) else None

    teff = _to_float(row[teff_col])
    logg = _to_float(row[logg_col])
    feh = _to_float(row[feh_col]) if feh_col is not None else None

    has_partial = teff is not None and logg is not None
    if not has_partial:
        return None, False, snrg_best
    if feh is None:
        return None, True, snrg_best

    return {
        "teff": teff,
        "teff_err": _to_float(row[e_teff_col]) if e_teff_col else None,
        "logg": logg,
        "logg_err": _to_float(row[e_logg_col]) if e_logg_col else None,
        "feh": feh,
        "feh_err": _to_float(row[e_feh_col]) if e_feh_col else None,
        "source": "LAMOST",
    }, True, snrg_best


# ---------------------------------------------------------------------------
# Per-star processing
# ---------------------------------------------------------------------------

def process_star(
    tic_id: int,
    stats: Stats,
    logger: logging.Logger,
) -> tuple[str, Optional[dict], str]:
    """Resolve labels for one TIC. Returns (status, output_row_or_None, error_msg).

    status ∈ {'apogee', 'gsp_spec', 'lamost', 'no_label', 'error'}.

    Fallback order: APOGEE DR17 → Gaia DR3 GSP-Spec → LAMOST DR11. Acceptance
    is Option 1 (require teff+logg+feh). When all three sources strict-fail
    we additionally record whether any source had teff+logg present (Option 3
    what-if), and which one in fallback order would have first matched.
    """
    try:
        ra, dec, gaia_dr2 = get_tic_position(tic_id, logger)
    except Exception as e:
        logger.error(f"TIC {tic_id}: TIC lookup failed: {type(e).__name__}: {e}")
        return ("error", None, f"tic: {type(e).__name__}: {e}")

    # APOGEE DR17 (priority 1).
    time.sleep(VIZIER_THROTTLE_S)
    try:
        apogee_strict, apogee_partial = query_apogee(ra, dec, tic_id, logger)
    except Exception as e:
        logger.error(f"TIC {tic_id}: APOGEE query failed: {type(e).__name__}: {e}")
        return ("error", None, f"apogee: {type(e).__name__}: {e}")

    if apogee_strict is not None:
        stats.apogee_matches += 1
        row = {"tic_id": int(tic_id), "gaia_source_id": gaia_dr2, **apogee_strict}
        return ("apogee", row, "")

    # Gaia DR3 GSP-Spec (priority 2).
    time.sleep(VIZIER_THROTTLE_S)
    try:
        gsp_strict, gsp_partial = query_gsp_spec(ra, dec, tic_id, logger)
    except Exception as e:
        logger.error(f"TIC {tic_id}: GSP-Spec query failed: {type(e).__name__}: {e}")
        return ("error", None, f"gsp_spec: {type(e).__name__}: {e}")

    if gsp_strict is not None:
        stats.gsp_spec_matches += 1
        row = {"tic_id": int(tic_id), "gaia_source_id": gaia_dr2, **gsp_strict}
        return ("gsp_spec", row, "")

    # LAMOST DR11 (priority 3).
    time.sleep(VIZIER_THROTTLE_S)
    try:
        lam_strict, lam_partial, lam_snrg = query_lamost(ra, dec, tic_id, logger)
    except Exception as e:
        logger.error(f"TIC {tic_id}: LAMOST query failed: {type(e).__name__}: {e}")
        return ("error", None, f"lamost: {type(e).__name__}: {e}")

    if lam_strict is not None:
        stats.lamost_matches += 1
        if lam_snrg is not None and lam_snrg < LAMOST_SNRG_STRICT:
            stats.lamost_kept_below_snr_strict += 1
        row = {"tic_id": int(tic_id), "gaia_source_id": gaia_dr2, **lam_strict}
        return ("lamost", row, "")

    # All three sources strict-failed → no_label. Record Option-3 what-if:
    # the first source in fallback order whose row had teff+logg present.
    if apogee_partial:
        stats.option3_gain_apogee += 1
    elif gsp_partial:
        stats.option3_gain_gsp_spec += 1
    elif lam_partial:
        stats.option3_gain_lamost += 1

    stats.no_label += 1
    return ("no_label", None, "")


# ---------------------------------------------------------------------------
# Progress + output CSVs
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["tic_id", "status", "error_msg"]
OUTPUT_COLS = [
    "tic_id", "gaia_source_id",
    "teff", "teff_err", "logg", "logg_err", "feh", "feh_err",
    "source",
]


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
    """Read existing label rows so --resume preserves them across runs."""
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
# Main
# ---------------------------------------------------------------------------

_INTERRUPTED = False


def _sigint_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    progress_csv = Path(args.progress_csv)
    log_file = Path(args.log_file)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
    output_rows = load_existing_output(out_csv)
    output_index = {int(r["tic_id"]): i for i, r in enumerate(output_rows)}

    if args.resume and len(progress) > 0:
        skip_statuses = {"apogee", "gsp_spec", "lamost", "no_label"}
        skip = set(
            progress.loc[progress["status"].isin(skip_statuses), "tic_id"].astype(int)
        )
        before = len(tic_ids)
        tic_ids = [t for t in tic_ids if t not in skip]
        logger.info(f"Resume: skipping {before - len(tic_ids)} already-resolved stars; "
                    f"{len(tic_ids)} remaining.")

    if args.limit is not None:
        tic_ids = tic_ids[: args.limit]
        logger.info(f"--limit set: processing only first {len(tic_ids)} stars.")

    signal.signal(signal.SIGINT, _sigint_handler)

    stats = Stats()
    progress_records: list[dict] = progress.to_dict("records")
    progress_index = {int(r["tic_id"]): i for i, r in enumerate(progress_records)}

    logger.info(f"Starting build_labels: {len(tic_ids)} TICs to process")
    t0 = time.time()
    consecutive_errors = 0
    aborted_for_outage = False

    for i, tic_id in enumerate(tic_ids, start=1):
        if _INTERRUPTED:
            logger.warning("Ctrl+C received — flushing progress and exiting cleanly.")
            break

        try:
            status, row, err = process_star(tic_id=tic_id, stats=stats, logger=logger)
        except KeyboardInterrupt:
            _sigint_handler(None, None)
            break
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"TIC {tic_id}: unhandled exception:\n{tb}")
            status, row, err = ("error", None, f"{type(e).__name__}: {e}")

        if status == "error":
            stats.errors += 1
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        if row is not None:
            if int(tic_id) in output_index:
                output_rows[output_index[int(tic_id)]] = row
            else:
                output_index[int(tic_id)] = len(output_rows)
                output_rows.append(row)

        prec = {"tic_id": int(tic_id), "status": status, "error_msg": err}
        if int(tic_id) in progress_index:
            progress_records[progress_index[int(tic_id)]] = prec
        else:
            progress_index[int(tic_id)] = len(progress_records)
            progress_records.append(prec)

        if (
            args.max_consecutive_errors > 0
            and consecutive_errors >= args.max_consecutive_errors
        ):
            logger.error(
                f"Aborting: {consecutive_errors} consecutive errored stars "
                f"(>= --max-consecutive-errors={args.max_consecutive_errors}). "
                f"Likely a network outage — flushing progress and exiting so "
                f"--resume can pick up later."
            )
            aborted_for_outage = True
            break

        if i % args.checkpoint_every == 0 or i == len(tic_ids):
            save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), progress_csv)
            save_output(output_rows, out_csv)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(tic_ids) - i
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            logger.info(
                f"Progress {i}/{len(tic_ids)}  "
                f"apogee={stats.apogee_matches} gsp_spec={stats.gsp_spec_matches} "
                f"lamost={stats.lamost_matches} no_label={stats.no_label} "
                f"err={stats.errors}  rate={rate:.2f} stars/s  ETA={eta_min:.1f} min"
            )

    save_progress(pd.DataFrame(progress_records, columns=PROGRESS_COLS), progress_csv)
    save_output(output_rows, out_csv)

    total_input = len(df_in)
    labeled = stats.apogee_matches + stats.gsp_spec_matches + stats.lamost_matches
    opt3_gain = (stats.option3_gain_apogee
                 + stats.option3_gain_gsp_spec
                 + stats.option3_gain_lamost)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Label Build Summary")
    logger.info("=" * 60)
    logger.info(f"Input TICs (in df_final.csv):     {total_input}")
    logger.info(f"TICs processed this run:          {len(tic_ids)}")
    logger.info(f"  APOGEE   matches:               {stats.apogee_matches}")
    logger.info(f"  GSP-Spec matches (strict flags):{stats.gsp_spec_matches}")
    logger.info(f"  LAMOST   matches (snrg>={LAMOST_SNRG_MIN}):    {stats.lamost_matches}")
    logger.info(f"  No label (all sources missed):  {stats.no_label}")
    logger.info(f"  Errors:                         {stats.errors}")
    logger.info(f"Output rows (cumulative on disk): {len(output_rows)}")
    logger.info(f"Output file: {out_csv}")
    logger.info("")
    logger.info("--- What-if comparisons (informational; not applied) ---")
    logger.info("Option 3 (require teff+logg only; allow feh=NaN) would gain:")
    logger.info(f"  +{stats.option3_gain_apogee} APOGEE   rows (teff+logg ok, feh missing)")
    logger.info(f"  +{stats.option3_gain_gsp_spec} GSP-Spec rows (teff+logg ok, feh missing)")
    logger.info(f"  +{stats.option3_gain_lamost} LAMOST   rows (teff+logg ok, feh missing)")
    logger.info(f"  => +{opt3_gain} labeled rows total ({labeled} -> {labeled + opt3_gain})")
    logger.info(f"Stricter LAMOST cut (snrg>={LAMOST_SNRG_STRICT}) would lose:")
    logger.info(f"  -{stats.lamost_kept_below_snr_strict} LAMOST rows "
                f"(best-row snrg currently in [{LAMOST_SNRG_MIN}, {LAMOST_SNRG_STRICT}))")
    logger.info("=" * 60)

    if _INTERRUPTED:
        return 130
    if aborted_for_outage:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
