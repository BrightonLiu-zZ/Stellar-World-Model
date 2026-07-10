"""build_variability_labels.py — Stage 0d of the stellar world model pipeline.

Cross-matches TIC IDs against five catalog families to produce multi-label
variability annotations for v1 linear-probe evaluation:

  1. TARS          → rotation (v1-supplementary)
  2. flatwrm2      → flare_ever (excluded from v1 eval, ADR-0001)
  3. NASA TOI      → transit (v1 primary)
  4. Villanova     → eb (v1 primary); single source, eb strictly 0/1 (ADR-0007)
  5. Gao+2025      → pulsating (v1 primary)

Output: labels/variability_labels_star.csv
Columns: tic_id, rotation, rotation_period, flare_ever,
         transit, toi_id, transit_disposition,
         eb, eb_period,
         pulsating, pulsating_period, pulsating_subtype

Usage:
    python build_variability_labels.py
    python build_variability_labels.py --resume
    python build_variability_labels.py --limit 5        # smoke test
"""
from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests.exceptions

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCEPTED_TRANSIT_DISPOSITIONS = {"CP", "KP", "PC", "APC"}
TRANSIT_DISPOSITION_PRIORITY = {"CP": 0, "KP": 1, "PC": 2, "APC": 3}

GAO_PULSATING_SUBTYPES = {
    "DSCT", "HADS",
    "GDOR",
    "CEP", "DCEP", "CEPHEIDS", "T2CEP",
    "RRAB", "RRC", "RRL", "RRCD",
    "SXPHE",
    "BCEP", "SPB",
    "ROAP",
}

OUTPUT_COLS = [
    "tic_id",
    "rotation", "rotation_period",
    "flare_ever",
    "transit", "toi_id", "transit_disposition",
    "eb", "eb_period",
    "pulsating", "pulsating_period", "pulsating_subtype",
]

PROGRESS_COLS = ["tic_id", "status", "error_msg"]

_NPZ_TIC_RE = re.compile(r"TIC(\d{10})")

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
        description=(
            "Stage 0d: multi-label variability labels "
            "(rotation/flare/transit/EB/pulsating)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-csv", default=None,
                    help="CSV with TIC IDs in column 'ID' or 'tic_id' "
                         "(default: processed/spoc_sector_map.csv)")
    ap.add_argument("--out-csv", default=None,
                    help="Output CSV path "
                         "(default: labels/variability_labels_star.csv)")
    ap.add_argument("--progress-csv", default=None,
                    help="Per-star checkpoint CSV "
                         "(default: labels/build_variability_labels_progress.csv)")
    ap.add_argument("--tars-catalog", default=None,
                    help="Path to tars_table_2.feather "
                         "(default: data/tars_table_2.feather)")
    ap.add_argument("--flare-catalog", default=None,
                    help="Path to flatwrm2 Table 3 CSV "
                         "(default: data/Table3_flare_catalog.csv)")
    ap.add_argument("--log-file", default=None,
                    help="Log file path "
                         "(default: build_variability_labels.log)")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Flush progress + output CSVs every N stars")
    ap.add_argument("--resume", action="store_true",
                    help="Skip TIC IDs already marked 'done' in progress CSV; "
                         "retry 'error' rows")
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
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Retry decorator (shared by all network catalog loaders)
# ---------------------------------------------------------------------------

_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionError,
    TimeoutError,
)


def _make_retry(attempts: int = 3):
    """Return a tenacity retry decorator for network catalog fetches."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Catalog loaders — local disk (TARS, flatwrm2)
# ---------------------------------------------------------------------------

def load_tars(path: Path, logger: logging.Logger) -> tuple[set[int], dict[int, float]]:
    """Load TARS feather. Returns (tic_set, {tic_id: period})."""
    logger.info(f"Loading TARS catalog from {path}")
    df = pd.read_feather(path)
    logger.info(f"  TARS rows: {len(df)}")
    tic_col, period_col = "TICID", "adopted_period"
    if tic_col not in df.columns:
        raise KeyError(f"Expected '{tic_col}' not in TARS. Found: {df.columns.tolist()}")
    if period_col not in df.columns:
        raise KeyError(f"Expected '{period_col}' not in TARS. Found: {df.columns.tolist()}")
    tars_dict: dict[int, float] = dict(
        zip(df[tic_col].astype(int), df[period_col].astype(float))
    )
    logger.info(f"  TARS unique TIC IDs: {len(tars_dict)}")
    return set(tars_dict.keys()), tars_dict


def load_flares(path: Path, logger: logging.Logger) -> set[int]:
    """Load flatwrm2 Table 3. Returns set of TIC IDs with any flare."""
    logger.info(f"Loading flatwrm2 flare catalog from {path}")
    df = pd.read_csv(path)
    logger.info(f"  flatwrm2 rows: {len(df)}")
    tic_col = "TIC"
    if tic_col not in df.columns:
        raise KeyError(f"Expected '{tic_col}' not in flatwrm2. Found: {df.columns.tolist()}")
    flare_tic_set: set[int] = set(df[tic_col].astype(int).unique())
    logger.info(f"  flatwrm2 unique TIC IDs with flares: {len(flare_tic_set)}")
    return flare_tic_set


# ---------------------------------------------------------------------------
# Catalog loaders — NASA TOI (network, essential)
# ---------------------------------------------------------------------------

@_make_retry()
def _fetch_toi_raw() -> pd.DataFrame:
    """Fetch full TOI table from NASA Exoplanet Archive (retried)."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    return NasaExoplanetArchive.query_criteria(
        table="toi",
        select="tid,toi,tfopwg_disp",
    ).to_pandas()


def fetch_toi_table(
    logger: logging.Logger,
) -> dict[int, tuple[str, str]]:
    """Fetch TOI table, apply whitelist, build best-disposition lookup.

    Returns {tic_id: (toi_id_str, disposition_str)}.
    Raises on failure (transit is v1 primary — fail-loud).
    """
    raw = _fetch_toi_raw()
    logger.info(f"TOI table fetched: {len(raw)} total rows")

    # Per-disposition counts before filtering
    disp_counts = raw["tfopwg_disp"].value_counts(dropna=False)
    for disp, cnt in sorted(disp_counts.items(), key=lambda x: str(x[0])):
        logger.info(f"  tfopwg_disp={disp!r}: {cnt}")

    filtered = raw[raw["tfopwg_disp"].isin(ACCEPTED_TRANSIT_DISPOSITIONS)].copy()
    logger.info(
        f"  After whitelist (CP/KP/PC/APC): {len(filtered)} rows kept"
    )

    # Build best-disposition lookup: for each TIC keep highest-confidence TOI
    toi_dict: dict[int, tuple[str, str]] = {}
    for _, row in filtered.iterrows():
        tid = int(row["tid"])
        toi_id = str(row["toi"])
        disp = str(row["tfopwg_disp"])
        prio = TRANSIT_DISPOSITION_PRIORITY[disp]

        if tid not in toi_dict:
            toi_dict[tid] = (toi_id, disp)
        else:
            existing_prio = TRANSIT_DISPOSITION_PRIORITY[toi_dict[tid][1]]
            if prio < existing_prio:
                toi_dict[tid] = (toi_id, disp)

    logger.info(f"  TOI lookup: {len(toi_dict)} unique TIC IDs")
    return toi_dict


# ---------------------------------------------------------------------------
# Catalog loaders — VizieR helpers
# ---------------------------------------------------------------------------

def _vizier_fetch(
    catalog_id: str,
    columns: list[str],
    logger: logging.Logger,
    label: str,
) -> Optional[pd.DataFrame]:
    """Try to fetch a VizieR catalog. Returns DataFrame or None."""
    try:
        from astroquery.vizier import Vizier

        @_make_retry()
        def _attempt() -> pd.DataFrame:
            v = Vizier(columns=columns, row_limit=-1)
            tables = v.get_catalogs(catalog_id)
            if not tables:
                raise ValueError(f"VizieR returned empty for {catalog_id}")
            return tables[0].to_pandas()

        df = _attempt()
        logger.info(f"  {label}: VizieR {catalog_id} → {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  {label}: VizieR fetch failed: {type(e).__name__}: {e}")
        return None


def _load_local_csv(
    path: Path,
    logger: logging.Logger,
    label: str,
) -> Optional[pd.DataFrame]:
    """Try to load a local CSV fallback. Returns DataFrame or None."""
    if not path.exists():
        logger.warning(f"  {label}: local fallback not found at {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"  {label}: loaded local {path.name} → {len(df)} rows")
    return df


def _load_catalog(
    vizier_id: str,
    vizier_columns: list[str],
    local_path: Path,
    logger: logging.Logger,
    label: str,
    essential: bool,
) -> pd.DataFrame:
    """Load catalog: VizieR first, then local CSV fallback.

    If essential=True, raises on failure. Otherwise returns empty DataFrame.
    """
    logger.info(f"Loading {label}...")
    df = _vizier_fetch(vizier_id, vizier_columns, logger, label)
    if df is None:
        df = _load_local_csv(local_path, logger, label)
    if df is None:
        msg = (
            f"{label}: all acquisition paths failed. "
            f"Place catalog at {local_path} and retry."
        )
        if essential:
            raise FileNotFoundError(msg)
        logger.warning(msg + " Proceeding without this catalog.")
        return pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Catalog loaders — EB (Villanova/Prša+2022 positive, single source)
# ---------------------------------------------------------------------------
#
# EB positive source = Villanova / Prša+2022 (J/ApJS/258/16) ONLY. See
# ADR-0007. Kostov+2025 was dropped: it is an FFI/faint catalog and overlaps
# the Tmag<10 SPOC 2-min corpus in only 22 TICs (dead task). Villanova is
# 2-min cadence (same target pool); though vetted on Sectors 1–26 — disjoint
# from the corpus's Sectors 27+ windows — its bright EBs recur in later-cycle
# 2-min observations (1,541 overlap with windowed TICs), and EB-ness is an
# epoch-stable stellar property, so the label transfers. eb is now strictly
# 0/1: no negative mask, no NaN branch.

_VILLANOVA_PERIOD_CANDIDATES = ["Per", "Porb", "Period"]


def load_villanova_eb(
    root: Path, logger: logging.Logger,
) -> tuple[set[int], dict[int, float]]:
    """Load Villanova/Prša+2022 EB catalog as the v1 positive source.

    Returns (tic_set, {tic_id: orbital_period_days}). A TIC with a missing/NaN
    period is still a valid positive (eb=1, eb_period=NaN) — period is
    supplementary metadata, not a gate on membership.

    Essential — raises on failure (it is the sole EB positive source; a silent
    empty set would resurrect the dead-task state ADR-0007 fixes).
    """
    df = _load_catalog(
        vizier_id="J/ApJS/258/16",
        vizier_columns=["TIC", "Per", "**"],
        local_path=root / "data" / "villanova_eb.csv",
        logger=logger,
        label="Villanova EB",
        essential=True,
    )
    cols_lower = {c.lower(): c for c in df.columns}
    tic_col = cols_lower.get("tic") or cols_lower.get("ticid") or cols_lower.get("tid")
    if tic_col is None:
        raise KeyError(
            f"Villanova: no TIC column. Columns: {df.columns.tolist()}"
        )
    per_col = next(
        (cols_lower[c.lower()] for c in _VILLANOVA_PERIOD_CANDIDATES
         if c.lower() in cols_lower),
        None,
    )
    if per_col is None:
        raise KeyError(
            f"Villanova: no period column among {_VILLANOVA_PERIOD_CANDIDATES}. "
            f"Columns: {df.columns.tolist()}"
        )

    tics = pd.to_numeric(df[tic_col], errors="coerce")
    pers = pd.to_numeric(df[per_col], errors="coerce")
    period_dict: dict[int, float] = {}
    for tid, per in zip(tics, pers):  # share df.index → positional zip aligns
        if pd.isna(tid):
            continue
        tid = int(tid)
        if tid not in period_dict:
            period_dict[tid] = float(per) if pd.notna(per) else float("nan")

    tic_set = set(period_dict.keys())
    logger.info(
        f"  Villanova EB: {len(tic_set)} unique TICs "
        f"(TIC col={tic_col!r}, period col={per_col!r})"
    )
    return tic_set, period_dict


# ---------------------------------------------------------------------------
# Catalog loaders — Pulsating (Gao+2025 positive)
# ---------------------------------------------------------------------------

def load_gao_pulsating(
    root: Path, logger: logging.Logger,
) -> tuple[set[int], dict[int, float], dict[int, str]]:
    """Load Gao+2025, filter to pulsating subtypes.

    Returns (tic_set, {tic: period}, {tic: subtype}).
    Essential — raises on failure.
    """
    df = _load_catalog(
        vizier_id="J/ApJS/276/57",
        vizier_columns=["TIC", "Per", "Type", "**"],
        local_path=root / "data" / "gao_2025_periodic_variables.csv",
        logger=logger,
        label="Gao+2025 periodic variables",
        essential=True,
    )
    cols_lower = {c.lower(): c for c in df.columns}
    tic_col = cols_lower.get("tic") or cols_lower.get("ticid") or cols_lower.get("tid")
    per_col = cols_lower.get("per") or cols_lower.get("period")
    type_col = cols_lower.get("type") or cols_lower.get("class") or cols_lower.get("subtype")
    if tic_col is None:
        raise KeyError(f"Gao+2025: no TIC column. Columns: {df.columns.tolist()}")
    if per_col is None:
        raise KeyError(f"Gao+2025: no period column. Columns: {df.columns.tolist()}")
    if type_col is None:
        raise KeyError(f"Gao+2025: no type column. Columns: {df.columns.tolist()}")

    logger.info(f"  Gao+2025 total rows: {len(df)}")
    subtype_counts = df[type_col].value_counts()
    for st, cnt in subtype_counts.items():
        logger.info(f"    {st}: {cnt}")

    # Filter to pulsating subtypes
    df["_subtype_upper"] = df[type_col].astype(str).str.upper().str.strip()
    pulsating_mask = df["_subtype_upper"].isin(GAO_PULSATING_SUBTYPES)
    puls = df[pulsating_mask].copy()
    logger.info(
        f"  Gao+2025 pulsating subset: {len(puls)} rows "
        f"(from {GAO_PULSATING_SUBTYPES})"
    )

    period_dict: dict[int, float] = {}
    subtype_dict: dict[int, str] = {}
    for _, row in puls.iterrows():
        tid = int(row[tic_col])
        if tid not in period_dict:
            period_dict[tid] = float(row[per_col])
            subtype_dict[tid] = str(row[type_col]).strip()

    tic_set = set(period_dict.keys())
    logger.info(f"  Gao+2025 pulsating: {len(tic_set)} unique TICs")
    return tic_set, period_dict, subtype_dict




# ---------------------------------------------------------------------------
# Segment-level npz scan
# ---------------------------------------------------------------------------

def scan_sequences(
    sequences_dir: Path, logger: logging.Logger,
) -> dict[int, int]:
    """Scan processed/sequences/*.npz, return {tic_id: segment_count}.

    Skips gracefully if directory does not exist.
    """
    if not sequences_dir.exists():
        logger.info(
            f"Sequences dir not found ({sequences_dir}); "
            "segment-level counts will be skipped."
        )
        return {}
    logger.info(f"Scanning {sequences_dir} for segment-level counts...")
    tic_counts: dict[int, int] = {}
    for p in sequences_dir.glob("TIC*_s*_*.npz"):
        m = _NPZ_TIC_RE.match(p.name)
        if m:
            tid = int(m.group(1))
            tic_counts[tid] = tic_counts.get(tid, 0) + 1
    logger.info(
        f"  Segment scan: {sum(tic_counts.values())} npz files, "
        f"{len(tic_counts)} unique TICs"
    )
    return tic_counts


# ---------------------------------------------------------------------------
# Progress / output CSV
# ---------------------------------------------------------------------------

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
    # Schema migration guard: reject old-schema files
    if "eb" not in df.columns:
        return []
    return df.to_dict("records")


def save_output(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    else:
        df = pd.DataFrame(columns=OUTPUT_COLS)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Per-star lookup (pure in-memory — no network calls)
# ---------------------------------------------------------------------------

def process_star(
    tic_id: int,
    tars_dict: dict[int, float],
    flare_tic_set: set[int],
    toi_dict: dict[int, tuple[str, str]],
    villanova_eb_set: set[int],
    villanova_eb_periods: dict[int, float],
    gao_pulsating_set: set[int],
    gao_pulsating_periods: dict[int, float],
    gao_pulsating_subtypes: dict[int, str],
) -> tuple[str, dict, str]:
    """Look up one TIC in all catalog families. Returns (status, row_dict, error_msg)."""
    try:
        # --- Rotation (TARS) ---
        if tic_id in tars_dict:
            rotation = 1
            rotation_period = tars_dict[tic_id]
            if not np.isfinite(rotation_period):
                rotation_period = float("nan")
        else:
            rotation = 0
            rotation_period = float("nan")

        # --- Flare (flatwrm2, excluded from v1 eval) ---
        flare_ever = 1 if tic_id in flare_tic_set else 0

        # --- Transit (NASA TOI, whitelist + best-disposition picker) ---
        if tic_id in toi_dict:
            transit = 1
            toi_id, transit_disposition = toi_dict[tic_id]
        else:
            transit = 0
            toi_id = float("nan")
            transit_disposition = float("nan")

        # --- EB (Villanova/Prša+2022 positive, single source; ADR-0007) ---
        # eb is strictly 0/1: no negative mask, no NaN branch.
        if tic_id in villanova_eb_set:
            eb = 1
            eb_period = villanova_eb_periods.get(tic_id, float("nan"))
            if not np.isfinite(eb_period):
                eb_period = float("nan")
        else:
            eb = 0
            eb_period = float("nan")

        # --- Pulsating (Gao+2025 positive only) ---
        if tic_id in gao_pulsating_set:
            pulsating = 1
            pulsating_period = gao_pulsating_periods.get(tic_id, float("nan"))
            pulsating_subtype = gao_pulsating_subtypes.get(tic_id, float("nan"))
            if not np.isfinite(pulsating_period):
                pulsating_period = float("nan")
        else:
            pulsating = 0
            pulsating_period = float("nan")
            pulsating_subtype = float("nan")

        row = {
            "tic_id": int(tic_id),
            "rotation": rotation,
            "rotation_period": rotation_period,
            "flare_ever": flare_ever,
            "transit": transit,
            "toi_id": toi_id,
            "transit_disposition": transit_disposition,
            "eb": eb,
            "eb_period": eb_period,
            "pulsating": pulsating,
            "pulsating_period": pulsating_period,
            "pulsating_subtype": pulsating_subtype,
        }
        return ("done", row, "")
    except Exception as e:
        return ("error", {}, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

_ROTATION_BUCKETS = [
    ("P_rot <= 2d", 0.0, 2.0),
    ("2 < P_rot <= 5d", 2.0, 5.0),
    ("5 < P_rot <= 10d", 5.0, 10.0),
    ("P_rot > 10d", 10.0, 1e9),
]

_EB_PERIOD_BUCKETS = [
    ("P < 1d", 0.0, 1.0),
    ("1 <= P < 5d", 1.0, 5.0),
    ("5 <= P < 10d", 5.0, 10.0),
    ("P >= 10d", 10.0, 1e9),
]

_PULSATING_PERIOD_BUCKETS = [
    ("P < 0.2d", 0.0, 0.2),
    ("0.2 <= P < 2d", 0.2, 2.0),
    ("2 <= P < 5d", 2.0, 5.0),
    ("P >= 5d", 5.0, 1e9),
]


def _bucket_counts(
    periods: pd.Series, buckets: list[tuple[str, float, float]],
) -> list[tuple[str, int]]:
    """Count how many periods fall into each bucket."""
    valid = periods.dropna()
    results = []
    for label, lo, hi in buckets:
        cnt = int(((valid >= lo) & (valid < hi)).sum())
        results.append((label, cnt))
    return results


def _segment_count_for_tics(
    tic_mask: pd.Series,
    df: pd.DataFrame,
    seg_counts: dict[int, int],
) -> int:
    """Count total segments for TICs matching a boolean mask."""
    if not seg_counts:
        return 0
    return sum(
        seg_counts.get(int(tid), 0)
        for tid in df.loc[tic_mask, "tic_id"]
    )


def print_summary(
    rows: list[dict],
    total_input: int,
    n_processed: int,
    seg_counts: dict[int, int],
    logger: logging.Logger,
) -> None:
    if not rows:
        logger.info("No output rows to summarize.")
        return

    df = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df["rotation"] = pd.to_numeric(df["rotation"], errors="coerce").fillna(0).astype(int)
    df["flare_ever"] = pd.to_numeric(df["flare_ever"], errors="coerce").fillna(0).astype(int)
    df["transit"] = pd.to_numeric(df["transit"], errors="coerce")
    df["eb"] = pd.to_numeric(df["eb"], errors="coerce")
    df["pulsating"] = pd.to_numeric(df["pulsating"], errors="coerce")

    r = df["rotation"] == 1
    f = df["flare_ever"] == 1
    t = df["transit"] == 1
    e = df["eb"] == 1
    p = df["pulsating"] == 1

    puls_nan = df["pulsating"].isna()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Variability Label Build Summary")
    logger.info("=" * 70)
    logger.info(f"Input TICs:                        {total_input}")
    logger.info(f"TICs processed this run:           {n_processed}")
    logger.info(f"Output rows (cumulative on disk):  {len(rows)}")

    # --- Per-task star-level counts ---
    logger.info("")
    logger.info("Per-task star-level counts:")
    logger.info(f"  rotation=1:     {r.sum()}")
    logger.info(f"  flare_ever=1:   {f.sum()}  (excluded from v1 eval)")
    logger.info(f"  transit=1:      {t.sum()}")

    # Transit disposition breakdown
    if t.sum() > 0:
        disp_col = df.loc[t, "transit_disposition"]
        for d in ["CP", "KP", "PC", "APC"]:
            cnt = int((disp_col == d).sum())
            if cnt > 0:
                logger.info(f"    {d}: {cnt}")

    # EB: star-level positives, plus how many have ≥1 .npz window (the count
    # the linear probe can actually use; the gap is the star-vs-window delta).
    if seg_counts:
        windowed = set(seg_counts.keys())
        eb_windowed = int(df.loc[e, "tic_id"].astype(int).isin(windowed).sum())
        logger.info(
            f"  eb=1:           {e.sum()}  (with ≥1 window: {eb_windowed})"
        )
    else:
        logger.info(f"  eb=1:           {e.sum()}")
    logger.info(f"  pulsating=1:    {p.sum()}  (NaN-excluded: {puls_nan.sum()})")

    # Pulsating subtype breakdown
    if p.sum() > 0:
        st_col = df.loc[p, "pulsating_subtype"]
        st_counts = st_col.value_counts()
        for st, cnt in st_counts.items():
            logger.info(f"    {st}: {cnt}")

    # --- Segment-level counts ---
    if seg_counts:
        logger.info("")
        logger.info("Segment-level positive counts:")
        logger.info(
            f"  rotation=1:   {_segment_count_for_tics(r, df, seg_counts)} segments"
        )
        logger.info(
            f"  transit=1:    {_segment_count_for_tics(t, df, seg_counts)} segments"
        )
        logger.info(
            f"  eb=1:         {_segment_count_for_tics(e, df, seg_counts)} segments"
        )
        logger.info(
            f"  pulsating=1:  {_segment_count_for_tics(p, df, seg_counts)} segments"
        )

    # --- Period-bucket stratification ---
    logger.info("")
    logger.info("Period-bucket stratification (star-level):")

    logger.info("  Rotation:")
    rot_periods = df.loc[r, "rotation_period"].astype(float)
    for label, cnt in _bucket_counts(rot_periods, _ROTATION_BUCKETS):
        logger.info(f"    {label}: {cnt}")

    logger.info("  EB:")
    eb_periods = df.loc[e, "eb_period"].astype(float)
    for label, cnt in _bucket_counts(eb_periods, _EB_PERIOD_BUCKETS):
        logger.info(f"    {label}: {cnt}")

    logger.info("  Pulsating:")
    puls_periods = df.loc[p, "pulsating_period"].astype(float)
    for label, cnt in _bucket_counts(puls_periods, _PULSATING_PERIOD_BUCKETS):
        logger.info(f"    {label}: {cnt}")

    # --- Combination counts (primary tasks only) ---
    logger.info("")
    logger.info("Primary-task combinations (transit/EB/pulsating):")
    logger.info(f"  transit only:           {(t & ~e & ~p).sum()}")
    logger.info(f"  eb only:                {(~t & e & ~p).sum()}")
    logger.info(f"  pulsating only:         {(~t & ~e & p).sum()}")
    logger.info(f"  transit + eb:           {(t & e & ~p).sum()}")
    logger.info(f"  transit + pulsating:    {(t & ~e & p).sum()}")
    logger.info(f"  eb + pulsating:         {(~t & e & p).sum()}")
    logger.info(f"  all three:              {(t & e & p).sum()}")

    quiet = (~r) & (~f) & (~t) & (~e) & (~p)
    logger.info(f"  quiet (no labels):      {quiet.sum()}")
    logger.info("=" * 70)


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

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else root / "processed" / "spoc_sector_map.csv"
    )
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else root / "labels" / "variability_labels_star.csv"
    )
    prog_csv = (
        Path(args.progress_csv)
        if args.progress_csv
        else root / "labels" / "build_variability_labels_progress.csv"
    )
    tars_path = (
        Path(args.tars_catalog)
        if args.tars_catalog
        else root / "data" / "tars_table_2.feather"
    )
    flare_path = (
        Path(args.flare_catalog)
        if args.flare_catalog
        else root / "data" / "Table3_flare_catalog.csv"
    )
    log_file = (
        Path(args.log_file)
        if args.log_file
        else root / "build_variability_labels.log"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    # ------------------------------------------------------------------
    # Validate local inputs
    # ------------------------------------------------------------------
    if not input_csv.exists():
        logger.error(f"Input CSV not found: {input_csv}")
        return 1
    if not tars_path.exists():
        logger.error(f"TARS catalog not found: {tars_path}")
        logger.error(
            "Download from Zenodo 10.5281/zenodo.19917941 "
            "and place at data/tars_table_2.feather"
        )
        return 1
    if not flare_path.exists():
        logger.error(f"flatwrm2 catalog not found: {flare_path}")
        logger.error(
            "Download Table3_flare_catalog.csv from "
            "Zenodo 10.5281/zenodo.14179313 and place in data/"
        )
        return 1

    # ------------------------------------------------------------------
    # Load all catalogs into memory (sequential, fail-loud for essential)
    # ------------------------------------------------------------------
    try:
        _, tars_dict = load_tars(tars_path, logger)
    except Exception as e:
        logger.error(f"Failed to load TARS: {e}")
        return 1

    try:
        flare_tic_set = load_flares(flare_path, logger)
    except Exception as e:
        logger.error(f"Failed to load flatwrm2: {e}")
        return 1

    logger.info("Fetching NASA Exoplanet Archive TOI table...")
    try:
        toi_dict = fetch_toi_table(logger)
    except Exception as e:
        logger.error(f"TOI fetch failed: {type(e).__name__}: {e}")
        return 1

    try:
        villanova_eb_set, villanova_eb_periods = load_villanova_eb(root, logger)
    except Exception as e:
        logger.error(f"Failed to load Villanova EB: {e}")
        return 1

    try:
        gao_pulsating_set, gao_pulsating_periods, gao_pulsating_subtypes = (
            load_gao_pulsating(root, logger)
        )
    except Exception as e:
        logger.error(f"Failed to load Gao+2025 pulsating: {e}")
        return 1

    # ------------------------------------------------------------------
    # Scan sequences directory for segment-level counts
    # ------------------------------------------------------------------
    seg_counts = scan_sequences(root / "processed" / "sequences", logger)

    # ------------------------------------------------------------------
    # Load star list
    # ------------------------------------------------------------------
    df_in = pd.read_csv(input_csv)
    if "ID" in df_in.columns:
        tic_ids = df_in["ID"].astype(int).tolist()
    elif "tic_id" in df_in.columns:
        tic_ids = df_in["tic_id"].drop_duplicates().astype(int).tolist()
    else:
        logger.error(
            f"Input CSV missing 'ID' or 'tic_id' column. "
            f"Found: {list(df_in.columns)}"
        )
        return 1
    total_input = len(tic_ids)
    corpus_set = set(tic_ids)
    windowed_tics = set(seg_counts.keys())  # TICs with ≥1 .npz window
    logger.info(
        f"Villanova EB input overlap: {len(villanova_eb_set & corpus_set)} of "
        f"{len(villanova_eb_set)} Villanova TICs present in {total_input}-TIC "
        f"input list (spoc_sector_map)"
    )
    if windowed_tics:
        logger.info(
            f"Villanova EB window-grounded overlap: "
            f"{len(villanova_eb_set & windowed_tics)} of {len(villanova_eb_set)} "
            f"Villanova TICs have ≥1 .npz window "
            f"({len(windowed_tics)} windowed TICs total) — this is the count the "
            f"linear probe can actually use"
        )

    # ------------------------------------------------------------------
    # Resume logic (with schema migration guard)
    # ------------------------------------------------------------------
    progress = load_progress(prog_csv)
    output_rows = load_existing_output(out_csv)

    if args.resume and output_rows == [] and out_csv.exists():
        # Old-schema file exists but load_existing_output rejected it
        try:
            old_df = pd.read_csv(out_csv, nrows=1)
            if "eb" not in old_df.columns:
                logger.error(
                    "Existing output CSV has old schema (missing 'eb' column). "
                    "Schema changed — clean run required. "
                    "Delete or rename the old output and progress files, "
                    "then rerun without --resume."
                )
                return 1
        except Exception:
            pass

    output_index = {int(r["tic_id"]): i for i, r in enumerate(output_rows)}

    if args.resume and len(progress) > 0:
        skip = set(
            progress.loc[progress["status"] == "done", "tic_id"].astype(int)
        )
        before = len(tic_ids)
        tic_ids = [t for t in tic_ids if t not in skip]
        logger.info(
            f"Resume: skipping {before - len(tic_ids)} done stars; "
            f"{len(tic_ids)} remaining."
        )

    if args.limit is not None:
        tic_ids = tic_ids[: args.limit]
        logger.info(f"--limit {args.limit}: processing {len(tic_ids)} stars.")

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------
    signal.signal(signal.SIGINT, _sigint_handler)

    progress_records: list[dict] = progress.to_dict("records")
    progress_index = {
        int(r["tic_id"]): i for i, r in enumerate(progress_records)
    }

    n_done = 0
    n_error = 0
    t0 = time.time()

    logger.info(f"Starting: {len(tic_ids)} TICs to process")

    for i, tic_id in enumerate(tic_ids, start=1):
        if _INTERRUPTED:
            logger.warning(
                "Ctrl+C received — flushing progress and exiting cleanly."
            )
            break

        status, row, err = process_star(
            tic_id,
            tars_dict=tars_dict,
            flare_tic_set=flare_tic_set,
            toi_dict=toi_dict,
            villanova_eb_set=villanova_eb_set,
            villanova_eb_periods=villanova_eb_periods,
            gao_pulsating_set=gao_pulsating_set,
            gao_pulsating_periods=gao_pulsating_periods,
            gao_pulsating_subtypes=gao_pulsating_subtypes,
        )

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
            save_progress(
                pd.DataFrame(progress_records, columns=PROGRESS_COLS),
                prog_csv,
            )
            save_output(output_rows, out_csv)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(tic_ids) - i
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            logger.info(
                f"Progress {i}/{len(tic_ids)}  done={n_done} err={n_error}  "
                f"rate={rate:.0f} stars/s  ETA={eta_min:.1f} min"
            )

    save_progress(
        pd.DataFrame(progress_records, columns=PROGRESS_COLS), prog_csv
    )
    save_output(output_rows, out_csv)

    print_summary(output_rows, total_input, len(tic_ids), seg_counts, logger)

    if _INTERRUPTED:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
