"""Step 0 of the Phase-0 data-quality roadmap: one shared, cached pull of the enriched TOI tables.

Tasks A (SPOC filter), B (EB cross-check tier A) and C (transit window coverage) all need TOI
metadata; rather than each hitting the network, this pulls once and writes two caches under labels/qc/:

  toi_nasa.csv    NASA Exoplanet Archive `toi` table — ephemeris (pl_orbper, pl_tranmid[BJD],
                  pl_trandurh) + tfopwg_disp, keyed by tid (=TIC). Source of truth for Task C folding.
  toi_exofop.csv  ExoFOP TOI list — the ONLY source of the detection-pipeline field (`Detection`,
                  e.g. SPOC / QLP / FAINT / SPOC/QLP) and an explicit EB adjudication (`TESS Disposition`
                  == 'EB'). Keyed by 'TIC ID'. Drives Task A (SPOC) and Task B tier A.

NASA TAP has NO pipeline column (verified 2026-07-14), which is why ExoFOP is mandatory here.

Run (astro env, from repo root):
    python src/qc/fetch_toi_enriched.py
    python src/qc/fetch_toi_enriched.py --refresh   # ignore cache, re-pull
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import qc_common` when run as a script
from qc_common import call_with_retry, find_project_root, setup_logging

NASA_COLS = [
    "tid", "toi", "tfopwg_disp",
    "pl_orbper", "pl_tranmid", "pl_trandurh", "pl_trandep",
    "st_tmag", "sectors",
]
EXOFOP_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"


def fetch_nasa_toi(logger) -> pd.DataFrame:
    """Full NASA `toi` table, trimmed to the columns Task C needs. Ephemeris epoch is BJD (TDB)."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    def _q():
        return NasaExoplanetArchive.query_criteria(table="toi", select="*").to_pandas()

    df = call_with_retry(_q, "NASA toi", logger)
    logger.info(f"NASA toi: {len(df)} rows, {len(df.columns)} cols")
    keep = [c for c in NASA_COLS if c in df.columns]
    missing = [c for c in NASA_COLS if c not in df.columns]
    if missing:
        logger.warning(f"NASA toi missing expected cols (kept going): {missing}")
    return df[keep].copy()


def fetch_exofop_toi(logger) -> pd.DataFrame:
    """Full ExoFOP TOI CSV. Carries the detection-pipeline field absent from NASA TAP."""
    import requests

    def _q():
        r = requests.get(EXOFOP_URL, timeout=300)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))

    df = call_with_retry(_q, "ExoFOP TOI CSV", logger)
    logger.info(f"ExoFOP TOI: {len(df)} rows, {len(df.columns)} cols")
    for col in ("Detection", "TESS Disposition", "TFOPWG Disposition"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).head(8).to_dict()
            logger.info(f"  {col!r} top: {vc}")
        else:
            logger.warning(f"ExoFOP missing expected column {col!r}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 0: cached enriched TOI pull (NASA + ExoFOP).")
    ap.add_argument("--refresh", action="store_true", help="Ignore existing caches and re-pull.")
    ap.add_argument("--out-dir", default=None, help="Default: <root>/labels/qc")
    args = ap.parse_args()

    root = find_project_root()
    out_dir = Path(args.out_dir) if args.out_dir else root / "labels" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "qc_fetch_toi_enriched.log", "fetch_toi_enriched")

    nasa_path = out_dir / "toi_nasa.csv"
    exofop_path = out_dir / "toi_exofop.csv"

    # NASA
    if nasa_path.exists() and not args.refresh:
        n = len(pd.read_csv(nasa_path))
        logger.info(f"cache hit {nasa_path} ({n} rows) — skip (use --refresh to re-pull)")
    else:
        nasa = fetch_nasa_toi(logger)
        assert len(nasa) > 0, "NASA toi returned 0 rows — refusing to write empty cache"
        nasa.to_csv(nasa_path, index=False)
        logger.info(f"wrote {nasa_path} ({len(nasa)} rows)")

    # ExoFOP
    if exofop_path.exists() and not args.refresh:
        n = len(pd.read_csv(exofop_path))
        logger.info(f"cache hit {exofop_path} ({n} rows) — skip (use --refresh to re-pull)")
    else:
        exofop = fetch_exofop_toi(logger)
        assert len(exofop) > 0, "ExoFOP TOI returned 0 rows — refusing to write empty cache"
        assert "Detection" in exofop.columns, "ExoFOP CSV lacks 'Detection' — Task A cannot proceed"
        exofop.to_csv(exofop_path, index=False)
        logger.info(f"wrote {exofop_path} ({len(exofop)} rows)")

    logger.info("Step 0 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
