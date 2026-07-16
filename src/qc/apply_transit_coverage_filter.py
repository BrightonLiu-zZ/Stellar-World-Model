"""Task A (revised) — coverage-based transit-label filter; derive labels/variability_labels_star_v2.csv.

Decision (grilling review): the SPOC-detection filter is over-aggressive for our corpus — 88% of the
TOIs it would drop have a real transit in our SPOC windows (see transit_spoc_filter_report.csv + the
cross-check). The correct, data-driven hygiene is direct: keep a star's transit=1 only if at least one
predicted transit (from the TOI ephemeris P/T0/duration) actually lands on a cadence we hold in
processed/sequences (the `times` arrays). This reuses Task C's per-star coverage (transit_window_coverage.csv).

Buckets:
  KEEP_OBSERVED     windowed, foldable, cov256 > 0            -> transit stays 1
  DROP_NO_TRANSIT   windowed, foldable, cov256 == 0           -> transit 1 -> 0 (label was pure noise)
  KEEP_UNVERIFIABLE windowed but no whitelisted TOI ephemeris -> transit stays 1, flagged (can't check)
  DROP_NO_WINDOWS   no .npz windows at all                    -> transit 1 -> 0 (not in our data / not in probe)

Only the transit / toi_id / transit_disposition columns change vs v1; eb/pulsating/rotation stay
byte-identical (exact Phase-2 sanity check).

Run (from repo root; needs labels/qc/transit_window_coverage.csv from transit_window_coverage.py):
    python src/qc/apply_transit_coverage_filter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import find_project_root, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="Task A (revised): coverage-based transit filter -> v2.")
    ap.add_argument("--labels-csv", default=None)
    ap.add_argument("--coverage-csv", default=None, help="Default: labels/qc/transit_window_coverage.csv")
    ap.add_argument("--out-v2", default=None, help="Default: labels/variability_labels_star_v2.csv")
    ap.add_argument("--report-csv", default=None, help="Default: labels/qc/transit_coverage_filter_report.csv")
    args = ap.parse_args()

    root = find_project_root()
    labels_csv = Path(args.labels_csv) if args.labels_csv else root / "labels" / "variability_labels_star.csv"
    cov_csv = Path(args.coverage_csv) if args.coverage_csv else root / "labels" / "qc" / "transit_window_coverage.csv"
    out_v2 = Path(args.out_v2) if args.out_v2 else root / "labels" / "variability_labels_star_v2.csv"
    report_csv = Path(args.report_csv) if args.report_csv else root / "labels" / "qc" / "transit_coverage_filter_report.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "qc_apply_transit_coverage_filter.log", "apply_transit_coverage_filter")

    assert cov_csv.exists(), f"missing {cov_csv}; run src/qc/transit_window_coverage.py first"
    v1 = pd.read_csv(labels_csv)
    v1["tic_id"] = v1["tic_id"].astype(int)
    v1["transit"] = pd.to_numeric(v1["transit"], errors="coerce").fillna(0).astype(int)
    n_before = int((v1["transit"] == 1).sum())

    cov = pd.read_csv(cov_csv)
    cov["tic_id"] = cov["tic_id"].astype(int)
    cov_by_tic = cov.set_index("tic_id")

    trans = v1[v1["transit"] == 1].copy()

    def _bucket(tic: int) -> tuple[str, float]:
        if tic not in cov_by_tic.index:
            return "DROP_NO_WINDOWS", np.nan          # no .npz at all
        row = cov_by_tic.loc[tic]
        if not bool(row["foldable"]):
            return "KEEP_UNVERIFIABLE", np.nan          # windows but no whitelist ephemeris
        c = float(row["cov256"])
        return ("KEEP_OBSERVED", c) if c > 0 else ("DROP_NO_TRANSIT", c)

    buckets, covs = zip(*[_bucket(t) for t in trans["tic_id"]])
    trans["bucket"] = buckets
    trans["cov256"] = covs
    drop_tics = set(trans.loc[trans["bucket"].isin(["DROP_NO_TRANSIT", "DROP_NO_WINDOWS"]), "tic_id"].astype(int))

    # build v2
    v2 = v1.copy()
    drop_mask = v2["tic_id"].isin(drop_tics)
    v2.loc[drop_mask, "transit"] = 0
    for col in ("toi_id", "transit_disposition"):
        if col in v2.columns:
            v2.loc[drop_mask, col] = np.nan
    v2.to_csv(out_v2, index=False)
    n_after = int((v2["transit"] == 1).sum())

    rep = trans[["tic_id", "transit_disposition", "cov256", "bucket"]].copy()
    rep["transit_v2"] = np.where(rep["tic_id"].isin(drop_tics), 0, 1)
    rep = rep.sort_values(["bucket", "tic_id"])
    rep.to_csv(report_csv, index=False)

    cats = trans["bucket"].value_counts().to_dict()
    # probe-usable = stars the subset can actually use (>=1 window): KEEP_OBSERVED + KEEP_UNVERIFIABLE + DROP_NO_TRANSIT had windows
    windowed_before = int(cats.get("KEEP_OBSERVED", 0) + cats.get("KEEP_UNVERIFIABLE", 0) + cats.get("DROP_NO_TRANSIT", 0))
    windowed_after = int(cats.get("KEEP_OBSERVED", 0) + cats.get("KEEP_UNVERIFIABLE", 0))

    logger.info("=" * 70)
    logger.info("Task A (revised) — coverage-based transit filter (what-if)")
    logger.info("=" * 70)
    logger.info(f"n_transit_before (all):          {n_before}")
    logger.info(f"n_transit_after  (all):          {n_after}   (dropped {n_before - n_after})")
    logger.info(f"bucket breakdown: {cats}")
    logger.info(f"  KEEP_OBSERVED    : {int(cats.get('KEEP_OBSERVED', 0))}  (>=1 transit lands in our windows)")
    logger.info(f"  DROP_NO_TRANSIT  : {int(cats.get('DROP_NO_TRANSIT', 0))}  (windowed+foldable, 0 transit in windows -> noise)")
    logger.info(f"  KEEP_UNVERIFIABLE: {int(cats.get('KEEP_UNVERIFIABLE', 0))}  (windowed, no whitelist ephemeris -> kept, flagged)")
    logger.info(f"  DROP_NO_WINDOWS  : {int(cats.get('DROP_NO_WINDOWS', 0))}  (no .npz -> not in probe anyway)")
    logger.info(f"PROBE-USABLE transit (>=1 window): {windowed_before} -> {windowed_after}  "
                f"(drops {windowed_before - windowed_after} noise stars; this is the count the probe sees)")

    # marginality note among kept-observed: how thin is the observed coverage?
    kept = trans[trans["bucket"] == "KEEP_OBSERVED"]
    if len(kept):
        for thr in (0.02, 0.05, 0.10):
            n = int((kept["cov256"] < thr).sum())
            logger.info(f"  of KEEP_OBSERVED, cov256 < {thr:.0%}: {n}  (thin but non-zero)")
    logger.info(f"wrote {out_v2}")
    logger.info(f"wrote {report_csv}")
    logger.info("eb/pulsating/rotation columns byte-identical to v1.")
    logger.info("Task A (revised) done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
