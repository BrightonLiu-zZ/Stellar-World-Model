"""EB manual-review drops — apply user decisions to labels/variability_labels_star_v2.csv.

Consumes labels/qc/eb_manual_review.csv (produced by src/notebooks/eb_manual_review.ipynb; the
`user_decision` column is filled by the user per star: drop / keep). Every `drop` star gets
eb 1 -> 0 and eb_period cleared in the v2 label file; `keep` stars are untouched. The eb column
stays strictly 0/1 (no NaN branch, no negative mask) — dropped stars become plain negatives.

Status: the 2026-07-16 decisions are a DEFAULT drop-all (30 CONTAMINATION_CONFIRMED_PLANET planet
hosts + 2 SUSPECT_CONFIRMED_FULL_LC no-signal + 1 anchor, likely delta Sct) pending Prof
confirmation; reversible by editing the review CSV and re-running the chain below. This edits
individual Villanova entries only — the single-source rule (ADR-0007) is unchanged.

Ordering: apply_transit_coverage_filter.py regenerates v2 from v1 (transit columns), so it must
run FIRST; this script then layers the eb drops on top. Idempotent — re-running on an
already-applied v2 is a logged no-op.

Run (from repo root, astro env):
    python src/qc/apply_eb_review_drops.py
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
    ap = argparse.ArgumentParser(description="Apply EB manual-review drop decisions to the v2 labels.")
    ap.add_argument("--review-csv", default=None, help="Default: labels/qc/eb_manual_review.csv")
    ap.add_argument("--v2-csv", default=None, help="Default: labels/variability_labels_star_v2.csv")
    ap.add_argument("--report-csv", default=None, help="Default: labels/qc/eb_review_drops_report.csv")
    args = ap.parse_args()

    root = find_project_root()
    review_csv = Path(args.review_csv) if args.review_csv else root / "labels" / "qc" / "eb_manual_review.csv"
    v2_csv = Path(args.v2_csv) if args.v2_csv else root / "labels" / "variability_labels_star_v2.csv"
    report_csv = Path(args.report_csv) if args.report_csv else root / "labels" / "qc" / "eb_review_drops_report.csv"
    logger = setup_logging(root / "qc_apply_eb_review_drops.log", "apply_eb_review_drops")

    assert review_csv.exists(), f"missing {review_csv}; run the eb_manual_review notebook first"
    assert v2_csv.exists(), f"missing {v2_csv}; run src/qc/apply_transit_coverage_filter.py first"

    review = pd.read_csv(review_csv)
    review["tic_id"] = review["tic_id"].astype(int)
    decisions = review["user_decision"].astype(str).str.strip().str.lower()
    blank = review[decisions.isin(["", "nan"])]
    assert len(blank) == 0, f"{len(blank)} review rows have no user_decision — fill drop/keep first: {blank['tic_id'].tolist()}"
    bad = review[~decisions.isin(["drop", "keep"])]
    assert len(bad) == 0, f"user_decision must be drop/keep, got: {bad[['tic_id', 'user_decision']].to_dict('records')}"
    drop = review[decisions == "drop"].copy()
    drop_tics = set(drop["tic_id"].tolist())

    v2 = pd.read_csv(v2_csv)
    v2["tic_id"] = v2["tic_id"].astype(int)
    v2["eb"] = pd.to_numeric(v2["eb"], errors="coerce").fillna(0).astype(int)
    n_before = int((v2["eb"] == 1).sum())

    missing = drop_tics - set(v2["tic_id"].tolist())
    assert len(missing) == 0, f"drop TICs absent from v2 labels: {sorted(missing)}"
    drop_mask = v2["tic_id"].isin(drop_tics)
    already_zero = int(((v2["eb"] == 0) & drop_mask).sum())
    if already_zero:
        logger.info(f"{already_zero}/{len(drop_tics)} drop TICs already eb=0 (re-run; idempotent no-op for those)")

    v2.loc[drop_mask, "eb"] = 0
    v2.loc[drop_mask, "eb_period"] = np.nan
    v2.to_csv(v2_csv, index=False)
    n_after = int((v2["eb"] == 1).sum())

    # report: what was dropped, and whether the star survives as a positive elsewhere (e.g. transit)
    rep = review.merge(v2[["tic_id", "transit", "pulsating", "rotation"]], on="tic_id", how="left")
    rep["eb_v2"] = np.where(rep["tic_id"].isin(drop_tics), 0, 1)
    rep = rep.sort_values(["final_verdict", "tic_id"])
    rep.to_csv(report_csv, index=False)

    logger.info("=" * 70)
    logger.info("EB manual-review drops (what-if) — default drop-all pending Prof confirmation")
    logger.info("=" * 70)
    logger.info(f"review rows: {len(review)}  (drop {len(drop_tics)}, keep {len(review) - len(drop_tics)})")
    logger.info(f"per-verdict drops: {drop['final_verdict'].value_counts().to_dict()}")
    logger.info(f"n_eb_before: {n_before}")
    logger.info(f"n_eb_after:  {n_after}   (dropped {n_before - n_after}, {100 * (n_before - n_after) / max(n_before, 1):.2f}%)")
    still_transit = int(rep.loc[rep['eb_v2'] == 0, 'transit'].fillna(0).sum())
    logger.info(f"dropped stars still transit=1 in v2: {still_transit} (planet hosts stay in the subset as transit positives)")
    logger.info(f"wrote {v2_csv}")
    logger.info(f"wrote {report_csv}")
    logger.info("transit/pulsating/rotation columns untouched by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
