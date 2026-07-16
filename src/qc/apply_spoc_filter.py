"""Task A — SPOC-detection filter on the transit labels; derive labels/variability_labels_star_v2.csv.

Our light curves are SPOC 2-minute PDCSAP. A TOI detected only by QLP (30-min FFI) or the faint-star
FFI search is a *different* light curve than the one the model ever sees, so its transit dip may not be
present in our data at all. This filter keeps a transit label only when the TOI was detected by SPOC.

Pipeline provenance lives ONLY in the ExoFOP TOI list (`Detection` column: SPOC / QLP / FAINT /
SPOC/QLP / ...); the NASA TAP `toi` table has no such field (verified 2026-07-14). Inclusive rule:
"SPOC-detected" == 'SPOC' appears in Detection (so SPOC/QLP and SPOC/FAINT are kept — if SPOC saw it,
the dip is in our 2-min data).

Derivation (grilling decision): v2 is v1 with ONLY the transit / toi_id / transit_disposition columns
possibly changed; eb / pulsating / rotation / flare stay byte-identical so the Phase-2 sanity check is
exact. We remove a transit label only on POSITIVE evidence of non-SPOC detection; a TIC we cannot match
in ExoFOP (or whose Detection is blank) is KEPT and flagged 'unmatched', never silently dropped.

Run (astro or any env with pandas; from repo root; needs labels/qc/toi_exofop.csv):
    python src/qc/apply_spoc_filter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import find_project_root, setup_logging

WHITELIST = {"CP", "KP", "PC", "APC"}


def build_spoc_detection(exofop_csv: Path, logger) -> pd.DataFrame:
    """Per-TIC SPOC-detection evidence over the TIC's whitelisted TOIs.

    Returns a frame indexed by tic_id with columns:
      any_spoc   : True if any whitelisted TOI of the TIC has 'SPOC' in Detection
      n_wl_rows  : number of whitelisted TOI rows for the TIC
      detections : sorted unique Detection strings (for the report)
    """
    ex = pd.read_csv(exofop_csv)
    ex.columns = [c.strip() for c in ex.columns]
    tic_col = "TIC ID"
    disp_col = "TFOPWG Disposition"
    det_col = "Detection"
    for c in (tic_col, disp_col, det_col):
        assert c in ex.columns, f"ExoFOP CSV missing {c!r}; cols={ex.columns.tolist()}"

    ex["_tic"] = pd.to_numeric(ex[tic_col], errors="coerce")
    ex["_disp"] = ex[disp_col].astype(str).str.strip().str.upper()
    ex["_det"] = ex[det_col].astype(str).str.strip()
    ex = ex[ex["_tic"].notna()].copy()
    wl = ex[ex["_disp"].isin(WHITELIST)].copy()
    logger.info(f"ExoFOP: {len(ex)} rows; whitelisted-disposition rows: {len(wl)}")

    wl["_is_spoc"] = wl["_det"].str.upper().str.contains("SPOC", na=False)
    grp = wl.groupby(wl["_tic"].astype(int))
    out = pd.DataFrame({
        "any_spoc": grp["_is_spoc"].any(),
        "n_wl_rows": grp.size(),
        "detections": grp["_det"].apply(lambda s: ";".join(sorted(set(s.dropna())))),
    })
    out.index.name = "tic_id"
    logger.info(f"ExoFOP whitelisted TICs: {len(out)}; any_spoc True: {int(out['any_spoc'].sum())}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Task A: SPOC-detection transit filter -> v2 labels.")
    ap.add_argument("--labels-csv", default=None, help="Default: labels/variability_labels_star.csv (v1).")
    ap.add_argument("--exofop-csv", default=None, help="Default: labels/qc/toi_exofop.csv")
    ap.add_argument("--out-v2", default=None, help="Default: labels/variability_labels_star_v2.csv")
    ap.add_argument("--report-csv", default=None, help="Default: labels/qc/transit_spoc_filter_report.csv")
    args = ap.parse_args()

    root = find_project_root()
    labels_csv = Path(args.labels_csv) if args.labels_csv else root / "labels" / "variability_labels_star.csv"
    exofop_csv = Path(args.exofop_csv) if args.exofop_csv else root / "labels" / "qc" / "toi_exofop.csv"
    out_v2 = Path(args.out_v2) if args.out_v2 else root / "labels" / "variability_labels_star_v2.csv"
    report_csv = Path(args.report_csv) if args.report_csv else root / "labels" / "qc" / "transit_spoc_filter_report.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "qc_apply_spoc_filter.log", "apply_spoc_filter")

    assert exofop_csv.exists(), f"missing {exofop_csv}; run src/qc/fetch_toi_enriched.py first"
    v1 = pd.read_csv(labels_csv)
    v1["tic_id"] = v1["tic_id"].astype(int)
    v1["transit"] = pd.to_numeric(v1["transit"], errors="coerce").fillna(0).astype(int)
    n_before = int((v1["transit"] == 1).sum())

    spoc = build_spoc_detection(exofop_csv, logger)

    # classify each v1 transit-positive TIC
    trans = v1[v1["transit"] == 1].copy()
    trans = trans.join(spoc, on="tic_id")

    def _category(row) -> str:
        if pd.isna(row.get("any_spoc")):
            return "KEEP_UNMATCHED"       # not in ExoFOP whitelist rows -> no evidence -> keep + flag
        return "KEEP_SPOC" if bool(row["any_spoc"]) else "DROP_NONSPOC"

    trans["category"] = trans.apply(_category, axis=1)
    drop_tics = set(trans.loc[trans["category"] == "DROP_NONSPOC", "tic_id"].astype(int))

    # build v2: copy v1, blank transit columns for dropped TICs only
    v2 = v1.copy()
    drop_mask = v2["tic_id"].isin(drop_tics)
    v2.loc[drop_mask, "transit"] = 0
    for col in ("toi_id", "transit_disposition"):
        if col in v2.columns:
            v2.loc[drop_mask, col] = np.nan
    v2.to_csv(out_v2, index=False)
    n_after = int((v2["transit"] == 1).sum())

    # report
    rep = trans[["tic_id", "transit_disposition", "n_wl_rows", "detections", "category"]].copy()
    rep = rep.rename(columns={"transit_disposition": "v1_disposition", "detections": "exofop_detections"})
    rep["transit_v2"] = np.where(rep["tic_id"].isin(drop_tics), 0, 1)
    rep = rep.sort_values(["category", "tic_id"])
    rep.to_csv(report_csv, index=False)

    cats = trans["category"].value_counts().to_dict()
    logger.info("=" * 66)
    logger.info("Task A — SPOC-detection transit filter (what-if)")
    logger.info("=" * 66)
    logger.info(f"n_transit_before:   {n_before}")
    logger.info(f"n_transit_after:    {n_after}")
    logger.info(f"n_dropped:          {n_before - n_after}  (positively-confirmed non-SPOC)")
    logger.info(f"category breakdown: {cats}")
    unmatched = int(cats.get("KEEP_UNMATCHED", 0))
    logger.info(f"  KEEP_SPOC:       {int(cats.get('KEEP_SPOC', 0))}  (kept, SPOC in Detection)")
    logger.info(f"  DROP_NONSPOC:    {int(cats.get('DROP_NONSPOC', 0))}  (transit 1->0 in v2)")
    logger.info(f"  KEEP_UNMATCHED:  {unmatched}  (no ExoFOP whitelist row / blank Detection — kept, flagged)")
    if n_before - n_after > 0:
        dd = trans[trans["category"] == "DROP_NONSPOC"]
        det_counts = dd["detections"].value_counts().head(10).to_dict()
        logger.info(f"dropped-by-detection-string: {det_counts}")
    logger.info(f"wrote {out_v2}")
    logger.info(f"wrote {report_csv}")
    logger.info("eb/pulsating/rotation columns in v2 are byte-identical to v1 (only transit cols touched).")
    logger.info("Task A done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
