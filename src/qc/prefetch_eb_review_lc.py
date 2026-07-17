"""Warm the lightkurve cache for the EB manual-review notebook (plan 2026-07-16 D7 step 2).

The review notebook plots full-PDCSAP folds for the 30 CONTAMINATION_CONFIRMED_PLANET stars, the 2
SUSPECT_CONFIRMED_FULL_LC stars, and the anchor TIC 233169434 (33 total, from
labels/qc/eb_villanova_audit.csv). Multi-sector SPOC downloads are the wall-clock bottleneck, so this
script pre-downloads every target's SPOC 2-min light curves into the lightkurve cache in the
background while Phase-1 label work proceeds; the notebook's own download_all() then hits the cache.

Resume semantics: done/no_data TICs are skipped on --resume without re-verifying the cache — the
notebook re-calls download_all() itself, so a stale cache entry degrades to a re-download there, never
to silent data loss. error rows are always retried.

Run (astro env, from repo root):
    python src/qc/prefetch_eb_review_lc.py --limit 2    # smoke
    python src/qc/prefetch_eb_review_lc.py --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import call_with_retry, find_project_root, setup_logging

REVIEW_VERDICTS = ("CONTAMINATION_CONFIRMED_PLANET", "SUSPECT_CONFIRMED_FULL_LC")
ANCHOR_TIC = 233169434  # Prof's counter-example, panel 33 of the review notebook


def review_targets(audit_csv: Path) -> list[int]:
    """The 33 review TICs: both drop-candidate verdict groups plus the anchor, ascending order."""
    audit = pd.read_csv(audit_csv)
    tics = audit.loc[audit["final_verdict"].isin(REVIEW_VERDICTS), "tic_id"].astype(int).tolist()
    if ANCHOR_TIC not in tics:
        tics.append(ANCHOR_TIC)
    return sorted(tics)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prefetch SPOC light curves for the EB review notebook.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N targets (smoke).")
    ap.add_argument("--resume", action="store_true", help="Skip done/no_data rows; retry error rows.")
    ap.add_argument("--audit-csv", default=None, help="Default: labels/qc/eb_villanova_audit.csv")
    args = ap.parse_args()

    root = find_project_root()
    audit_csv = Path(args.audit_csv) if args.audit_csv else root / "labels" / "qc" / "eb_villanova_audit.csv"
    progress_csv = root / "labels" / "qc" / "prefetch_eb_review_progress.csv"
    logger = setup_logging(root / "qc_prefetch_eb_review.log", "prefetch_eb_review")

    assert audit_csv.exists(), f"missing {audit_csv}; run src/qc/eb_villanova_audit.py first"
    tics = review_targets(audit_csv)
    if args.limit:
        tics = tics[: args.limit]
    logger.info(f"{len(tics)} review targets (verdicts {REVIEW_VERDICTS} + anchor {ANCHOR_TIC})")

    if args.resume and progress_csv.exists():
        prog = pd.read_csv(progress_csv)
        skip = set(prog.loc[prog["status"].isin(["done", "no_data"]), "tic_id"].astype(int))
    else:
        prog = pd.DataFrame(columns=["tic_id", "status", "error_msg", "n_lc"])
        skip = set()
    prog_records: list[dict] = prog.to_dict("records")

    import lightkurve as lk
    from tqdm.auto import tqdm

    def flush() -> None:
        pd.DataFrame(prog_records).drop_duplicates("tic_id", keep="last").to_csv(progress_csv, index=False)

    n_done = n_err = 0
    try:
        for tic in tqdm(tics, desc="prefetch EB review LCs", total=len(tics)):
            if tic in skip:
                continue
            try:
                def _search():
                    return lk.search_lightcurve(f"TIC {tic}", author="SPOC", cadence=120)
                sr = call_with_retry(_search, f"search TIC {tic}", logger)
                if len(sr) == 0:
                    prog_records.append({"tic_id": tic, "status": "no_data", "error_msg": "", "n_lc": 0})
                    flush()
                    continue

                def _download():
                    return sr.download_all(quality_bitmask="none")  # keep all rows; notebook needs real gaps
                lcs = call_with_retry(_download, f"download TIC {tic}", logger)
                n_lc = len(lcs) if lcs is not None else 0
                prog_records.append({"tic_id": tic, "status": "done", "error_msg": "", "n_lc": n_lc})
                n_done += 1
                logger.info(f"TIC {tic}: cached {n_lc} sector light curves")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.exception(f"failed on TIC {tic}")
                prog_records.append({"tic_id": tic, "status": "error", "error_msg": repr(exc)[:500], "n_lc": 0})
                n_err += 1
            flush()
    except KeyboardInterrupt:
        logger.warning("interrupted; progress flushed")
        flush()
        return 130

    flush()
    logger.info(f"prefetch done: {n_done} downloaded, {n_err} errors, {len(skip)} skipped (resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
