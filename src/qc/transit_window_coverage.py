"""Task C — transit window-coverage diagnostic (gates roadmap Phase 1).

Question: when we broadcast a star-level `transit=1` label to every packed window of that TIC, what
fraction of those windows actually contain an in-transit cadence? If most windows show no transit, the
per-star mu the linear probe sees is dominated by out-of-transit flux → the label is mostly noise.

Method: for each v1 transit-positive TIC that has >=1 .npz window, fold every cadence's real BTJD time
onto the TOI ephemeris (period pl_orbper, epoch pl_tranmid[BJD]->BTJD, duration pl_trandurh) and mark
in-transit cadences. A window is "covered" iff it holds >=1 in-transit cadence. Coverage is reported at
three granularities:
  - 256-cadence window  -> THE GATE (atomic encoding unit of exp01/02/03). median < 10% fires Phase 1.
  - 1024-cadence window -> v1 exp00 baseline unit (reported, not gated).
  - per-.npz segment     -> contiguous gap-free run == the Phase-1 (tic_id, seg_id) label key (reported).

Ephemeris handling (locked in grilling): fold needs P + T0; a TIC lacking either is 'unfoldable'
(excluded from the gated median, counted). Missing duration is imputed with the population median
duration (counted). Multi-planet TICs: a cadence is in-transit if ANY of the TIC's whitelisted TOIs
covers it.

Run (astro env, from repo root; needs labels/qc/toi_nasa.csv from fetch_toi_enriched.py):
    python src/qc/transit_window_coverage.py
    python src/qc/transit_window_coverage.py --limit 20   # smoke on first 20 transit TICs
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import find_project_root, setup_logging

BTJD_OFFSET = 2457000.0  # BTJD = BJD - 2457000 (TESS convention); pl_tranmid is BJD(TDB)
_NPZ_TIC_RE = re.compile(r"^TIC(\d{10})_.*\.npz$")


def index_npz_by_tic(seq_dir: Path, logger) -> dict[int, list[Path]]:
    """One pass over the sequences dir -> {tic: [npz paths]}. Avoids re-globbing a 400k-file dir per TIC."""
    idx: dict[int, list[Path]] = {}
    n = 0
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_TIC_RE.match(entry.name)
            if m:
                idx.setdefault(int(m.group(1)), []).append(Path(entry.path))
                n += 1
    logger.info(f"indexed {n} .npz across {len(idx)} TICs in {seq_dir}")
    return idx
WHITELIST = {"CP", "KP", "PC", "APC"}
NATIVE = 1024
SUB = 256
GATE_THRESHOLD = 0.10


def load_ephemerides(nasa_csv: Path, logger) -> dict[int, list[tuple[float, float, float]]]:
    """{tic: [(P_days, T0_btjd, dur_days_or_nan), ...]} over whitelisted TOIs with finite P and T0."""
    df = pd.read_csv(nasa_csv)
    df["tfopwg_disp"] = df["tfopwg_disp"].astype(str).str.strip().str.upper()
    df = df[df["tfopwg_disp"].isin(WHITELIST)].copy()
    P = pd.to_numeric(df["pl_orbper"], errors="coerce")
    T0 = pd.to_numeric(df["pl_tranmid"], errors="coerce")
    dur_h = pd.to_numeric(df["pl_trandurh"], errors="coerce")
    eph: dict[int, list[tuple[float, float, float]]] = {}
    n_rows = 0
    for tid, p, t0, dh in zip(df["tid"], P, T0, dur_h):
        if pd.isna(tid) or pd.isna(p) or pd.isna(t0) or float(p) <= 0:
            continue  # need positive P + finite T0 to fold
        dur_days = float(dh) / 24.0 if pd.notna(dh) and float(dh) > 0 else float("nan")
        eph.setdefault(int(tid), []).append((float(p), float(t0) - BTJD_OFFSET, dur_days))
        n_rows += 1
    logger.info(f"ephemerides: {n_rows} whitelisted TOI rows over {len(eph)} TICs (finite P+T0)")
    return eph


def load_ephemerides_with_disp(
    nasa_csv: Path, logger, whitelist: set[str] = WHITELIST
) -> dict[int, list[tuple[float, float, float, str]]]:
    """Like `load_ephemerides` but keeps the TFOPWG disposition on every TOI row.

    Returns {tic: [(P_days, T0_btjd, dur_days_or_nan, disp), ...]} over `whitelist` dispositions
    with finite P (>0) and T0. `disp` is the upper-cased tfopwg_disp (CP/KP/PC/APC). The window
    annotation path loads the FULL whitelist here and lets the notebook select any disposition
    subset at eval time, so the same fold serves the KP+CP and KP+CP+PC knobs.
    """
    df = pd.read_csv(nasa_csv)
    df["tfopwg_disp"] = df["tfopwg_disp"].astype(str).str.strip().str.upper()
    df = df[df["tfopwg_disp"].isin(whitelist)].copy()
    P = pd.to_numeric(df["pl_orbper"], errors="coerce")
    T0 = pd.to_numeric(df["pl_tranmid"], errors="coerce")
    dur_h = pd.to_numeric(df["pl_trandurh"], errors="coerce")
    eph: dict[int, list[tuple[float, float, float, str]]] = {}
    n_rows = 0
    for tid, p, t0, dh, disp in zip(df["tid"], P, T0, dur_h, df["tfopwg_disp"]):
        if pd.isna(tid) or pd.isna(p) or pd.isna(t0) or float(p) <= 0:
            continue  # need positive P + finite T0 to fold
        dur_days = float(dh) / 24.0 if pd.notna(dh) and float(dh) > 0 else float("nan")
        eph.setdefault(int(tid), []).append((float(p), float(t0) - BTJD_OFFSET, dur_days, str(disp)))
        n_rows += 1
    logger.info(f"ephemerides+disp: {n_rows} TOI rows over {len(eph)} TICs "
                f"(whitelist={sorted(whitelist)}, finite P+T0)")
    return eph


def _median_duration(eph: dict) -> float:
    durs = [row[2] for rows in eph.values() for row in rows if np.isfinite(row[2])]  # row=(P,T0,dur[,disp])
    return float(np.median(durs)) if durs else (2.0 / 24.0)  # fallback 2 h


def in_transit_mask(times: np.ndarray, rows: list[tuple[float, float, float]], median_dur: float) -> np.ndarray:
    """Boolean mask over `times` (any shape): True where a cadence falls inside any TOI's transit."""
    t = times.astype(np.float64)
    mask = np.zeros(t.shape, dtype=bool)
    for (P, T0, dur) in rows:
        d = dur if np.isfinite(dur) else median_dur
        phase = np.mod(t - T0 + 0.5 * P, P) - 0.5 * P  # [-P/2, P/2)
        mask |= np.abs(phase) <= 0.5 * d
    return mask


def coverage_for_tic(
    npz_paths: list[Path], rows: list[tuple[float, float, float]], median_dur: float
) -> dict:
    """Coverage at 256-window / 1024-window / segment granularity, aggregated over a TIC's .npz files."""
    w1024_tot = w1024_cov = 0
    w256_tot = w256_cov = 0
    seg_tot = seg_cov = 0
    cad_tot = cad_cov = 0
    for p in npz_paths:
        with np.load(p) as data:
            times = data["times"]  # (N, 1024) BTJD
        if times.ndim != 2 or times.shape[1] != NATIVE:
            # unexpected native length; skip defensively (v1 corpus is 1024)
            continue
        n = times.shape[0]
        m1024 = in_transit_mask(times, rows, median_dur)  # (N, 1024)
        # 1024-window coverage
        w1024_tot += n
        w1024_cov += int(m1024.any(axis=1).sum())
        # 256-window coverage: split each 1024 row into 4 contiguous blocks
        m256 = m1024.reshape(n, NATIVE // SUB, SUB)  # (N, 4, 256)
        w256_tot += n * (NATIVE // SUB)
        w256_cov += int(m256.any(axis=2).sum())
        # segment (this .npz = one contiguous run) coverage
        seg_tot += 1
        seg_cov += int(bool(m1024.any()))
        # cadence-level (sanity)
        cad_tot += m1024.size
        cad_cov += int(m1024.sum())
    return {
        "n_win1024": w1024_tot, "cov1024": (w1024_cov / w1024_tot) if w1024_tot else np.nan,
        "n_win256": w256_tot, "cov256": (w256_cov / w256_tot) if w256_tot else np.nan,
        "n_seg": seg_tot, "cov_seg": (seg_cov / seg_tot) if seg_tot else np.nan,
        "n_cad": cad_tot, "frac_cad_intransit": (cad_cov / cad_tot) if cad_tot else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Task C: transit window-coverage diagnostic.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N transit TICs (smoke).")
    ap.add_argument("--labels-csv", default=None, help="Default: labels/variability_labels_star.csv (v1).")
    ap.add_argument("--nasa-csv", default=None, help="Default: labels/qc/toi_nasa.csv")
    ap.add_argument("--sequences-dir", default=None, help="Default: processed/sequences")
    ap.add_argument("--out-dir", default=None, help="Default: labels/qc")
    args = ap.parse_args()

    root = find_project_root()
    labels_csv = Path(args.labels_csv) if args.labels_csv else root / "labels" / "variability_labels_star.csv"
    nasa_csv = Path(args.nasa_csv) if args.nasa_csv else root / "labels" / "qc" / "toi_nasa.csv"
    seq_dir = Path(args.sequences_dir) if args.sequences_dir else root / "processed" / "sequences"
    out_dir = Path(args.out_dir) if args.out_dir else root / "labels" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "qc_transit_window_coverage.log", "transit_window_coverage")

    assert nasa_csv.exists(), f"missing {nasa_csv}; run src/qc/fetch_toi_enriched.py first"
    eph = load_ephemerides(nasa_csv, logger)
    median_dur = _median_duration(eph)
    logger.info(f"population median transit duration = {median_dur * 24:.2f} h ({median_dur:.4f} d)")

    labels = pd.read_csv(labels_csv)
    labels["transit"] = pd.to_numeric(labels["transit"], errors="coerce").fillna(0).astype(int)
    transit_tics = labels.loc[labels["transit"] == 1, "tic_id"].astype(int).tolist()
    logger.info(f"v1 transit-positive TICs: {len(transit_tics)}")

    from tqdm.auto import tqdm

    npz_index = index_npz_by_tic(seq_dir, logger)

    records = []
    n_windowed = n_foldable = n_unfoldable = n_dur_imputed = 0
    tics = transit_tics[: args.limit] if args.limit else transit_tics
    for tic in tqdm(tics, desc="Task C coverage", total=len(tics)):
        npz_paths = sorted(npz_index.get(tic, []))
        if not npz_paths:
            continue  # no windows -> outside the probe-usable population
        n_windowed += 1
        rows = eph.get(tic)
        if not rows:
            n_unfoldable += 1
            records.append({"tic_id": tic, "n_toi": 0, "foldable": False, "dur_imputed": False,
                            "n_npz": len(npz_paths)})
            continue
        n_foldable += 1
        imputed = any(not np.isfinite(d) for (_, _, d) in rows)
        if imputed:
            n_dur_imputed += 1
        cov = coverage_for_tic(npz_paths, rows, median_dur)
        periods = [P for (P, _, _) in rows]
        records.append({
            "tic_id": tic, "n_toi": len(rows), "foldable": True, "dur_imputed": imputed,
            "n_npz": len(npz_paths), "P_min_d": min(periods), "P_max_d": max(periods), **cov,
        })

    df = pd.DataFrame(records)
    out_csv = out_dir / "transit_window_coverage.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv} ({len(df)} rows)")

    foldable = df[df["foldable"] == True]  # noqa: E712
    med256 = float(foldable["cov256"].median()) if len(foldable) else float("nan")
    med1024 = float(foldable["cov1024"].median()) if len(foldable) else float("nan")
    medseg = float(foldable["cov_seg"].median()) if len(foldable) else float("nan")

    logger.info("=" * 68)
    logger.info("Task C — transit window-coverage summary")
    logger.info("=" * 68)
    logger.info(f"v1 transit positives:            {len(transit_tics)}")
    logger.info(f"  with >=1 .npz window:          {n_windowed}")
    logger.info(f"  foldable (P+T0 present):       {n_foldable}")
    logger.info(f"  unfoldable (no whitelist eph): {n_unfoldable}")
    logger.info(f"  duration imputed (median):     {n_dur_imputed}")
    logger.info(f"median coverage — 256-window:    {med256:.4f}   <-- GATE")
    logger.info(f"median coverage — 1024-window:   {med1024:.4f}")
    logger.info(f"median coverage — segment:       {medseg:.4f}")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        logger.info(f"  cov256 q{int(q*100):02d}: {float(foldable['cov256'].quantile(q)):.4f}")

    fired = np.isfinite(med256) and med256 < GATE_THRESHOLD
    logger.info(f"GATE (median 256-window coverage < {GATE_THRESHOLD:.0%}): "
                f"{'FIRES -> Phase 1' if fired else 'does not fire (star-level probe stands)'}")

    # coverage vs period bins (expect coverage ~ window/P, i.e. lower coverage at longer P)
    if len(foldable):
        pb = pd.cut(foldable["P_min_d"], [0, 1, 3, 5, 10, 1e9],
                    labels=["<1d", "1-3d", "3-5d", "5-10d", ">=10d"])
        by = foldable.groupby(pb, observed=True)["cov256"].agg(["count", "median"])
        logger.info(f"cov256 by shortest-period bin:\n{by}")

    # histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, col, name, med in ((ax[0], "cov256", "256-window (GATE)", med256),
                                  (ax[1], "cov1024", "1024-window", med1024)):
            vals = foldable[col].dropna()
            a.hist(vals, bins=40, range=(0, 1), color="#4472c4", edgecolor="white")
            a.axvline(med, color="crimson", ls="--", label=f"median={med:.3f}")
            a.axvline(GATE_THRESHOLD, color="green", ls=":", label=f"gate={GATE_THRESHOLD:.0%}")
            a.set_xlabel(f"per-star transit coverage ({name})")
            a.set_ylabel("n stars")
            a.legend()
        fig.suptitle(f"Transit window coverage (n_foldable={n_foldable})")
        fig.tight_layout()
        fig_path = out_dir / "transit_window_coverage_hist.png"
        fig.savefig(fig_path, dpi=120)
        logger.info(f"wrote {fig_path}")
    except Exception as e:
        logger.warning(f"histogram skipped: {type(e).__name__}: {e}")

    logger.info("Task C done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
