"""Task B — Villanova EB purity audit (two-tier, report-only; ADR-0007 source stays intact).

Trust in the Villanova EB labels broke when Prof. Theissen showed an eb=1 star whose light curve has
no eclipse. This audit checks every Villanova positive two ways and writes a per-star verdict table;
it does NOT change any label (eb in v2 stays byte-identical to v1 until a deliberate follow-up).

Tier A (authoritative human vetting, only exists for EBs that are also TOIs): join to the ExoFOP TOI
cache; `TESS Disposition == 'EB'` corroborates, `TFOPWG Disposition in {CP, KP}` means the dip is a
confirmed planet --> contamination suspect.

Tier B (the Prof's criterion at scale, works for all windowed EBs): fold the star's corpus cadences
at the Villanova period and score an eclipse-depth SNR on the phase-binned median curve. A flat
folded curve (low SNR) with well-sampled phase = no visible eclipse = suspect. Poorly-sampled phase
or missing period = unverifiable, never suspect (the eclipse could hide in the unseen phase).

Pass 2 (--pass2): the corpus gap-guard discards cadences near gaps, which can manufacture a false
suspect. Re-download the FULL SPOC 2-min PDCSAP for pass-1 suspects only, re-fold on everything, and
clear stars whose eclipse was hiding in discarded cadences. Resume-safe (progress CSV, tenacity).

Run (astro env, from repo root):
    python src/qc/eb_villanova_audit.py                 # tier A + tier B pass 1
    python src/qc/eb_villanova_audit.py --limit 30      # smoke
    python src/qc/eb_villanova_audit.py --pass2         # re-download + re-fold pass-1 suspects
    python src/qc/eb_villanova_audit.py --pass2 --resume
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
from qc_common import call_with_retry, find_project_root, setup_logging

PROF_COUNTER_EXAMPLE_TIC = 233169434
_NPZ_TIC_RE = re.compile(r"^TIC(\d{10})_.*\.npz$")
PLANET_DISPS = {"CP", "KP"}
MIN_PHASE_FILLED = 0.8  # below this an unseen phase range could hide the eclipse --> unverifiable
MIN_POINTS_PER_BIN = 3
N_PHASE_BINS = 200


def index_npz_by_tic(seq_dir: Path, logger) -> dict[int, list[Path]]:
    """One pass over processed/sequences --> {tic: [npz paths]}; avoids per-TIC globs of a 400k-file dir."""
    idx: dict[int, list[Path]] = {}
    n = 0
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_TIC_RE.match(entry.name)
            if m:
                idx.setdefault(int(m.group(1)), []).append(Path(entry.path))
                n += 1
    logger.info(f"indexed {n} .npz across {len(idx)} TICs")
    return idx


def eclipse_snr(times: np.ndarray, flux: np.ndarray, period: float) -> dict:
    """
    Score how confidently the folded light curve shows coherent variability at the Villanova period.
    Fold (epoch unknown --> eclipse may sit at any phase), take the median flux per phase bin, and
    compare the deepest bin against the NOISE OF A BIN MEDIAN (within-bin point scatter / sqrt(n)).
    This keeps smooth contact-binary (EW) sinusoids high-SNR: their structure is signal, not noise —
    scoring against across-bin scatter would let the curve's own shape mask the detection.
    A flat folded curve scores ~3 (extreme of ~200 noise bins); any real EB scores far higher.
    Caveat (documented, report-only): this detects periodicity at the catalog period, not eclipse
    morphology — a pulsator whose period landed in Villanova would still score high. `dip_asymmetry`
    (depth below baseline / rise above it) is reported as a weak morphology hint: EA >> 1, EW ~ 1-2.
    Returns snr, depth (per-segment MAD units), dip_asymmetry, phase_filled, n_bins_filled.
    """
    t = times.astype(np.float64)
    phase = np.mod(t - t.min(), period) / period  # [0, 1)
    bin_idx = np.minimum((phase * N_PHASE_BINS).astype(int), N_PHASE_BINS - 1)

    bin_medians = []
    bin_counts = []
    resid = []
    for b in range(N_PHASE_BINS):
        vals = flux[bin_idx == b]
        if len(vals) >= MIN_POINTS_PER_BIN:
            med = np.median(vals)
            bin_medians.append(med)
            bin_counts.append(len(vals))
            resid.append(vals - med)  # within-bin residuals --> pure photometric noise
    n_filled = len(bin_medians)
    filled = n_filled / N_PHASE_BINS
    if n_filled < 20:
        return {"eclipse_snr": np.nan, "depth": np.nan, "dip_asymmetry": np.nan,
                "phase_filled": filled, "n_bins_filled": n_filled}

    bm = np.asarray(bin_medians)
    baseline = np.median(bm)
    depth = baseline - bm.min()  # eclipses are dips below the baseline
    rise = max(float(bm.max() - baseline), 1e-6)
    r = np.concatenate(resid)
    sigma_pt = 1.4826 * np.median(np.abs(r))  # 1.4826*MAD --> sigma for gaussian noise
    sigma_bin = max(sigma_pt / np.sqrt(np.median(bin_counts)), 1e-6)  # noise of one bin's median
    return {"eclipse_snr": depth / sigma_bin, "depth": depth, "dip_asymmetry": depth / rise,
            "phase_filled": filled, "n_bins_filled": n_filled}


def load_corpus_cadences(npz_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (times, flux) over all a star's windows. Flux is already per-segment MAD-normalized."""
    ts = []
    fs = []
    for p in npz_paths:
        with np.load(p) as data:
            ts.append(data["times"].ravel())
            fs.append(data["windows"].ravel())
    return np.concatenate(ts), np.concatenate(fs)


def tier_a(eb_tics: list[int], exofop_csv: Path, logger) -> pd.DataFrame:
    """ExoFOP adjudication for the Villanova EBs that are TOIs; the rest have no row (no conflict possible)."""
    ex = pd.read_csv(exofop_csv)
    ex["_tic"] = pd.to_numeric(ex["TIC ID"], errors="coerce")
    ex = ex[ex["_tic"].notna()].copy()
    ex["_tic"] = ex["_tic"].astype(int)

    rows = []
    eb_set = set(eb_tics)
    hits = ex[ex["_tic"].isin(eb_set)]
    for tic, g in hits.groupby("_tic"):
        tess_disps = sorted(set(g["TESS Disposition"].dropna().astype(str)))
        tfop_disps = sorted(set(g["TFOPWG Disposition"].dropna().astype(str)))
        rows.append({
            "tic_id": int(tic),
            "in_exofop": True,
            "exofop_tess_disp": ";".join(tess_disps),
            "exofop_tfopwg_disp": ";".join(tfop_disps),
            "exofop_toi": ";".join(g["TOI"].astype(str)),
        })
    out = pd.DataFrame(rows)
    logger.info(f"tier A: {len(out)} of {len(eb_tics)} Villanova EBs appear in the ExoFOP TOI list")
    return out


def fetch_exofop_target_json(tic: int, logger) -> str:
    """Targeted single-TIC ExoFOP lookup (works even when the star is not a TOI). Returns a short summary string."""
    import requests

    def _q():
        r = requests.get(f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}&json", timeout=120)
        r.raise_for_status()
        return r.json()

    try:
        j = call_with_retry(_q, f"ExoFOP target {tic}", logger)
    except Exception as e:
        return f"fetch failed: {type(e).__name__}: {e}"
    tois = j.get("tois", [])
    parts = []
    for t in tois:
        parts.append(f"TOI {t.get('toi')}: tfopwg={t.get('disposition')!r} tess={t.get('tess_disposition')!r}")
    if not parts:
        return "in ExoFOP, no TOI entries (never a planet candidate)"
    return " | ".join(parts)


def run_pass1(args, root: Path, logger) -> None:
    labels = pd.read_csv(root / "labels" / "variability_labels_star.csv")
    labels["eb"] = pd.to_numeric(labels["eb"], errors="coerce").fillna(0).astype(int)
    eb = labels[labels["eb"] == 1][["tic_id", "eb_period"]].copy()
    eb["tic_id"] = eb["tic_id"].astype(int)
    logger.info(f"Villanova EB positives (v1): {len(eb)}")

    npz_index = index_npz_by_tic(root / "processed" / "sequences", logger)
    exofop = tier_a(eb["tic_id"].tolist(), root / "labels" / "qc" / "toi_exofop.csv", logger)

    from tqdm.auto import tqdm

    records = []
    todo = eb.head(args.limit) if args.limit else eb
    for _, row in tqdm(todo.iterrows(), desc="Task B eclipse fold", total=len(todo)):
        tic = int(row["tic_id"])
        period = float(row["eb_period"]) if pd.notna(row["eb_period"]) else np.nan
        npz_paths = npz_index.get(tic, [])
        rec = {"tic_id": tic, "eb_period": period, "n_npz": len(npz_paths)}
        if not npz_paths:
            rec["tier_b_verdict"] = "NO_WINDOWS"
        elif not np.isfinite(period) or period <= 0:
            rec["tier_b_verdict"] = "NO_PERIOD"
        else:
            times, flux = load_corpus_cadences(npz_paths)
            rec["n_cad"] = len(times)
            rec.update(eclipse_snr(times, flux, period))
            snr = rec.get("eclipse_snr", np.nan)
            if not np.isfinite(snr):
                rec["tier_b_verdict"] = "TOO_FEW_BINS"
            elif rec["phase_filled"] < MIN_PHASE_FILLED:
                rec["tier_b_verdict"] = "POORLY_SAMPLED_PHASE"
            elif snr >= args.snr_threshold:
                rec["tier_b_verdict"] = "ECLIPSE_SEEN"
            else:
                rec["tier_b_verdict"] = "SUSPECT_NO_ECLIPSE"
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.merge(exofop, on="tic_id", how="left")
    df["in_exofop"] = df["in_exofop"] == True  # noqa: E712  (fillna(False) downcast is deprecated)

    def _final(row) -> str:
        tfop = str(row.get("exofop_tfopwg_disp") or "")
        planet_hit = False
        for d in tfop.split(";"):
            if d.strip() in PLANET_DISPS:
                planet_hit = True
        if planet_hit:
            return "CONTAMINATION_CONFIRMED_PLANET"  # Villanova calls it EB, TFOPWG confirmed a planet
        tess = str(row.get("exofop_tess_disp") or "")
        if "EB" in tess.split(";"):
            return "CORROBORATED_EXOFOP_EB"
        return str(row["tier_b_verdict"])

    finals = []
    for _, row in df.iterrows():
        finals.append(_final(row))
    df["final_verdict"] = finals

    out_csv = root / "labels" / "qc" / "eb_villanova_audit.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv} ({len(df)} rows)")

    logger.info("=" * 68)
    logger.info("Task B — Villanova EB audit, pass 1")
    logger.info("=" * 68)
    vc = df["final_verdict"].value_counts()
    for verdict, cnt in vc.items():
        logger.info(f"  {verdict}: {cnt}")
    checkable = df[df["tier_b_verdict"].isin(["ECLIPSE_SEEN", "SUSPECT_NO_ECLIPSE"])]
    n_suspect = int((df["final_verdict"] == "SUSPECT_NO_ECLIPSE").sum())
    if len(checkable):
        logger.info(f"tier-B checkable stars: {len(checkable)}; suspects: {n_suspect} "
                    f"({n_suspect / len(checkable):.1%} of checkable)")
    snr_ok = df[df["final_verdict"].isin(["ECLIPSE_SEEN", "CORROBORATED_EXOFOP_EB"])]["eclipse_snr"].dropna()
    snr_bad = df[df["final_verdict"] == "SUSPECT_NO_ECLIPSE"]["eclipse_snr"].dropna()
    if len(snr_ok) and len(snr_bad):
        logger.info(f"eclipse_snr corroborated median: {snr_ok.median():.1f}; suspect median: {snr_bad.median():.1f}")

    # the Prof's counter-example, always reported explicitly
    anchor = df[df["tic_id"] == PROF_COUNTER_EXAMPLE_TIC]
    if len(anchor):
        a = anchor.iloc[0]
        logger.info(f"ANCHOR TIC {PROF_COUNTER_EXAMPLE_TIC}: period={a.get('eb_period')} "
                    f"snr={a.get('eclipse_snr')} asym={a.get('dip_asymmetry')} "
                    f"filled={a.get('phase_filled')} verdict={a.get('final_verdict')}")
    elif int(PROF_COUNTER_EXAMPLE_TIC) in set(eb["tic_id"].astype(int)):
        logger.info(f"ANCHOR TIC {PROF_COUNTER_EXAMPLE_TIC}: eb=1 in v1 but outside this --limit slice")
    else:
        logger.info(f"ANCHOR TIC {PROF_COUNTER_EXAMPLE_TIC}: not a Villanova eb=1 star in v1 labels")
    logger.info(f"ANCHOR ExoFOP targeted lookup: {fetch_exofop_target_json(PROF_COUNTER_EXAMPLE_TIC, logger)}")
    logger.info("Task B pass 1 done.")


def run_pass2(args, root: Path, logger) -> None:
    """
    Re-download the full SPOC 2-min PDCSAP for pass-1 suspects and re-fold on ALL cadences.
    Clears stars whose eclipse fell in cadences the corpus gap-guard discarded.
    Per-sector median/MAD normalization before combining, mirroring the corpus convention.
    """
    audit_csv = root / "labels" / "qc" / "eb_villanova_audit.csv"
    assert audit_csv.exists(), "run pass 1 first"
    df = pd.read_csv(audit_csv)
    suspects = df[df["final_verdict"] == "SUSPECT_NO_ECLIPSE"].copy()
    logger.info(f"pass 2: {len(suspects)} suspects to re-check with full PDCSAP")

    prog_csv = root / "labels" / "qc" / "eb_audit_pass2_progress.csv"
    if args.resume and prog_csv.exists():
        prog = pd.read_csv(prog_csv)
    else:
        prog = pd.DataFrame(columns=["tic_id", "status", "error_msg"])
    done = set(prog.loc[prog["status"] == "done", "tic_id"].astype(int))

    import lightkurve as lk
    from tqdm.auto import tqdm

    results = {}
    prog_records = prog.to_dict("records")
    todo = suspects.head(args.limit) if args.limit else suspects
    for i, (_, row) in enumerate(tqdm(todo.iterrows(), desc="Task B pass 2", total=len(todo)), start=1):
        tic = int(row["tic_id"])
        period = float(row["eb_period"])
        if tic in done:
            continue
        try:
            def _search():
                return lk.search_lightcurve(f"TIC {tic}", author="SPOC", cadence=120)
            sr = call_with_retry(_search, f"search TIC {tic}", logger)
            if len(sr) == 0:
                prog_records.append({"tic_id": tic, "status": "no_data", "error_msg": ""})
                continue

            def _download():
                return sr.download_all(quality_bitmask="none")  # keep all rows; gap detection needs them
            lcs = call_with_retry(_download, f"download TIC {tic}", logger)

            ts = []
            fs = []
            for lc in lcs:
                cols = {c.lower(): c for c in lc.colnames}  # MAST casing is inconsistent
                flux_col = cols.get("pdcsap_flux")
                if flux_col is None:
                    continue
                t = np.asarray(lc["time"].value, dtype=np.float64)
                f = np.asarray(lc[flux_col].value, dtype=np.float64)
                q = np.asarray(lc["quality"].value) if "quality" in cols else np.zeros(len(f))
                f[q != 0] = np.nan
                keep = np.isfinite(t) & np.isfinite(f)
                t = t[keep]
                f = f[keep]
                if len(f) < 100:
                    continue
                med = np.median(f)
                mad = np.median(np.abs(f - med))
                if mad <= 0:
                    continue
                ts.append(t)
                fs.append((f - med) / (1.4826 * mad))  # per-sector MAD normalization, corpus convention
            if not ts:
                prog_records.append({"tic_id": tic, "status": "no_data", "error_msg": "no usable PDCSAP"})
                continue
            res = eclipse_snr(np.concatenate(ts), np.concatenate(fs), period)
            results[tic] = res
            prog_records.append({"tic_id": tic, "status": "done", "error_msg": ""})
        except Exception as e:
            logger.error(f"TIC {tic}: {type(e).__name__}: {e}")
            prog_records.append({"tic_id": tic, "status": "error", "error_msg": repr(e)[:300]})
        if i % 20 == 0:
            pd.DataFrame(prog_records).to_csv(prog_csv, index=False)
    pd.DataFrame(prog_records).to_csv(prog_csv, index=False)

    # merge pass-2 numbers into the audit table and upgrade verdicts where the full LC shows the eclipse
    df["pass2_snr"] = np.nan
    df["pass2_phase_filled"] = np.nan
    for tic, res in results.items():
        mask = df["tic_id"] == tic
        df.loc[mask, "pass2_snr"] = res["eclipse_snr"]
        df.loc[mask, "pass2_phase_filled"] = res["phase_filled"]
        if np.isfinite(res["eclipse_snr"]) and res["eclipse_snr"] >= args.snr_threshold:
            df.loc[mask, "final_verdict"] = "ECLIPSE_SEEN_FULL_LC"  # corpus gap artifact, cleared
        elif np.isfinite(res["eclipse_snr"]) and res["phase_filled"] >= MIN_PHASE_FILLED:
            df.loc[mask, "final_verdict"] = "SUSPECT_CONFIRMED_FULL_LC"
    df.to_csv(audit_csv, index=False)
    logger.info(f"updated {audit_csv}")

    vc = df["final_verdict"].value_counts()
    logger.info("Task B pass-2 verdicts:")
    for verdict, cnt in vc.items():
        logger.info(f"  {verdict}: {cnt}")
    logger.info("Task B pass 2 done.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Task B: Villanova EB purity audit (report-only).")
    ap.add_argument("--pass2", action="store_true", help="Re-download full PDCSAP for pass-1 suspects.")
    ap.add_argument("--resume", action="store_true", help="pass 2: skip TICs already done in progress CSV.")
    ap.add_argument("--limit", type=int, default=None, help="Only first N stars (smoke).")
    ap.add_argument("--snr-threshold", type=float, default=5.0,
                    help="Eclipse-depth SNR at/above which the folded curve counts as showing an eclipse.")
    args = ap.parse_args()

    root = find_project_root()
    logger = setup_logging(root / "qc_eb_villanova_audit.log", "eb_villanova_audit")
    (root / "labels" / "qc").mkdir(parents=True, exist_ok=True)

    if args.pass2:
        run_pass2(args, root, logger)
    else:
        run_pass1(args, root, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
