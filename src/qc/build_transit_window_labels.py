"""Phase 1 — three-state window-level transit labels aligned to the packed 256-window rows.

Plan: docs/plans/2026-07-16-phase1-window-level-transit-labels.md (D1/D2). The star-level transit=1
broadcast is mostly noise at window granularity (Task C: median 256-window coverage 8.3%), so this
script folds every packed window's real cadence times onto the TOI ephemeris and emits one label per
packed window row of every v1 transit-positive star:

  label =  1  window holds >=1 in-transit cadence (|phase| <= 0.5*dur, any whitelisted TOI)
  label =  0  clean negative
  label = -1  quarantine, excluded from fit AND eval:
              - near_transit: no in-transit cadence but >=1 cadence with |phase| <= NEAR_FACTOR*dur
                (absorbs T0 drift / duration error; factor widens when the duration was imputed)
              - unfoldable: the star has no whitelisted TOI ephemeris (KEEP_UNVERIFIABLE bucket) --
                its transits cannot be placed, so no window of it may serve as either class

Windows of non-transit stars are label 0 by construction and are NOT materialized here; an absent
(split, row) key means clean negative.

Alignment contract (verified fail-loud per segment): the packed index row's (tic_id, sector, seg_idx)
maps to exactly one processed/sequences npz (the packed subset has zero duplicate seg_ids); replaying
pack.py's absmax guard + 1024->256 subdivision on the npz reproduces n_win exactly, so window j of the
segment is packed row row_start+j. A sampled exact-float32 flux comparison against {split}_windows.dat
re-verifies the mapping on every run.

Run (astro env, from repo root; needs labels/qc/toi_nasa.csv from fetch_toi_enriched.py):
    python src/qc/build_transit_window_labels.py --limit 20   # smoke on first 20 transit TICs
    python src/qc/build_transit_window_labels.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import find_project_root, setup_logging
from transit_window_coverage import _median_duration, load_ephemerides, load_ephemerides_with_disp

NEAR_FACTOR = 1.5          # quarantine band: |phase| <= 1.5*dur from mid-transit (no in-transit cadence)
NEAR_FACTOR_IMPUTED = 2.0  # wider band when the TOI duration was imputed with the population median
FLUX_CHECK_PER_SPLIT = 50  # segments per split re-verified against the packed .dat (exact float32)

DISP_PRIORITY = {"CP": 0, "KP": 1, "PC": 2, "APC": 3}  # lower = higher confidence


def _best_disp(disps: list[str]) -> str:
    """Highest-confidence disposition (CP>KP>PC>APC) present in `disps`; '' if none."""
    ds = [d for d in disps if d in DISP_PRIORITY]
    return min(ds, key=lambda d: DISP_PRIORITY[d]) if ds else ""


def annotate_segment(
    times: np.ndarray, toi_rows: list[tuple[float, float, float, str]], median_dur: float, near_factor: float
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    """Per-window (full_best, part_best, near_best, n_intransit_cad) for one segment.

    `times` is (n_win, window) BTJD, in packed row order. Each TOI is classified per window into its
    STRONGEST overlap kind and contributes its disposition to that bucket:
      full    -> a whole transit interval [mid-0.5d, mid+0.5d] lies inside the window's [t0, t_last]
                 (requires a KNOWN duration; an imputed dur can never mint a `full`, per Q3a)
      partial -> >=1 in-transit cadence (|phase| <= 0.5d) but not fully contained (window-edge clip)
      near    -> a cadence within near_factor*d of mid-transit but none in-transit (buffer only)
    `*_best` = the best disposition among the TOIs that landed in that bucket ('' if empty). A window
    with no bucket for any TOI is a clean negative; the caller flags unfoldable stars separately.
    """
    n_win = times.shape[0]
    t = times.astype(np.float64)              # (n_win, window)
    t0 = t[:, 0]                              # (n_win,) window start time
    tL = t[:, -1]                            # (n_win,) window end time
    full: list[list[str]] = [[] for _ in range(n_win)]
    part: list[list[str]] = [[] for _ in range(n_win)]
    near: list[list[str]] = [[] for _ in range(n_win)]
    n_in = np.zeros(n_win, dtype=int)
    for (P, T0, dur, disp) in toi_rows:
        d_known = np.isfinite(dur)
        d = dur if d_known else median_dur
        phase = np.mod(t - T0 + 0.5 * P, P) - 0.5 * P     # (n_win, window) days from nearest mid
        in_cad = np.abs(phase) <= 0.5 * d                  # (n_win, window)
        has_in = in_cad.any(axis=1)                        # (n_win,)
        has_near = (np.abs(phase) <= near_factor * d).any(axis=1)
        n_in += in_cad.sum(axis=1)
        if d_known:  # full containment: exists a transit mid with the whole interval inside [t0, tL]
            k_lo = np.floor((t0 - T0) / P).astype(int) - 1
            k_hi = np.ceil((tL - T0) / P).astype(int) + 1
            is_full = np.zeros(n_win, dtype=bool)
            for w in range(n_win):
                for k in range(int(k_lo[w]), int(k_hi[w]) + 1):
                    mid = T0 + k * P
                    if mid - 0.5 * d >= t0[w] and mid + 0.5 * d <= tL[w]:
                        is_full[w] = True
                        break
        else:
            is_full = np.zeros(n_win, dtype=bool)
        for w in range(n_win):
            if is_full[w]:
                full[w].append(disp)
            elif has_in[w]:
                part[w].append(disp)
            elif has_near[w]:
                near[w].append(disp)
    return ([_best_disp(x) for x in full], [_best_disp(x) for x in part],
            [_best_disp(x) for x in near], n_in)


def build_annotations(root: Path, packed: Path, seq_dir: Path, labels_csv: Path, nasa_csv: Path,
                      out_path: Path, whitelist: set[str], limit: int | None, logger) -> int:
    """Emit the rich per-window annotation parquet (full_best/part_best/near_best/n_intransit_cad/
    unfoldable) over the full `whitelist`. No 0/1/-1 label is baked here: the notebook derives it per
    knob (disposition subset x include-partial). Reuses the label path's verified alignment + exact
    float32 flux re-check so the annotation rows map to the same packed windows."""
    manifest = json.loads((packed / "pack_manifest.json").read_text())
    window = int(manifest["window"])
    max_absmax = float(manifest["max_absmax"])
    logger.info(f"packed: {packed} (window={window}, max_absmax={max_absmax}); whitelist={sorted(whitelist)}")

    eph = load_ephemerides_with_disp(nasa_csv, logger, whitelist)
    median_dur = _median_duration(eph)
    logger.info(f"population median transit duration = {median_dur * 24:.2f} h")

    labels = pd.read_csv(labels_csv)
    labels["transit"] = pd.to_numeric(labels["transit"], errors="coerce").fillna(0).astype(int)
    transit_tics = set(labels.loc[labels["transit"] == 1, "tic_id"].astype(int))
    logger.info(f"v1 transit-positive TICs: {len(transit_tics)}")

    from tqdm.auto import tqdm

    npz_index = index_npz_by_segment(seq_dir, transit_tics, logger)
    rng = np.random.default_rng(0)  # script (not a plot cell): deterministic spot-check sample
    records: list[dict] = []
    n_flux_checked = n_unfoldable_stars = 0
    for split in ["train", "val", "test"]:
        index = pd.read_parquet(packed / f"{split}_index.parquet")
        total = int(index["n_win"].sum())
        dat = np.memmap(packed / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
        seg_rows = index[index["tic_id"].isin(transit_tics)].reset_index(drop=True)
        if limit:
            keep_tics = sorted(seg_rows["tic_id"].unique())[:limit]
            seg_rows = seg_rows[seg_rows["tic_id"].isin(keep_tics)].reset_index(drop=True)
        check_ids = set(rng.choice(len(seg_rows), size=min(FLUX_CHECK_PER_SPLIT, len(seg_rows)),
                                   replace=False).tolist()) if len(seg_rows) else set()
        seen_unfoldable: set[int] = set()
        for i, row in enumerate(tqdm(seg_rows.itertuples(index=False), desc=f"annot[{split}]", total=len(seg_rows))):
            tic = int(row.tic_id)
            toi_rows = eph.get(tic)
            times, flux = replay_segment(
                npz_path_for(npz_index, tic, int(row.sector), int(row.seg_idx)), window, max_absmax
            )
            assert times.shape[0] == int(row.n_win), (
                f"{row.seg_id}: replay produced {times.shape[0]} windows, index says {row.n_win} — alignment broken"
            )
            if i in check_ids:  # exact-float32 re-verification of the row mapping against the memmap
                packed_flux = np.array(dat[int(row.row_start): int(row.row_start) + int(row.n_win)])
                assert np.array_equal(flux.astype(np.float32), packed_flux), (
                    f"{row.seg_id}: replayed flux != packed rows — alignment broken"
                )
                n_flux_checked += 1

            if toi_rows is None:  # transit star with no whitelisted ephemeris -> every window quarantined
                if tic not in seen_unfoldable:
                    seen_unfoldable.add(tic)
                for j in range(int(row.n_win)):
                    records.append({"split": split, "seg_id": row.seg_id, "tic_id": tic, "sector": int(row.sector),
                                    "seg_idx": int(row.seg_idx), "row": int(row.row_start) + j, "win_in_seg": j,
                                    "full_best": "", "part_best": "", "near_best": "", "n_intransit_cad": 0,
                                    "unfoldable": True})
                continue

            dur_imputed = any(not np.isfinite(d) for (_, _, d, _) in toi_rows)
            near_factor = NEAR_FACTOR_IMPUTED if dur_imputed else NEAR_FACTOR
            full_b, part_b, near_b, n_in = annotate_segment(times, toi_rows, median_dur, near_factor)
            for j in range(int(row.n_win)):
                records.append({"split": split, "seg_id": row.seg_id, "tic_id": tic, "sector": int(row.sector),
                                "seg_idx": int(row.seg_idx), "row": int(row.row_start) + j, "win_in_seg": j,
                                "full_best": full_b[j], "part_best": part_b[j], "near_best": near_b[j],
                                "n_intransit_cad": int(n_in[j]), "unfoldable": False})
        n_unfoldable_stars += len(seen_unfoldable)

    df = pd.DataFrame(records)
    for c in ("full_best", "part_best", "near_best"):
        df[c] = df[c].astype("string").fillna("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"wrote {out_path} ({len(df)} window rows)")

    # what-if counters (stored user rule): positive-label size at every knob, visible not silent.
    logger.info("=" * 68)
    logger.info("Window annotation counters (knob = disposition subset x include-partial)")
    logger.info("=" * 68)
    logger.info(f"flux spot-checks passed: {n_flux_checked} segments (exact float32 vs .dat)")
    logger.info(f"annotated windows:       {len(df)}")
    logger.info(f"unfoldable transit stars (no whitelisted eph): {n_unfoldable_stars}")
    has_overlap = (df["full_best"] != "") | (df["part_best"] != "") | (df["near_best"] != "") | df["unfoldable"]
    logger.info(f"clean-negative windows (no overlap, any knob): {int((~has_overlap).sum())}")
    for name, ds, part in (("KP+CP", {"CP", "KP"}, False), ("KP+CP +part", {"CP", "KP"}, True),
                           ("KP+CP+PC", {"CP", "KP", "PC"}, False), ("KP+CP+PC +part", {"CP", "KP", "PC"}, True)):
        pos = df["full_best"].isin(ds) | (part & df["part_best"].isin(ds))
        n_star = df.loc[pos, "tic_id"].nunique()
        logger.info(f"  knob {name:<16} transit=1 windows: {int(pos.sum()):>6}  over {n_star} stars")
    logger.info("Annotation build done.")
    return 0


def masks_for_times(
    times: np.ndarray, rows: list[tuple[float, float, float]], median_dur: float, near_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """(in_transit, near_transit) boolean masks over `times` for a TIC's whitelisted TOIs, OR-combined."""
    t = times.astype(np.float64)
    in_tr = np.zeros(t.shape, dtype=bool)
    near = np.zeros(t.shape, dtype=bool)
    for (P, T0, dur) in rows:
        d = dur if np.isfinite(dur) else median_dur
        phase = np.mod(t - T0 + 0.5 * P, P) - 0.5 * P  # (…,) days from nearest mid-transit
        in_tr |= np.abs(phase) <= 0.5 * d
        near |= np.abs(phase) <= near_factor * d
    return in_tr, near


_NPZ_RE = re.compile(r"^TIC(\d+)_s(\d+)_seg(\d+)_run\d+\.npz$")


def index_npz_by_segment(seq_dir: Path, tics: set[int], logger) -> dict[tuple[int, int, int], list[Path]]:
    """One scandir pass -> {(tic, sector, seg_idx): [npz paths]} for the given TICs (never glob per row)."""
    idx: dict[tuple[int, int, int], list[Path]] = {}
    n = 0
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_RE.match(entry.name)
            if not m:
                continue
            tic = int(m.group(1))
            if tic not in tics:
                continue
            idx.setdefault((tic, int(m.group(2)), int(m.group(3))), []).append(Path(entry.path))
            n += 1
    logger.info(f"indexed {n} npz across {len(idx)} segments for {len(tics)} transit TICs")
    return idx


def npz_path_for(npz_index: dict[tuple[int, int, int], list[Path]], tic_id: int, sector: int, seg_idx: int) -> Path:
    """The unique sequences npz behind one packed index row (asserts exactly one match)."""
    matches = npz_index.get((tic_id, sector, seg_idx), [])
    assert len(matches) == 1, f"expected 1 npz for TIC{tic_id} s{sector} seg{seg_idx}, got {len(matches)}"
    return matches[0]


def replay_segment(npz_path: Path, window: int, max_absmax: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Replay pack.py's load_and_filter on one segment: absmax guard at stored granularity, then
    subdivision to `window`. Returns (times, flux) both shaped (n_win, window), in packed row order.
    """
    with np.load(npz_path) as data:
        windows = data["windows"]  # (N, native, 1) float32
        times = data["times"]      # (N, native) float32 BTJD
    native = windows.shape[1]
    assert native % window == 0, f"window {window} does not divide stored {native}"
    absmax = np.abs(windows).max(axis=(1, 2))  # (N,) guard at STORED granularity, exactly as pack.py
    keep = absmax <= max_absmax
    k = native // window
    flux = windows[keep].reshape(-1, window)   # (M*k, window)
    t = times[keep].reshape(-1, window)        # (M*k, window)
    return t, flux


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1: window-level transit labels on the packed corpus.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N transit TICs (smoke).")
    ap.add_argument("--packed-dir", default=None, help="Default: experiments/exp01_window256_seq16/packed")
    ap.add_argument("--sequences-dir", default=None, help="Default: processed/sequences")
    ap.add_argument("--labels-csv", default=None, help="Default: labels/variability_labels_star.csv (v1)")
    ap.add_argument("--nasa-csv", default=None, help="Default: labels/qc/toi_nasa.csv")
    ap.add_argument("--out", default=None, help="Default depends on --emit (see below)")
    ap.add_argument("--emit", choices=["label", "annot"], default="label",
                    help="'label' = the baked 1/0/-1 parquet (default); 'annot' = rich per-window "
                         "disposition annotation, label derived per knob in the notebook.")
    ap.add_argument("--whitelist", default="CP,KP,PC,APC",
                    help="Comma-separated TFOPWG dispositions folded in --emit annot (default all four; "
                         "the notebook selects subsets, so keep this full unless deliberately narrowing).")
    args = ap.parse_args()

    root = find_project_root()
    packed = Path(args.packed_dir) if args.packed_dir else root / "experiments" / "exp01_window256_seq16" / "packed"
    seq_dir = Path(args.sequences_dir) if args.sequences_dir else root / "processed" / "sequences"
    labels_csv = Path(args.labels_csv) if args.labels_csv else root / "labels" / "variability_labels_star.csv"
    nasa_csv = Path(args.nasa_csv) if args.nasa_csv else root / "labels" / "qc" / "toi_nasa.csv"
    logger = setup_logging(root / "qc_transit_window_labels.log", "transit_window_labels")

    if args.emit == "annot":
        out_path = Path(args.out) if args.out else root / "labels" / "qc" / "transit_window_labels_w256_annot.parquet"
        whitelist = {d.strip().upper() for d in args.whitelist.split(",") if d.strip()}
        assert nasa_csv.exists(), f"missing {nasa_csv}; run src/qc/fetch_toi_enriched.py first"
        return build_annotations(root, packed, seq_dir, labels_csv, nasa_csv, out_path, whitelist, args.limit, logger)

    out_path = Path(args.out) if args.out else root / "labels" / "qc" / "transit_window_labels_w256.parquet"

    manifest = json.loads((packed / "pack_manifest.json").read_text())
    window = int(manifest["window"])
    max_absmax = float(manifest["max_absmax"])
    logger.info(f"packed: {packed} (window={window}, max_absmax={max_absmax})")

    assert nasa_csv.exists(), f"missing {nasa_csv}; run src/qc/fetch_toi_enriched.py first"
    eph = load_ephemerides(nasa_csv, logger)
    median_dur = _median_duration(eph)
    logger.info(f"population median transit duration = {median_dur * 24:.2f} h")

    labels = pd.read_csv(labels_csv)
    labels["transit"] = pd.to_numeric(labels["transit"], errors="coerce").fillna(0).astype(int)
    transit_tics = set(labels.loc[labels["transit"] == 1, "tic_id"].astype(int))
    logger.info(f"v1 transit-positive TICs: {len(transit_tics)}")

    from tqdm.auto import tqdm

    npz_index = index_npz_by_segment(seq_dir, transit_tics, logger)
    rng = np.random.default_rng(0)  # script (not a notebook plot cell): deterministic spot-check sample
    records: list[dict] = []
    star_rows: list[dict] = []
    n_flux_checked = 0
    for split in ["train", "val", "test"]:
        index = pd.read_parquet(packed / f"{split}_index.parquet")
        total = int(index["n_win"].sum())
        dat = np.memmap(packed / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
        seg_rows = index[index["tic_id"].isin(transit_tics)].reset_index(drop=True)
        if args.limit:
            keep_tics = sorted(seg_rows["tic_id"].unique())[: args.limit]
            seg_rows = seg_rows[seg_rows["tic_id"].isin(keep_tics)].reset_index(drop=True)
        check_ids = set(rng.choice(len(seg_rows), size=min(FLUX_CHECK_PER_SPLIT, len(seg_rows)), replace=False).tolist())

        per_star: dict[int, dict] = {}
        for i, row in enumerate(tqdm(seg_rows.itertuples(index=False), desc=f"label[{split}]", total=len(seg_rows))):
            tic = int(row.tic_id)
            toi_rows = eph.get(tic)
            times, flux = replay_segment(
                npz_path_for(npz_index, tic, int(row.sector), int(row.seg_idx)), window, max_absmax
            )
            assert times.shape[0] == int(row.n_win), (
                f"{row.seg_id}: replay produced {times.shape[0]} windows, index says {row.n_win} — alignment broken"
            )
            if i in check_ids:  # exact-float32 re-verification of the row mapping against the memmap
                packed_flux = np.array(dat[int(row.row_start) : int(row.row_start) + int(row.n_win)])
                assert np.array_equal(flux.astype(np.float32), packed_flux), (
                    f"{row.seg_id}: replayed flux != packed rows — alignment broken"
                )
                n_flux_checked += 1

            stat = per_star.setdefault(tic, {"tic_id": tic, "split": split, "foldable": toi_rows is not None,
                                             "dur_imputed": False, "n_win": 0, "n_pos": 0, "n_neg": 0, "n_quar": 0})
            if toi_rows is None:  # KEEP_UNVERIFIABLE: no whitelisted ephemeris -> every window quarantined
                for j in range(int(row.n_win)):
                    records.append({"split": split, "seg_id": row.seg_id, "tic_id": tic, "sector": int(row.sector),
                                    "seg_idx": int(row.seg_idx), "row": int(row.row_start) + j, "win_in_seg": j,
                                    "label": -1, "n_intransit_cad": 0, "reason": "unfoldable"})
                stat["n_win"] += int(row.n_win)
                stat["n_quar"] += int(row.n_win)
                continue

            dur_imputed = any(not np.isfinite(d) for (_, _, d) in toi_rows)
            near_factor = NEAR_FACTOR_IMPUTED if dur_imputed else NEAR_FACTOR
            in_tr, near = masks_for_times(times, toi_rows, median_dur, near_factor)  # (n_win, window) each
            n_in = in_tr.sum(axis=1)   # (n_win,) in-transit cadences per window
            any_near = near.any(axis=1)
            stat["dur_imputed"] = stat["dur_imputed"] or dur_imputed
            for j in range(int(row.n_win)):
                if n_in[j] > 0:
                    label, reason = 1, ""
                elif any_near[j]:
                    label, reason = -1, ("near_transit_imputed_dur" if dur_imputed else "near_transit")
                else:
                    label, reason = 0, ""
                records.append({"split": split, "seg_id": row.seg_id, "tic_id": tic, "sector": int(row.sector),
                                "seg_idx": int(row.seg_idx), "row": int(row.row_start) + j, "win_in_seg": j,
                                "label": label, "n_intransit_cad": int(n_in[j]), "reason": reason})
                stat["n_win"] += 1
                if label == 1:
                    stat["n_pos"] += 1
                elif label == 0:
                    stat["n_neg"] += 1
                else:
                    stat["n_quar"] += 1
        star_rows.extend(per_star.values())

    df = pd.DataFrame(records)
    df["label"] = df["label"].astype(np.int8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"wrote {out_path} ({len(df)} window rows)")

    stars = pd.DataFrame(star_rows)
    stars["quar_share"] = stars["n_quar"] / stars["n_win"]
    summary_path = out_path.with_name("transit_window_labels_summary.csv")
    stars.to_csv(summary_path, index=False)
    logger.info(f"wrote {summary_path} ({len(stars)} stars)")

    # what-if counters (stored user rule): every cut and both buffer widths, visible not silent
    logger.info("=" * 68)
    logger.info("Phase 1 window-label counters")
    logger.info("=" * 68)
    logger.info(f"flux spot-checks passed:  {n_flux_checked} segments (exact float32 vs .dat)")
    logger.info(f"windows labeled:          {len(df)}")
    for label_val, name in ((1, "positive (in-transit)"), (0, "clean negative"), (-1, "quarantine")):
        logger.info(f"  label {label_val:>2} {name}: {int((df['label'] == label_val).sum())}")
    for reason, n in df.loc[df["label"] == -1, "reason"].value_counts().items():
        logger.info(f"    quarantine reason {reason}: {n}")
    logger.info(f"stars: {len(stars)} (foldable {int(stars['foldable'].sum())}, "
                f"dur_imputed {int(stars['dur_imputed'].sum())})")
    logger.info(f"per-star quarantine share: median {stars['quar_share'].median():.4f}, "
                f"q90 {stars['quar_share'].quantile(0.9):.4f}")
    foldable = stars[stars["foldable"]]
    logger.info(f"foldable stars with 0 positive windows: {int((foldable['n_pos'] == 0).sum())} "
                f"(expected ≈ the DROP_NO_TRANSIT bucket members inside the packed subset)")
    logger.info(f"what-if NEAR_FACTOR=1.0 (no buffer): quarantined near_transit windows "
                f"{int((df['reason'] == 'near_transit').sum())} would become clean negatives")
    logger.info("Phase 1 label build done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
