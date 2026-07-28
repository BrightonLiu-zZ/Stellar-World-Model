"""Task 2 (plan 2026-07-22) — join the 4 external new-task catalogs to the npz corpus.

Produces one probe-ready star-level table, `labels/new_task_labels_star.csv`, holding the measured
regression targets and the detection/contrastive class flags for the asteroseismic + rotation-period
probe menu. Pure local joins (all inputs are on disk under labels/external/); no network, so this runs
identically in astro or swm. The v1 flare and rotation-period targets are NOT here: they come from the
canonical `labels/variability_labels_star.csv` at pool/scorecard time.

Shapes emitted (see plan D3):
  detection binary   osc_giant       Hon+2021,   1 where numax_hon present (numax > 10 uHz)
                     solar_like_osc  Hatt+2023,  1 where numax_hatt present (numax > 10 uHz)
  contrastive binary rgb_vs_heb      Sreenivas+2025, State 1 (RGB) -> 1, State 2 (HeB) -> 0, drop 0
  regression         numax_hon, numax_hatt, dnu_hatt, prot_kounkel (median Per per TIC)

A row exists for every corpus TIC covered by at least one of the four catalogs; a task's columns are
NaN where that catalog does not cover the star. Non-covered corpus stars are absent here and become
detection negatives at the join (v1 "safe-with-asterisk" convention).

Run (astro or swm env, from repo root):
    python src/labels/build_new_task_labels.py            # full build
    python src/labels/build_new_task_labels.py --limit 200  # smoke on a catalog head slice
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUMAX_FLOOR = 10.0  # uHz, oscillation period < 28 h fits one 5.7-d segment (plan D5; Hon and Hatt)
PROT_SEGMENT_CAP = 5.7  # d, one packed segment length; longer periods flagged for the extrapolation check

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("build_new_task_labels")

_NPZ_RE = re.compile(r"^TIC(\d+)_s\d+_seg\d+_run\d+\.npz$")


def find_project_root() -> Path:
    """Walk up from CWD until CLAUDE.md is found, marking the repo root."""
    p = Path.cwd()
    for _ in range(10):
        if (p / "CLAUDE.md").exists():
            return p
        p = p.parent
    raise FileNotFoundError("CLAUDE.md not found — cannot determine project root")


def corpus_tics(seq_dir: Path) -> set[int]:
    """Set of TIC IDs with at least one packed-window npz, i.e. the encodable 119k-star corpus."""
    tics: set[int] = set()
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_RE.match(entry.name)
            if m:
                tics.add(int(m.group(1)))
    log.info(f"npz corpus: {len(tics)} unique TICs under {seq_dir}")
    return tics


def load_hon(path: Path, limit: int | None) -> pd.DataFrame:
    """Hon+2021 oscillating-red-giant table --> osc_giant flag + numax_hon regression target (numax > 10)."""
    df = pd.read_csv(path, nrows=limit)
    df["TIC"] = pd.to_numeric(df["TIC"], errors="coerce")
    df["numax"] = pd.to_numeric(df["numax"], errors="coerce")
    n_before = len(df)
    df = df[df["numax"] > NUMAX_FLOOR]
    log.info(f"Hon: {n_before} rows, {len(df)} after numax > {NUMAX_FLOOR} (dropped {n_before - len(df)})")
    out = df[["TIC", "numax"]].rename(columns={"TIC": "tic_id", "numax": "numax_hon"})
    out["osc_giant"] = 1
    return out


def load_hatt(path: Path, limit: int | None) -> pd.DataFrame:
    """Hatt+2023 solar-like-oscillator table --> solar_like_osc flag + numax_hatt/dnu_hatt targets."""
    df = pd.read_csv(path, nrows=limit)
    df["TIC"] = pd.to_numeric(df["TIC"], errors="coerce")
    df["numax"] = pd.to_numeric(df["numax"], errors="coerce")
    df["dnu"] = pd.to_numeric(df["dnu"], errors="coerce")
    n_before = len(df)
    df = df[df["numax"] > NUMAX_FLOOR]
    log.info(f"Hatt: {n_before} rows, {len(df)} after numax > {NUMAX_FLOOR} (dropped {n_before - len(df)}); "
             f"dnu NaN {int(df['dnu'].isna().sum())}")
    out = df[["TIC", "numax", "dnu"]].rename(
        columns={"TIC": "tic_id", "numax": "numax_hatt", "dnu": "dnu_hatt"})
    out["solar_like_osc"] = 1
    return out


def load_sreenivas(path: Path, limit: int | None) -> pd.DataFrame:
    """Sreenivas+2025 evolutionary-state table --> rgb_vs_heb (RGB=1, HeB=0) + gold_sample flag; drop State 0."""
    df = pd.read_csv(path, nrows=limit)
    df["TIC"] = pd.to_numeric(df["TIC"], errors="coerce")
    df["State"] = pd.to_numeric(df["State"], errors="coerce")
    n_before = len(df)
    df = df[df["State"].isin([1, 2])]
    log.info(f"Sreenivas: {n_before} rows, {len(df)} after dropping State=0 (dropped {n_before - len(df)})")
    out = pd.DataFrame({
        "tic_id": df["TIC"],
        "rgb_vs_heb": (df["State"] == 1).astype(int),  # 1 = RGB, 0 = He-burning
        "gold_sample": pd.to_numeric(df["GoldSample"], errors="coerce").fillna(0).astype(int),
    })
    return out


def load_kounkel(path: Path, limit: int | None) -> pd.DataFrame:
    """Kounkel+2022 rotation-period table --> prot_kounkel (median Per per TIC) + gt57 extrapolation flag."""
    df = pd.read_csv(path, nrows=limit)
    df["TIC"] = pd.to_numeric(df["TIC"], errors="coerce")
    df["Per"] = pd.to_numeric(df["Per"], errors="coerce")
    df = df.dropna(subset=["TIC", "Per"])
    prot = df.groupby("TIC")["Per"].median()  # within-TIC period spread is negligible (std ~0.025 d)
    out = pd.DataFrame({"tic_id": prot.index, "prot_kounkel": prot.to_numpy()})
    out["prot_kounkel_gt57"] = (out["prot_kounkel"] > PROT_SEGMENT_CAP).astype(int)
    n_head = int((out["prot_kounkel"] <= PROT_SEGMENT_CAP).sum())
    log.info(f"Kounkel: {df['TIC'].nunique()} unique TICs after median-collapse; "
             f"{n_head} at P <= {PROT_SEGMENT_CAP} d (headline), {int(out['prot_kounkel_gt57'].sum())} longer (flagged)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Join the 4 external new-task catalogs to the npz corpus.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N rows of each catalog (smoke).")
    ap.add_argument("--external-dir", default=None, help="Default: labels/external")
    ap.add_argument("--sequences-dir", default=None, help="Default: processed/sequences")
    ap.add_argument("--out", default=None, help="Default: labels/new_task_labels_star.csv")
    args = ap.parse_args()

    root = find_project_root()
    ext = Path(args.external_dir) if args.external_dir else root / "labels" / "external"
    seq_dir = Path(args.sequences_dir) if args.sequences_dir else root / "processed" / "sequences"
    out_path = Path(args.out) if args.out else root / "labels" / "new_task_labels_star.csv"

    hon = load_hon(ext / "hon2021.csv", args.limit)
    hatt = load_hatt(ext / "hatt2023.csv", args.limit)
    sreenivas = load_sreenivas(ext / "sreenivas2025.csv", args.limit)
    kounkel = load_kounkel(ext / "kounkel2022.csv", args.limit)

    merged = hon
    for other in [hatt, sreenivas, kounkel]:
        merged = merged.merge(other, on="tic_id", how="outer")
    merged["tic_id"] = merged["tic_id"].astype(int)

    corpus = corpus_tics(seq_dir)
    n_all = len(merged)
    merged = merged[merged["tic_id"].isin(corpus)].reset_index(drop=True)
    log.info(f"catalog union {n_all} TICs, {len(merged)} in npz corpus (dropped {n_all - len(merged)})")

    for flag in ["osc_giant", "solar_like_osc"]:
        merged[flag] = pd.to_numeric(merged[flag], errors="coerce").fillna(0).astype(int)
    cols = ["tic_id", "osc_giant", "numax_hon", "solar_like_osc", "numax_hatt", "dnu_hatt",
            "rgb_vs_heb", "gold_sample", "prot_kounkel", "prot_kounkel_gt57"]
    merged = merged[cols].sort_values("tic_id").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(merged)} rows)")

    log.info("=" * 68)
    log.info("New-task corpus join counts (measured, in npz corpus)")
    log.info("=" * 68)
    log.info(f"osc_giant (Hon):        {int(merged['osc_giant'].sum())}")
    log.info(f"solar_like_osc (Hatt):  {int(merged['solar_like_osc'].sum())}")
    log.info(f"rgb_vs_heb present:     {int(merged['rgb_vs_heb'].notna().sum())} "
             f"(RGB {int((merged['rgb_vs_heb'] == 1).sum())}, HeB {int((merged['rgb_vs_heb'] == 0).sum())}, "
             f"gold {int(merged['gold_sample'].sum())})")
    log.info(f"prot_kounkel present:   {int(merged['prot_kounkel'].notna().sum())} "
             f"(headline P<=5.7d {int((merged['prot_kounkel_gt57'] == 0).sum())})")
    log.info(f"dnu_hatt present:       {int(merged['dnu_hatt'].notna().sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
