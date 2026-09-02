"""exp10 forensics -- the footing gates, run before any new number in this directory is read.

Three checks, all mandatory, and all of them comparisons against artifacts that were published before
this session existed. The point is not that the derived-cache path is plausible; it is that a derived
cache which quietly holds the wrong columns produces a table that looks exactly like a finding.

    gate 1  the identity (passthrough) derived cache must reproduce F1's published numbers. The named
            target is `eb` @ `mean`, hann0p3_fbwd, 6 seeds = 0.7710 (tol 5e-4), and since the whole
            probe table is available the check is widened to every row the two artifacts share.
    gate 2  `features_only` under GBM must reproduce `c3_feature_controls` EXACTLY (0.0). Same scorers,
            same seeds; anything else means one of the two scripts has drifted.
    gate 3  the resolved cache paths are recorded in every output CSV, so a mislabelled table stays
            detectable after the fact. Asserted here rather than assumed.

Run (repo root, swm env, PYTHONPATH=src; seconds):
    python experiments/exp10_footing_checks.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_footing")

metric_of = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
join_keys = ["block", "task", "arm", "seed", "readout", "readout_family", "arm_set"]


def headline(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the wide probe table to one score column, so two artifacts can be differenced row-wise."""
    out = frame.copy()
    values = []
    for _, row in out.iterrows():
        values.append(float(row[metric_of[row["shape"]]]))
    out["score"] = values
    return out


def compare(new: pd.DataFrame, ref: pd.DataFrame, label: str, tol: float) -> dict:
    """One gate: inner-join two probe tables on their identifying keys and report the worst disagreement."""
    merged = headline(new).merge(headline(ref), on=join_keys, suffixes=("_new", "_ref"))
    assert len(merged) > 0, f"{label}: the two artifacts share no rows, so nothing was checked"
    merged["abs_diff"] = (merged["score_new"] - merged["score_ref"]).abs()
    worst = merged.sort_values("abs_diff", ascending=False).head(3)
    log.info(f"{label}: {len(merged)} shared rows, max |diff| {merged['abs_diff'].max():.3e}\n"
             + worst[join_keys + ["score_new", "score_ref", "abs_diff"]].round(6).to_string(index=False))
    passed = bool(merged["abs_diff"].max() <= tol)
    return {"gate": label, "n_rows": len(merged), "max_abs_diff": float(merged["abs_diff"].max()),
            "tol": tol, "passed": passed}


def main() -> int:
    ap = argparse.ArgumentParser(description="exp10 forensics footing gates.")
    ap.add_argument("--footing-dir", default="experiments/exp10_forensics/footing")
    args = ap.parse_args()

    footing_dir = repo_root / args.footing_dir
    probe = pd.read_csv(footing_dir / "f1_probe.csv")
    rows = []

    eb = probe[(probe["family"] == "hann0p3_fbwd") & (probe["task"] == "eb")
               & (probe["readout"] == "mean") & (probe["readout_family"] == "linear")
               & (probe["arm_set"] == "mu")]
    assert len(eb) == 6, f"expected 6 seeds for the eb repro target, found {len(eb)}"
    rows.append({"gate": "gate1_eb_mu_mean_6seed", "n_rows": 6,
                 "max_abs_diff": float(abs(eb["pr_auc"].mean() - 0.7710)), "tol": 5e-4,
                 "passed": bool(abs(eb["pr_auc"].mean() - 0.7710) < 5e-4)})
    log.info(f"gate 1 named target: eb @mean mu, 6 seeds = {eb['pr_auc'].mean():.4f} vs published 0.7710")

    linear_ref = pd.read_csv(repo_root / "experiments" / "f1_fusion_scorecard" / "f1_probe.csv")
    rows.append(compare(probe[probe["readout_family"] == "linear"],
                        linear_ref[linear_ref["readout"] == "mean"], "gate1_identity_vs_f1_linear", 5e-4))

    nonlinear_ref = pd.read_csv(repo_root / "experiments" / "f1_nonlinear_control" / "f1_probe.csv")
    rows.append(compare(probe[probe["readout_family"] == "gbm"], nonlinear_ref,
                        "gate1_identity_vs_c3b_gbm", 5e-4))

    c3 = pd.read_csv(repo_root / "experiments" / "c3_feature_controls" / "c3_probe.csv")
    c3 = headline(c3[c3["family"] == "gbm"])
    ours = headline(probe[(probe["readout_family"] == "gbm") & (probe["arm_set"] == "features_only")])
    merged = ours.merge(c3, on=["block", "task", "seed"], suffixes=("_new", "_ref"))
    assert len(merged) > 0, "gate 2: no shared features_only rows with C3"
    merged["abs_diff"] = (merged["score_new"] - merged["score_ref"]).abs()
    log.info(f"gate 2: {len(merged)} shared features_only GBM rows, "
             f"max |diff| {merged['abs_diff'].max():.3e}")
    rows.append({"gate": "gate2_features_only_gbm_vs_c3", "n_rows": len(merged),
                 "max_abs_diff": float(merged["abs_diff"].max()), "tol": 0.0,
                 "passed": bool(merged["abs_diff"].max() == 0.0)})

    paths = probe[["pool_cache", "subset_cache"]].drop_duplicates()
    assert len(paths) == 1 and paths["pool_cache"].iloc[0].endswith("mu_cache"), \
        "gate 3: the probe table does not record exactly one resolved cache pair"
    rows.append({"gate": "gate3_cache_paths_recorded", "n_rows": len(probe),
                 "max_abs_diff": np.nan, "tol": np.nan, "passed": True})

    result = pd.DataFrame(rows)
    result.to_csv(footing_dir / "footing_gates.csv", index=False)
    print("\nexp10 forensics footing gates:")
    print(result.to_string(index=False))
    assert result["passed"].all(), "a footing gate FAILED; no forensic number may be read"
    log.info(f"all gates passed; wrote {footing_dir / 'footing_gates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
