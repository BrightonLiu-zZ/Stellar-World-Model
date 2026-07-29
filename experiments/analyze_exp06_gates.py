"""exp06 verdicts: apply the pre-registered gates from the manifest to exp06_geometry_gap.csv.

Gates (manifest LOCKED DESIGN, grill 2026-07-27) - all on the dyn-on arm, amplitude-residualized
mean-pooled mu (pooling == mean_resid), 6 paired seeds, one-sided > 2*SE:

  H1-a   Delta_g(eb) - Delta_g(pulsating) > 0        for g in {512, 1024} vs 256
  H1-b   Delta_g(rotation) - Delta_g(pulsating) > 0
  ctrl   Delta_g(pulsating) itself ~ 0 (must NOT improve materially - coverage already 93%)
  C1-rep fwd_bwd - off > 2*SE per task at every geometry (paired by seed)
  H2     lag-1 mu-trajectory ACF (periodic, trained) per geometry; rollout eval re-opens only >= 0.3

Delta_g(task) = pr_auc(geometry g) - pr_auc(256x16), computed per seed then paired. The dyn-off arm is
reported as a same-direction robustness check, not gated. w2048 (dyn-off only) extends the coverage
curve for eb and is not gated.

Run after analyze_exp06_geometry_gap.py:
    python experiments/analyze_exp06_gates.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("pulsating", "eb", "rotation", "transit")
GEOMS = (512, 1024)


def paired_stats(deltas: np.ndarray) -> tuple[float, float, bool]:
    """Mean, SE over seeds, and the one-sided > 2*SE gate."""
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    return mean, se, bool(mean > 2 * se)


def pivot(gap: pd.DataFrame, arm_cells: dict[int, str], pooling: str, task: str) -> pd.DataFrame:
    """Seed-by-window PR-AUC matrix for one (cell-per-window mapping, pooling, task)."""
    rows = gap[(gap.pooling == pooling) & (gap.task == task) & (gap.seed >= 0)
               & (gap.cell.isin(arm_cells.values()))]
    return rows.pivot(index="seed", columns="window", values="pr_auc")


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 pre-registered gate evaluation")
    ap.add_argument("--gap-csv", default="experiments/exp06_geometry_gap.csv")
    ap.add_argument("--acf-csv", default="experiments/exp06_geometry_acf.csv")
    ap.add_argument("--out", default="experiments/exp06_gates.csv")
    args = ap.parse_args()

    gap = pd.read_csv(ROOT / args.gap_csv)
    acf = pd.read_csv(ROOT / args.acf_csv)
    fbwd = {256: "exp06_w256_fbwd", 512: "exp06_w512_fbwd", 1024: "exp06_w1024_fbwd"}
    off = {256: "exp06_w256_off", 512: "exp06_w512_off", 1024: "exp06_w1024_off", 2048: "exp06_w2048_off"}
    out_rows = []

    print("=" * 100)
    print("H1 differentials - GATE arm: dyn-on (fwd_bwd), pooling mean_resid   [robustness: dyn-off]")
    print("=" * 100)
    for arm_name, cells in [("dyn-on", fbwd), ("dyn-off", {k: v for k, v in off.items() if k != 2048})]:
        gated = arm_name == "dyn-on"
        for pooling in ["mean_resid", "mean"]:
            mats = {t: pivot(gap, cells, pooling, t) for t in TASKS}
            for g in GEOMS:
                d = {t: (mats[t][g] - mats[t][256]).to_numpy() for t in TASKS}
                for gate_name, diff in [("H1-a eb-puls", d["eb"] - d["pulsating"]),
                                        ("H1-b rot-puls", d["rotation"] - d["pulsating"])]:
                    mean, se, passed = paired_stats(diff)
                    flag = "PASS" if passed else "fail"
                    tag = " <== GATE" if (gated and pooling == "mean_resid") else ""
                    print(f"  [{arm_name:7s}|{pooling:10s}] {gate_name:14s} @{g:4d}: {mean:+.4f} +/- {se:.4f}  {flag}{tag}")
                    out_rows.append({"gate": gate_name, "arm": arm_name, "pooling": pooling, "geometry": g,
                                     "mean": mean, "se": se, "pass": passed, "is_gate": gated and pooling == "mean_resid"})
                mc, sec, _ = paired_stats(d["pulsating"])
                print(f"  [{arm_name:7s}|{pooling:10s}] ctrl puls dlt  @{g:4d}: {mc:+.4f} +/- {sec:.4f}  "
                      f"({'ok (no material gain)' if mc < 2 * sec else 'CONTROL MOVED - inspect'})")
                out_rows.append({"gate": "ctrl pulsating", "arm": arm_name, "pooling": pooling,
                                 "geometry": g, "mean": mc, "se": sec, "pass": mc < 2 * sec,
                                 "is_gate": False})

    print("=" * 100)
    print("C1 replication - paired fwd_bwd - off per geometry (pooling mean, exp05 protocol; and mean_resid)")
    print("=" * 100)
    for pooling in ["mean", "mean_resid"]:
        for g in (256, 512, 1024):
            for task in TASKS:
                on = pivot(gap, fbwd, pooling, task)[g].to_numpy()
                base = pivot(gap, off, pooling, task)[g].to_numpy()
                mean, se, passed = paired_stats(on - base)
                print(f"  [{pooling:10s}] C1 {task:9s} @{g:4d}: {mean:+.4f} +/- {se:.4f}  {'PASS' if passed else 'fail'}")
                out_rows.append({"gate": f"C1 {task}", "arm": "fbwd-off", "pooling": pooling,
                                 "geometry": g, "mean": mean, "se": se, "pass": passed, "is_gate": pooling == "mean"})

    print("=" * 100)
    print("Coverage curve (dyn-off cells + w2048) - eb and controls by window, mean over seeds")
    print("=" * 100)
    for pooling in ["mean_resid", "mean", "mean_std", "kmatch4"]:
        for task in TASKS:
            sub = gap[(gap.pooling == pooling) & (gap.task == task) & (gap.seed >= 0)
                      & (gap.cell.isin(off.values()))]
            if sub.empty:
                continue
            by_w = sub.groupby("window").pr_auc.agg(["mean", "sem"])
            vals = "  ".join(f"{w}:{m:.3f}+/-{s:.3f}" for w, (m, s) in by_w.iterrows())
            print(f"  [{pooling:10s}] {task:9s}  {vals}")

    print("=" * 100)
    print("Trained - untrained gap per geometry (capacity check; pooling mean_resid, dyn-on, seed mean)")
    print("=" * 100)
    unt = gap[(gap.arm == "untrained")]
    for task in TASKS:
        parts = []
        for g, cell in fbwd.items():
            tr = gap[(gap.cell == cell) & (gap.pooling == "mean_resid") & (gap.task == task)].pr_auc.mean()
            un = unt[(unt.window == g) & (unt.pooling == "mean_resid") & (unt.task == task)].pr_auc.mean()
            parts.append(f"{g}:{tr - un:+.3f}")
        print(f"  gap {task:9s}  " + "  ".join(parts))

    print("=" * 100)
    print("H2 pre-gate - lag-1 mu-ACF, periodic stars, trained arms (threshold 0.3 to re-open rollout eval)")
    print("=" * 100)
    a1 = acf[(acf.lag == 1) & (acf.cls == "periodic") & (acf.seed >= 0)]
    for (cell, w), grp in a1.groupby(["cell", "window"]):
        verdict = "RE-OPEN rollout eval" if grp.acf.mean() >= 0.3 else "stay closed"
        print(f"  {cell:18s} w{w:4d}: lag-1 ACF {grp.acf.mean():+.4f} (n_stars~{int(grp.n_stars.mean())})  -> {verdict}")
        out_rows.append({"gate": "H2 acf", "arm": "trained", "pooling": "-", "geometry": w,
                         "mean": float(grp.acf.mean()), "se": float(grp.acf.sem()),
                         "pass": bool(grp.acf.mean() >= 0.3), "is_gate": False, "cell": cell})

    pd.DataFrame(out_rows).to_csv(ROOT / args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
