"""exp05 criterion 2 (learned physics): aggregate rollout-vs-persistence per class across seeds.
gain_ratio = persistence_mse / rollout_mse (>1 => rollout beats copy-last). The pre-registered
prediction was periodic > quiet (physics helps periodic). Result (2026-07-23): rollout beats persistence
everywhere (all ratios >1) but MORE on quiet than periodic (periodic-quiet gap NEGATIVE) => PARTIAL, not a
clean fail; the rollout learns 'smooth is predictable', not periodic physics -- exp06 target (see decoded
example figs). NOTE: only multistep
cells were TRAINED for free-running rollout - fwd/fwd_bwd rollout numbers are off-distribution.
Run: python experiments/analyze_exp05_rollout.py
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
frames = [pd.read_csv(c) for c in ROOT.glob("experiments/exp05_*/results/rollout_vs_persistence.csv")]
df = pd.concat(frames, ignore_index=True)
if "run_id" not in df.columns:
    df["run_id"] = 0
df = df.drop_duplicates(subset=["exp", "seed", "class"], keep="last")

rows = []
for exp in sorted(df.exp.unique()):
    g = df[df.exp == exp]
    rec = {"cell": exp.replace("exp05_", ""), "mode": g.dyn_mode.iloc[0], "lambda_dyn": g.lambda_dyn.iloc[0]}
    for cls in ("periodic", "quiet"):
        h = g[g["class"] == cls]
        rec[f"{cls}_ratio"] = float(h.gain_ratio.mean()) if len(h) else float("nan")
        rec[f"{cls}_roll_mse"] = float(h.rollout_mse.mean()) if len(h) else float("nan")
        rec[f"{cls}_pers_mse"] = float(h.persistence_mse.mean()) if len(h) else float("nan")
    rec["periodic_minus_quiet_ratio"] = rec["periodic_ratio"] - rec["quiet_ratio"]
    rows.append(rec)
out = pd.DataFrame(rows).sort_values("cell")
out.to_csv(ROOT / "experiments" / "exp05_rollout_summary.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print("\nall ratios >1 but periodic_minus_quiet_ratio < 0 for all trained cells => criterion 2 PARTIAL (exp06 target).")
print("Judge rollout quality on MULTISTEP cells only (trained for free-running rollout).")
