"""exp05 primary-criterion aggregation: does weighting dynamics up (lambda>0) beat the lambda=0 'off'
cell? For each recipe (comb/lpsd), task, and treatment cell, computes the paired-by-seed probe delta
  Delta = pr_auc(cell, seed) - pr_auc(off, seed)   (logistic x mean, best_recon_aux, v1 labels)
across the 4 seeds, and flags |mean| > 2*SE (SE = sd/sqrt(n_seeds)). Also reports each cell's gap vs the
shared untrained reference for context. This is success criterion 1 of the exp05 plan.

Reads each cell's experiments/exp05_*/results/readout_sweep.csv (append-only; keeps the latest run per
seed x cell x task). Writes experiments/exp05_gap.csv (+ prints a table). Run: python experiments/analyze_exp05_gap.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["pulsating", "eb", "rotation", "transit"]
POOLING, READOUT, CKPT = "mean", "logistic", "best_recon_aux"
OFF = {"comb": "exp05_comb_off", "lpsd": "exp05_lpsd_off"}


def load_all() -> pd.DataFrame:
    frames = []
    for csv in ROOT.glob("experiments/exp05_*/results/readout_sweep.csv"):
        df = pd.read_csv(csv)
        frames.append(df)
    if not frames:
        raise SystemExit("no exp05 readout_sweep.csv found - run the eval scan first")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.pooling == POOLING) & (df.readout == READOUT) & (df.ckpt == CKPT)]
    df = df[df.get("labels_version", "v1").fillna("v1") == "v1"]
    # append-only -> keep the latest row per (exp_name, seed, task)
    sort_col = "run_id" if "run_id" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col)
    df = df.drop_duplicates(subset=["exp_name", "seed", "task"], keep="last")
    return df


def main():
    df = load_all()
    rows = []
    for recipe, off_name in OFF.items():
        cells = sorted(df.loc[df.exp_name.str.startswith(f"exp05_{recipe}_") & (df.exp_name != off_name), "exp_name"].unique())
        off = df[df.exp_name == off_name]
        for cell in cells:
            cur = df[df.exp_name == cell]
            for task in TASKS:
                o = off[off.task == task][["seed", "pr_auc"]].rename(columns={"pr_auc": "off"})
                c = cur[cur.task == task][["seed", "pr_auc", "gap"]].rename(columns={"pr_auc": "cell"})
                m = c.merge(o, on="seed", how="inner")
                if len(m) == 0:
                    continue
                delta = (m["cell"] - m["off"]).to_numpy()
                n = len(delta)
                dmean, dsd = float(delta.mean()), float(delta.std(ddof=1)) if n > 1 else float("nan")
                se = dsd / np.sqrt(n) if n > 1 else float("nan")
                rows.append({
                    "recipe": recipe, "cell": cell.replace(f"exp05_{recipe}_", ""), "task": task, "n_seeds": n,
                    "delta_vs_off_mean": dmean, "delta_sd": dsd, "delta_2se": 2 * se if n > 1 else float("nan"),
                    "confirm": (n > 1 and abs(dmean) > 2 * se),
                    "cell_prauc_mean": float(m["cell"].mean()), "off_prauc_mean": float(m["off"].mean()),
                    "gap_vs_untrained_mean": float(m["gap"].mean()),
                })
    out = pd.DataFrame(rows)
    out_path = ROOT / "experiments" / "exp05_gap.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n=== exp05 primary criterion: Delta = pr_auc(cell) - pr_auc(off), paired by seed (logistic x mean) ===")
    for recipe in OFF:
        sub = out[out.recipe == recipe]
        if sub.empty:
            continue
        print(f"\n## {recipe}  (off = {OFF[recipe]})")
        for task in TASKS:
            t = sub[sub.task == task].sort_values("delta_vs_off_mean", ascending=False)
            if t.empty:
                continue
            print(f"  [{task}]")
            for r in t.itertuples(index=False):
                flag = "CONFIRM" if r.confirm else ("     " if r.n_seeds < 2 else "  ns ")
                sign = "+" if r.delta_vs_off_mean >= 0 else ""
                print(f"    {r.cell:<14} d={sign}{r.delta_vs_off_mean:.4f} +/- {r.delta_sd:.4f} (2SE {r.delta_2se:.4f}) "
                      f"{flag}  | cell {r.cell_prauc_mean:.3f} vs off {r.off_prauc_mean:.3f} | gap-vs-untr {r.gap_vs_untrained_mean:+.3f}")
    n_conf = int(out.confirm.sum())
    print(f"\n{n_conf} of {len(out)} (cell x task) cells beat/hurt off by >2*SE. Wrote {out_path}")


if __name__ == "__main__":
    main()
