"""Appendix B figure for the ML4PS paper: validation reconstruction versus downstream probe score.

One point per pre-training recipe (the ten exp05 comb cells: same data, same loss functions, the
dynamics weighting lambda the only axis), one panel per v1 task. x = best post-warmup validation
reconstruction (mean over the cell's four seeds), y = the frozen linear probe's PR-AUC on mu at the
`mean` pooling (mean over the same seeds). This is finding F11 (`experiments/cross_experiment_findings.md`)
re-drawn at paper width; the numbers it prints are the ones quoted in Section 4 and Appendix B.

Inputs (all already on disk, nothing is re-run):
    experiments/exp05_forensics/curves_exp05/<cell>_B_seed<k>.csv    W&B histories (dump_wandb_history)
    experiments/<cell>/results/readout_sweep.csv                       per-seed probe scores

Outputs:
    paper/figures/figB_valloss.pdf     what LaTeX embeds
    paper/build/figB_valloss.png       screen copy
    paper/build/figB_data.csv          every plotted number, with provenance columns

Run (swm env, repo root):
    python experiments/plot_ml4ps_appendix_b.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

repo_root = Path(__file__).resolve().parents[1]
CURVES = repo_root / "experiments" / "exp05_forensics" / "curves_exp05"
WARMUP = 10  # beta warm-up epochs; selection is restricted to epoch >= WARMUP, as in the notebook
SEEDS = [0, 1, 2, 3]
CELLS = ["exp05_comb_off",
         "exp05_comb_fwd_c0p1", "exp05_comb_fwd_c0p3", "exp05_comb_fwd_c1p0",
         "exp05_comb_fbwd_c0p1", "exp05_comb_fbwd_c0p3", "exp05_comb_fbwd_c1p0",
         "exp05_comb_multi_c0p1", "exp05_comb_multi_c0p3", "exp05_comb_multi_c1p0"]
TASKS = ["pulsating", "eb", "rotation", "transit"]
TASK_LABEL = {"pulsating": "pulsating", "eb": "eclipsing binary", "rotation": "rotation", "transit": "transit"}
MODE_LABEL = {"off": "dynamics off", "fwd": "forward", "fbwd": "forward+backward", "multi": "multi-step"}
MARKER = {"off": "X", "fwd": "o", "fbwd": "s", "multi": "^"}


def load_curve(cell: str, seed: int) -> pd.DataFrame:
    """One run's W&B history, with a killed-and-resumed prefix stitched in front when one exists."""
    stem = f"{cell}_B_seed{seed}"
    main = pd.read_csv(CURVES / f"{stem}.csv")
    prefix_path = CURVES / f"{stem}.killedprefix.csv"
    if prefix_path.exists():
        prefix = pd.read_csv(prefix_path)
        prefix = prefix[prefix["epoch"] < main["epoch"].min()]
        main = pd.concat([prefix, main], ignore_index=True).sort_values("epoch").reset_index(drop=True)
    return main


def cell_table() -> pd.DataFrame:
    """One row per (cell, task): seed-mean best val/recon and seed-mean probe PR-AUC, plus the seeds' spread."""
    recon = {}
    for cell in CELLS:
        per_seed = []
        for seed in SEEDS:
            post = load_curve(cell, seed)
            post = post[post["epoch"] >= WARMUP]
            per_seed.append(float(post["val/recon"].min()))
        recon[cell] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)))
    rows = []
    for cell in CELLS:
        sweep = pd.read_csv(repo_root / "experiments" / cell / "results" / "readout_sweep.csv")
        sweep = sweep[(sweep["readout"] == "logistic") & (sweep["pooling"] == "mean")]
        for task in TASKS:
            scores = sweep[sweep["task"] == task]["pr_auc"].to_numpy(dtype=float)
            assert len(scores) == len(SEEDS), f"{cell}/{task}: {len(scores)} seeds in readout_sweep.csv"
            rows.append({"cell": cell, "mode": cell.split("_")[2], "task": task,
                         "val_recon_min": recon[cell][0], "val_recon_min_sd": recon[cell][1],
                         "pr_auc": float(scores.mean()), "pr_auc_sd": float(scores.std(ddof=1)),
                         "n_seeds": len(SEEDS)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Appendix B: val reconstruction vs probe score, per recipe.")
    ap.add_argument("--out-dir", default="paper")
    args = ap.parse_args()
    out_dir = repo_root / args.out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "build").mkdir(parents=True, exist_ok=True)

    table = cell_table()
    stats = []
    for task in TASKS:
        t = table[table["task"] == task]
        rho_all = spearmanr(-t["val_recon_min"], t["pr_auc"]).statistic
        without_off = t[t["mode"] != "off"]
        rho_no_off = spearmanr(-without_off["val_recon_min"], without_off["pr_auc"]).statistic
        loss_pick = t.loc[t["val_recon_min"].idxmin()]
        probe_pick = t.loc[t["pr_auc"].idxmax()]
        stats.append({"task": task, "rho_all_ten": float(rho_all), "rho_without_off": float(rho_no_off),
                      "loss_optimal_cell": loss_pick["cell"], "probe_optimal_cell": probe_pick["cell"],
                      "cost_of_trusting_loss": float(probe_pick["pr_auc"] - loss_pick["pr_auc"])})
    stats = pd.DataFrame(stats)
    table.merge(stats, on="task").to_csv(out_dir / "build" / "figB_data.csv", index=False)

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.35))
    for ax, task in zip(axes, TASKS):
        t = table[table["task"] == task]
        for _, r in t.iterrows():
            ax.scatter(r["val_recon_min"], r["pr_auc"], marker=MARKER[r["mode"]], s=16,
                       color="tab:red" if r["mode"] == "off" else "tab:blue", linewidths=0.5, zorder=3)
        s = stats[stats["task"] == task].iloc[0]
        ax.set_title(f"{TASK_LABEL[task]}: $\\rho={s['rho_all_ten']:.2f}$", pad=3)
        ax.tick_params(length=2, pad=1.5)
        ax.locator_params(axis="both", nbins=4)
    axes[0].set_ylabel("probe PR-AUC on $\\mu$")
    fig.supxlabel("best validation reconstruction loss (mean over four seeds)", y=-0.01)
    handles = [plt.Line2D([], [], marker=MARKER[m], color="tab:red" if m == "off" else "tab:blue",
                          linestyle="", markersize=4, label=MODE_LABEL[m]) for m in MARKER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.20),
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out_dir / "figures" / "figB_valloss.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "build" / "figB_valloss.png", dpi=200, bbox_inches="tight")

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(stats.to_string(index=False))
    print(f"wrote {out_dir / 'figures' / 'figB_valloss.pdf'} and build/figB_{{valloss.png,data.csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
