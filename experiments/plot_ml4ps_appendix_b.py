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


CURVES09 = repo_root / "experiments" / "exp09_forensics" / "curves_exp09"
# Cells whose row the body's VOID rule drops: one seed collapsed, and no seed is ever dropped.
VOID09 = {"exp09_dpss_impulse_w0p0125", "exp09_dpss_impulse_w0p02"}


def load_curve(cell: str, seed: int, curves: Path = CURVES) -> pd.DataFrame:
    """One run's W&B history, with a killed-and-resumed prefix stitched in front when one exists."""
    stem = f"{cell}_B_seed{seed}"
    main = pd.read_csv(curves / f"{stem}.csv")
    prefix_path = curves / f"{stem}.killedprefix.csv"
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


def exp09_table() -> pd.DataFrame:
    """The same estimator on the shipped family: every cell built on hann0p3 (the exp07 incumbent plus
    every exp09 knob), from the exp09 probe summaries (pooling=mean, linear readout) and curves."""
    files = sorted((repo_root / "experiments").glob("exp09_diag_*_probe_summary.csv"))
    sc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sc = sc[(sc["pooling"] == "mean") & (sc["cell"] != "untrained_w256")]
    # a cell scored in two summary files agrees to <= 0.008 PR-AUC per seed; average the duplicates
    sc = sc.groupby(["cell", "seed", "task"], as_index=False)["pr_auc"].mean()
    rows = []
    for cell, g in sc.groupby("cell"):
        seeds = sorted(int(s) for s in g["seed"].unique())
        per_seed = []
        for seed in seeds:
            post = load_curve(cell, seed, CURVES09)
            post = post[post["epoch"] >= WARMUP]
            per_seed.append(float(post["val/recon"].min()))
        for task in TASKS:
            scores = g[g["task"] == task].set_index("seed").loc[seeds, "pr_auc"].to_numpy(dtype=float)
            assert len(scores) == len(seeds), f"{cell}/{task}: {len(scores)} probe seeds vs {len(seeds)} curves"
            rows.append({"cell": cell, "void": cell in VOID09, "task": task,
                         "val_recon_min": float(np.mean(per_seed)), "val_recon_min_sd": float(np.std(per_seed, ddof=1)),
                         "pr_auc": float(scores.mean()), "pr_auc_sd": float(scores.std(ddof=1)),
                         "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


def exp09_stats(table: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for task in TASKS:
        t = table[table["task"] == task]
        kept = t[~t["void"]]
        loss_pick = kept.loc[kept["val_recon_min"].idxmin()]
        probe_pick = kept.loc[kept["pr_auc"].idxmax()]
        stats.append({"task": task, "n_cells_all": len(t), "n_cells_kept": len(kept),
                      "rho_all": float(spearmanr(-t["val_recon_min"], t["pr_auc"]).statistic),
                      "rho_kept": float(spearmanr(-kept["val_recon_min"], kept["pr_auc"]).statistic),
                      "p_kept": float(spearmanr(-kept["val_recon_min"], kept["pr_auc"]).pvalue),
                      "loss_optimal_cell": loss_pick["cell"], "probe_optimal_cell": probe_pick["cell"],
                      "cost_of_trusting_loss": float(probe_pick["pr_auc"] - loss_pick["pr_auc"])})
    return pd.DataFrame(stats)


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
    table09 = exp09_table()
    stats09 = exp09_stats(table09)
    table09.merge(stats09, on="task").to_csv(out_dir / "build" / "figB_exp09_data.csv", index=False)
    kept09 = table09[~table09["void"]]

    # Upper row: the exp05 comb sweep (dynamics-term axis). Lower row: the shipped hann0p3 family
    # (auxiliary-term axis). Same x and y definitions; each row keeps its own axis ranges.
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.75))
    for ax, task in zip(axes[0], TASKS):
        t = table[table["task"] == task]
        for _, r in t.iterrows():
            ax.scatter(r["val_recon_min"], r["pr_auc"], marker=MARKER[r["mode"]], s=16,
                       color="tab:red" if r["mode"] == "off" else "tab:blue", linewidths=0.5, zorder=3)
        s = stats[stats["task"] == task].iloc[0]
        ax.set_title(f"{TASK_LABEL[task]}: $\\rho={s['rho_all_ten']:.2f}$", pad=3)
    for ax, task in zip(axes[1], TASKS):
        t = kept09[kept09["task"] == task]
        for _, r in t.iterrows():
            if r["cell"] == "exp09_aux_none":
                style = dict(marker="D", color="tab:red", s=12)
            elif r["cell"].endswith("_off"):
                style = dict(marker="X", color="tab:red")
            elif r["cell"] == "exp07_hann0p3_fbwd":
                style = dict(marker="*", color="tab:blue", s=30)
            else:
                style = dict(marker="o", color="tab:blue")
            ax.scatter(r["val_recon_min"], r["pr_auc"], s=style.pop("s", 16), linewidths=0.5, zorder=3, **style)
        s = stats09[stats09["task"] == task].iloc[0]
        ax.set_title(f"{TASK_LABEL[task]}: $\\rho={s['rho_kept']:+.2f}$", pad=3)
    for ax in axes.ravel():
        ax.tick_params(length=2, pad=1.5)
        ax.locator_params(axis="both", nbins=4)
    axes[0][0].set_ylabel("PR-AUC on $\\mu$\n(a) exp05 sweep")
    axes[1][0].set_ylabel("PR-AUC on $\\mu$\n(b) shipped family")
    fig.supxlabel("best validation reconstruction loss (mean over seeds)", y=0.01)
    handles = [plt.Line2D([], [], marker=MARKER[m], color="tab:red" if m == "off" else "tab:blue",
                          linestyle="", markersize=4, label=MODE_LABEL[m] if m == "off" else f"(a) {MODE_LABEL[m]}")
               for m in MARKER]
    handles += [plt.Line2D([], [], marker="D", color="tab:red", linestyle="", markersize=3.5, label="(b) no auxiliary term"),
                plt.Line2D([], [], marker="*", color="tab:blue", linestyle="", markersize=5, label="(b) shipped recipe"),
                plt.Line2D([], [], marker="o", color="tab:blue", linestyle="", markersize=4, label="(b) other recipes")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    fig.savefig(out_dir / "figures" / "figB_valloss.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "build" / "figB_valloss.png", dpi=200, bbox_inches="tight")

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("exp05 comb family (the figure):")
        print(stats.to_string(index=False))
        print("\nshipped hann0p3 family (exp07 incumbent + exp09 cells), same estimator:")
        print(stats09.to_string(index=False))
        print(table09.pivot(index="cell", columns="task", values="pr_auc")
              .join(table09.groupby("cell")[["val_recon_min", "n_seeds", "void"]].first()).sort_values("val_recon_min")
              .to_string())
    print(f"wrote {out_dir / 'figures' / 'figB_valloss.pdf'} and build/figB_{{valloss.png,data.csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
