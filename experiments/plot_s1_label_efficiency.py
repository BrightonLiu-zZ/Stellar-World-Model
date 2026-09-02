"""S1 figures: does the fusion advantage grow as the label budget shrinks?

Two figures, because the question and the sanity check want different axes.

  s1_delta_curves.png   ONE panel, the paper candidate. Fusion delta (features (+) mu minus features)
                        against the number of labelled training stars, one line per printable task,
                        seed-paired 2*SE error bars, a zero line. This is the figure S1-E1 is scored
                        on: if the SSL representation is worth more when labels are scarce, these
                        lines rise to the LEFT.
  s1_arm_curves.png     small multiples, one per task, all four arms in absolute score. This is what
                        stops the delta panel being read as a claim about level: a delta can widen
                        because fusion improves or because the engineered baseline collapses, and only
                        the absolute panel distinguishes those.

The x axis is the absolute label count, not the fraction (user decision 2026-09-01). A fraction means
20x different label counts across these tasks -- 1 % is 160 stars for `osc_giant` and 8 for
`rgb_vs_heb` -- so plotting on fraction would put non-comparable budgets on the same tick. The fraction
of each task's train split rides in the CSV and in the delta panel's per-task annotation.

`flare` is drawn in grey and excluded from the S1-E1 count: it is UNPRINTABLE until L1's visual gate
lands (STATUS 2026-08-26d, reporting rule B). Drawing it greyed rather than deleting it is the honest
option -- a reader can see it was measured and see why it is not counted.

Run (repo root, swm env, PYTHONPATH=src; needs analyze_s1_label_efficiency.py to have run):
    python experiments/plot_s1_label_efficiency.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot is imported)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("s1_plot")

repo_root = Path(__file__).resolve().parents[1]
CELL_NAME = "hann0p3_fbwd"
TASK_ORDER = ("pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "flare",
              "rgb_vs_heb", "ijspeert", "numax_hon", "rotation_period")
UNPRINTABLE = {"flare"}
METRIC_LABEL = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "r2": "R2"}
# arm -> (label, colour, linestyle). The untrained pair is dashed: it is a control, not a result.
ARM_STYLE = {
    ("features_only", "features"): ("25 engineered features", "#444444", "-"),
    ("mu", CELL_NAME): ("SSL mu alone", "#1f77b4", "-"),
    ("features_plus_mu", CELL_NAME): ("features (+) mu", "#d62728", "-"),
    ("mu", "untrained"): ("untrained mu alone", "#1f77b4", "--"),
    ("features_plus_mu", "untrained"): ("features (+) untrained mu", "#d62728", "--"),
}


def delta_panel(summary: pd.DataFrame, growth: pd.DataFrame, out_path: Path) -> None:
    """The paper candidate: fusion delta against label count, one line per task.

    Error bars are the seed-paired 2*SE. The draw spread is deliberately NOT drawn: it answers "which
    stars got labelled", a different question from "how much does the representation add", and
    overlaying the two would invite reading one interval as the other. It is in the CSV.
    """
    fusion = summary[(summary["arm_set"] == "features_plus_mu") & (summary["family"] == CELL_NAME)]
    calls = growth.set_index("task")["call"].to_dict()
    plt.figure(figsize=(9, 6))
    for task in TASK_ORDER:
        rows = fusion[fusion["task"] == task].sort_values("n_target")
        if rows.empty:
            continue
        greyed = task in UNPRINTABLE
        if greyed:
            colour, alpha, label = "#999999", 0.5, f"{task} (unprintable, L1 gate open)"
        else:
            colour, alpha, label = None, 1.0, f"{task} [{calls.get(task, '-')}]"
        plt.errorbar(rows["n_target"], rows["delta_mean"], yerr=rows["seed_2se"], marker="o",
                     capsize=2, color=colour, alpha=alpha, label=label)
    plt.axhline(0, color="black", linewidth=1)
    plt.xscale("log")
    plt.xlabel("labelled training stars (log scale)")
    plt.ylabel("fusion delta: score(features + mu) - score(features)")
    plt.title("S1 -- does the fusion advantage grow as labels shrink?\n"
              f"readout `mean`, linear, {CELL_NAME} 6 seeds, error bars 2*SE over encoder seeds")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    log.info(f"wrote {out_path}")


def arm_panel(summary: pd.DataFrame, out_path: Path) -> None:
    """Small multiples: the absolute score of all five arms per task.

    Without this, a widening delta cannot be told from a collapsing baseline -- the delta panel alone
    is ambiguous about which arm moved.
    """
    tasks = []
    for task in TASK_ORDER:
        if not summary[summary["task"] == task].empty:
            tasks.append(task)
    ncol = 4
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for position, task in enumerate(tasks):
        ax = axes[position // ncol][position % ncol]
        block = summary[summary["task"] == task]
        metric = block["metric"].iloc[0]
        for key, (label, colour, style) in ARM_STYLE.items():
            arm_set, family = key
            if arm_set == "features_only":
                rows = block.drop_duplicates("n_target").sort_values("n_target")
                values, errors = rows["features_only"], None
            else:
                rows = block[(block["arm_set"] == arm_set)
                             & (block["family"] == family)].sort_values("n_target")
                values, errors = rows["score_mean"], rows["seed_2se"]
            if rows.empty:
                continue
            ax.errorbar(rows["n_target"], values, yerr=errors, marker="o", markersize=3, capsize=2,
                        color=colour, linestyle=style, label=label)
        ax.set_xscale("log")
        ax.set_title(f"{task} ({METRIC_LABEL.get(metric, metric)})", fontsize=10)
        ax.set_xlabel("labelled training stars")
        if task in UNPRINTABLE:
            ax.set_facecolor("#f0f0f0")
    for position in range(len(tasks), nrow * ncol):
        axes[position // ncol][position % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.04) if len(tasks) % ncol else (0.98, 0.02))
    fig.suptitle("S1 -- absolute score by label budget, five arms "
                 f"(readout `mean`, linear, {CELL_NAME} 6 seeds; grey panel = unprintable)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="S1 label-efficiency figures.")
    ap.add_argument("--in-dir", default="experiments/s1_label_efficiency")
    args = ap.parse_args()
    home = repo_root / args.in_dir
    summary = pd.read_csv(home / "s1_summary.csv")
    growth = pd.read_csv(home / "s1_growth.csv")
    delta_panel(summary, growth, home / "s1_delta_curves.png")
    arm_panel(summary, home / "s1_arm_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
