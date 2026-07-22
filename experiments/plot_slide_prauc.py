"""Slide figure: frozen linear-probe PR-AUC, untrained vs SSL-trained encoder, over the 3 v1 tasks.

Reads the exp04 3-seed sweep summary and draws the one honest headline chart for a talk: for the
reference recipe `exp03_fb0_b0p1_comb` (free-bits 0, beta 0.1, combined log-PSD+time aux) under the
LOCKED logistic probe, does self-supervised pretraining beat a capacity-matched *untrained* random-init
encoder? This is the only exp04 recipe where all four tasks confirm > 2*SE across seeds, so it is the
one we can quote without the retracted single-seed pulsating headline. transit/eb/pulsating are the v1
primary tasks; rotation is supplementary and left off the slide to keep it to one message.

Numbers come straight from experiments/exp04_sweep_summary.csv (gap_se = 3-seed SE of trained-untrained).
Nothing is recomputed here - this is a presentation view of the already-aggregated table.

Run (from repo root, any env with pandas+matplotlib):
    python experiments/plot_slide_prauc.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = REPO_ROOT / "experiments" / "exp04_sweep_summary.csv"
OUT = REPO_ROOT / "experiments" / "figs" / "prauc_trained_vs_untrained.png"

RECIPE = "exp03_fb0_b0p1_comb"  # only cell with all 4 tasks > 2*SE at 3 seeds (exp04 verdict)
READOUT = "logistic"            # v1 locks the probe to logistic regression - do not swap to gbm
TASKS = ["transit", "eb", "pulsating"]  # v1 primary; rotation is supplementary, off-slide
TASK_LABEL = {"transit": "Transit", "eb": "Eclipsing binary", "pulsating": "Pulsating"}


def main() -> None:
    df = pd.read_csv(SUMMARY)
    sel = df[(df.exp_name == RECIPE) & (df.readout == READOUT) & (df.task.isin(TASKS))]
    sel = sel.set_index("task").loc[TASKS]  # fix task order

    untrained = sel["pr_auc_untrained"].to_numpy()  # (3,) capacity-matched random-init encoder
    trained = sel["pr_auc_mean"].to_numpy()          # (3,) SSL encoder, mean over 3 seeds
    gap = sel["gap_mean"].to_numpy()                 # (3,)
    gap_se = sel["gap_se"].to_numpy()                # (3,) 3-seed SE of the (trained - untrained) gap

    x = range(len(TASKS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    ax.bar([i - w / 2 for i in x], untrained, w, label="Untrained encoder (random init)",
           color="#b9c2cc", edgecolor="#7a8592")
    # error bar on the trained bar = SE of the trained-untrained gap (the quantity we actually test)
    ax.bar([i + w / 2 for i in x], trained, w, label="Self-supervised (trained)",
           color="#2f6db0", edgecolor="#1f4d7a",
           yerr=gap_se, capsize=4, error_kw={"ecolor": "#123", "elinewidth": 1.2})

    # annotate the gap above each pair - this is the headline number
    for i in x:
        top = max(untrained[i], trained[i])
        ax.annotate(f"+{gap[i]:.3f}\n±{gap_se[i]:.3f}",
                    xy=(i, top), xytext=(i, top + 0.045),
                    ha="center", va="bottom", fontsize=9, color="#1f4d7a", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Linear-probe PR-AUC (test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Frozen linear probe: does self-supervision beat an untrained encoder?\n"
                 "TESS light curves · reference recipe · 3 seeds", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    # honest note below the axis (drawn at figure level so it clears the tick labels)
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.015,
             "Transit PR-AUC is near its base rate (~0.06) - weak-signal task; gap is small but seed-confirmed.",
             ha="center", fontsize=7.5, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    for t, u, tr, g, s in zip(TASKS, untrained, trained, gap, gap_se):
        print(f"  {t:10s} untrained={u:.3f} trained={tr:.3f} gap=+{g:.3f} +/-{s:.3f}")


if __name__ == "__main__":
    main()
