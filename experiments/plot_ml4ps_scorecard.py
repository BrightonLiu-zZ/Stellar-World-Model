"""
Build the two printed artifacts of the ML4PS 2026 paper from the cached score tables.
Reads F1's fusion scorecard and the C1/C2 supervised arms, restricts to the 10-task paper menu
(ADR-0010's 11 probes with `ijspeert` dropped as a duplicate of `eb`), and writes Table 1 as LaTeX
plus Figure 1 as PDF.
Nothing is recomputed here: every number traces to a CSV, so the paper cannot drift from the artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg") # headless backend: the interactive one fails to load its DLLs in this shell

import matplotlib.pyplot as plt

# 10-task paper menu, in the print order used by Table 1 and Figure 1.
# v1 variability block first, then the downstream probes (ADR-0010 minus `ijspeert`).
paper_tasks = [
    "pulsating",
    "eb",
    "rotation",
    "transit",
    "osc_giant",
    "solar_like_osc",
    "rgb_vs_heb",
    "numax_hon",
    "rotation_period",
    "flare",
]

task_labels = {
    "pulsating": "pulsating",
    "eb": "eclipsing binary",
    "rotation": "rotation",
    "transit": "transit",
    "osc_giant": "oscillating giant",
    "solar_like_osc": "solar-like osc.",
    "rgb_vs_heb": "RGB vs. HeB",
    "numax_hon": r"$\nu_{\max}$",
    "rotation_period": "rotation period",
    "flare": "flare activity",
}

metric_labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "r2": r"$R^2$"}

# The encoder that ships (D17, closed 2026-08-26): Hann-tapered comb aux loss, forward+backward dynamics.
encoder_family = "hann0p3_fbwd"
dynamics_off_family = "hann0p3_off" # same recipe, latent-dynamics term removed; the Results ablation
readout = "mean" # star-level pooling of per-window mu; `mean_std` is the reported alternative


def load_scorecard(f1_dir: Path, c1c2_dir: Path) -> pd.DataFrame:
    """
    Assemble one row per task carrying every arm printed in Table 1.
    Linear arms come from F1's absolute table, the two supervised arms from the C1/C2 table.
    Returns the frame in `paper_tasks` order with score and 2*SE columns per arm.
    """
    absolute = pd.read_csv(f1_dir / "f1_absolute.csv")
    absolute = absolute[(absolute["readout"] == readout) & absolute["task"].isin(paper_tasks)]
    supervised = pd.read_csv(c1c2_dir / "c1c2_absolute.csv")

    linear_arms = {
        "features": ("features", "features_only"),
        "mu": (encoder_family, "mu"),
        "fusion": (encoder_family, "features_plus_mu"),
        "untrained_fusion": ("untrained", "features_plus_mu"),
        # The dynamics ablation: the same recipe with the forward-backward prediction term removed,
        # 6 seeds like the shipped arm. It backs the Results paragraph that claims the dynamics term
        # is what buys the complementarity, and rides Figure 1a rather than a table we have no room for.
        "dynoff_fusion": (dynamics_off_family, "features_plus_mu"),
    }

    rows = []
    for task in paper_tasks:
        row = {"task": task}
        for arm, (family, arm_set) in linear_arms.items():
            sel = absolute[
                (absolute["family"] == family)
                & (absolute["arm_set"] == arm_set)
                & (absolute["task"] == task)
            ]
            assert len(sel) == 1, f"expected one {arm} row for {task}, got {len(sel)}"
            row[arm] = sel["score_mean"].iloc[0]
            row[f"{arm}_2se"] = sel["score_2se"].iloc[0]
            if arm == "features":
                row["metric"] = sel["metric"].iloc[0]
                row["n_test"] = int(sel["n_test"].iloc[0])
                row["n_test_pos"] = int(sel["n_test_pos"].iloc[0])
        for arm, arm_name in [("c1", "conv_supervised"), ("c2", "mlp_raw")]:
            sel = supervised[(supervised["arm"] == arm_name) & (supervised["task"] == task)]
            assert len(sel) == 1, f"expected one {arm_name} row for {task}, got {len(sel)}"
            row[arm] = sel["score_mean"].iloc[0]
            row[f"{arm}_2se"] = sel["score_2se"].iloc[0]
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    # Prevalence is meaningful only for the detection probes; the two regressions get NaN, never 0.
    prevalence = []
    for _, row in scorecard.iterrows():
        if row["metric"] == "r2":
            prevalence.append(np.nan)
        else:
            prevalence.append(row["n_test_pos"] / row["n_test"])
    scorecard["prevalence"] = prevalence
    return scorecard


def add_deltas(scorecard: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the three contrasts the paper claims on, each with the error bar it is judged at.
    `features` is fitted once and carries no seed spread, so the fusion-minus-features band is the
    mu side's alone; the supervised contrasts are unpaired, so their bands add in quadrature.
    """
    scorecard = scorecard.copy()
    scorecard["d_features"] = scorecard["fusion"] - scorecard["features"]
    scorecard["d_features_2se"] = scorecard["fusion_2se"]
    scorecard["d_features_untrained"] = scorecard["untrained_fusion"] - scorecard["features"]
    scorecard["d_features_dynoff"] = scorecard["dynoff_fusion"] - scorecard["features"]
    scorecard["d_features_dynoff_2se"] = scorecard["dynoff_fusion_2se"]
    scorecard["d_c1"] = scorecard["fusion"] - scorecard["c1"]
    scorecard["d_c1_2se"] = np.sqrt(scorecard["fusion_2se"] ** 2 + scorecard["c1_2se"] ** 2)
    scorecard["d_c2"] = scorecard["fusion"] - scorecard["c2"]
    scorecard["d_c2_2se"] = np.sqrt(scorecard["fusion_2se"] ** 2 + scorecard["c2_2se"] ** 2)
    return scorecard


def verdict_counts(delta: pd.Series, band: pd.Series) -> dict[str, int]:
    """
    Apply the project's standing 2*SE rule to one contrast.
    A gap counts as resolved only if it exceeds the combined 2*SE band; otherwise it is a tie.
    """
    counts = {"ahead": 0, "tied": 0, "behind": 0}
    for value, width in zip(delta, band):
        if value > width:
            counts["ahead"] += 1
        elif value < -width:
            counts["behind"] += 1
        else:
            counts["tied"] += 1
    return counts


def dynamics_ablation(f1_dir: Path) -> pd.DataFrame:
    """
    Price the latent-dynamics term: how much of the fusion gain survives removing it, per task.

    Paired by seed rather than pooled. The two families ran the same six pre-training seed indices, so
    seed k of the shipped recipe and seed k of the dynamics-free twin share their initialisation and
    their batch order, and differencing them cancels that shared noise. Pooling the two arms' spreads
    instead would be the same estimator applied to a design that is not unpaired, and it reports 7 of
    10 rather than 6 -- i.e. the paired form is the conservative one, which is why it is the one the
    Results paragraph quotes.
    """
    summary = pd.read_csv(f1_dir / "f1_summary.csv")
    summary = summary[
        (summary["readout"] == readout)
        & (summary["contrast"] == "fusion_minus_features")
        & summary["task"].isin(paper_tasks)
    ]
    seed_columns = [f"delta_s{index}" for index in range(6)]

    rows = []
    for task in paper_tasks:
        on = summary[(summary["task"] == task) & (summary["family"] == encoder_family)]
        off = summary[(summary["task"] == task) & (summary["family"] == dynamics_off_family)]
        assert len(on) == 1 and len(off) == 1, f"expected one on/off row for {task}"
        per_seed = on[seed_columns].to_numpy()[0] - off[seed_columns].to_numpy()[0]
        drop = float(per_seed.mean())
        band = float(2 * per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
        rows.append(
            {"task": task, "gain_on": on[seed_columns].to_numpy()[0].mean(),
             "gain_off": off[seed_columns].to_numpy()[0].mean(),
             "drop": drop, "drop_2se": band, "resolved": drop > band}
        )
    return pd.DataFrame(rows)


def write_table(scorecard: pd.DataFrame, out_path: Path) -> None:
    """
    Emit Table 1 as a booktabs LaTeX fragment.
    Bold marks an arm separated from its nearest rival by the combined 2*SE band; italic marks a
    leader that is only the point-estimate argmax, so the reader can see which wins are resolved.
    """
    arms = ["features", "mu", "fusion", "c1", "c2"]
    lines = []
    for _, row in scorecard.iterrows():
        scores = []
        for arm in arms:
            scores.append(row[arm])
        order = np.argsort(scores)[::-1]
        best = order[0]
        runner_up = order[1]
        gap = scores[best] - scores[runner_up]
        band = np.sqrt(
            np.nan_to_num(row[f"{arms[best]}_2se"]) ** 2
            + np.nan_to_num(row[f"{arms[runner_up]}_2se"]) ** 2
        )
        resolved = gap > band

        cells = []
        for index, arm in enumerate(arms):
            text = f"{scores[index]:.3f}"
            if index == best and resolved:
                text = r"\textbf{" + text + "}"
            elif index == best:
                text = r"\textit{" + text + "}"
            cells.append(text)

        if np.isnan(row["prevalence"]):
            context = f"{row['n_test']}"
        else:
            context = f"{row['n_test']} ({row['prevalence']:.3f})"
        lines.append(
            f"{task_labels[row['task']]} & {metric_labels[row['metric']]} & {context} & "
            + " & ".join(cells)
            + r" \\"
        )

    header = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{frozen linear readout} & \multicolumn{2}{c}{end-to-end supervised} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        r"task & metric & $n$ (prev.) & features & $\mu$ & features\,$\oplus$\,$\mu$ & Conv1D & MLP \\",
        r"\midrule",
    ]
    footer = [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(header + lines + footer) + "\n", encoding="utf-8")


def plot_deltas(scorecard: pd.DataFrame, out_path: Path, png_path: Path) -> None:
    """
    Draw Figure 1: the two claims as horizontal delta plots against a common zero line.
    Panel (a) is what the learned latent adds to the engineered features, with the untrained-encoder
    control beside it; panel (b) is the fusion probe against the two end-to-end supervised baselines.
    """
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.45), sharey=True)
    positions = np.arange(len(scorecard))[::-1]
    labels = []
    for task in scorecard["task"]:
        labels.append(task_labels[task])

    # Panel (a) carries three encoders against one zero: the shipped recipe, the same recipe with the
    # latent-dynamics term removed, and an untrained encoder. Stacking them here rather than in a
    # second table is what lets the dynamics claim in Results be read off a figure we already print.
    left = axes[0]
    left.errorbar(
        scorecard["d_features"],
        positions + 0.24,
        xerr=scorecard["d_features_2se"],
        fmt="o",
        markersize=3.5,
        capsize=1.8,
        linewidth=0.9,
        color="#1f4e79",
        label=r"trained $\mu$",
    )
    left.errorbar(
        scorecard["d_features_dynoff"],
        positions,
        xerr=scorecard["d_features_dynoff_2se"],
        fmt="^",
        markersize=3.2,
        capsize=1.8,
        linewidth=0.9,
        markerfacecolor="none",
        color="#4c9f70",
        label=r"$\mu$, dynamics removed",
    )
    left.plot(
        scorecard["d_features_untrained"],
        positions - 0.24,
        marker="x",
        linestyle="none",
        markersize=4,
        color="#999999",
        label=r"untrained $\mu$ (control)",
    )
    left.set_title(r"(a) gain over engineered features", fontsize=8)
    left.set_xlabel(r"$\Delta$ score,  (features $\oplus\,\mu$) $-$ features")

    right = axes[1]
    right.errorbar(
        scorecard["d_c1"],
        positions + 0.16,
        xerr=scorecard["d_c1_2se"],
        fmt="o",
        markersize=3.5,
        capsize=1.8,
        linewidth=0.9,
        color="#6a4c93", # NOT panel (a)'s blue: a blue circle must mean the shipped mu arm and only that
        label="vs. supervised Conv1D",
    )
    right.errorbar(
        scorecard["d_c2"],
        positions - 0.16,
        xerr=scorecard["d_c2_2se"],
        fmt="s",
        markersize=3.2,
        capsize=1.8,
        linewidth=0.9,
        color="#c26a2a",
        label="vs. MLP on raw flux",
    )
    right.set_title(r"(b) fusion probe vs. end-to-end supervision", fontsize=8)
    right.set_xlabel(r"$\Delta$ score (fusion $-$ baseline)")

    handles = []
    legend_labels = []
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.grid(axis="x", alpha=0.25, linewidth=0.5)
        axis.set_axisbelow(True)
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        legend_labels.extend(axis_labels)

    # Matplotlib returns each axis's handles in draw order, which interleaves the two panels' entries
    # once the legend wraps to two rows. Reorder so column 1 is panel (a) and column 3 is panel (b).
    wanted = [r"trained $\mu$", r"$\mu$, dynamics removed", r"untrained $\mu$ (control)",
              "vs. supervised Conv1D", "vs. MLP on raw flux"]
    order = [legend_labels.index(label) for label in wanted]
    handles = [handles[index] for index in order]
    legend_labels = [legend_labels[index] for index in order]

    # One shared legend below the panels: per-axis legends land on top of the points.
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=5, # one row: matplotlib fills a wrapped legend column-major, which interleaves the panels

        frameon=False,
        fontsize=7,
    )
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight") # screen copy; LaTeX uses the PDF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-dir", type=Path, default=Path("experiments/f1_fusion_scorecard"))
    parser.add_argument("--c1c2-dir", type=Path, default=Path("experiments/c1c2_supervised"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()

    scorecard = add_deltas(load_scorecard(args.f1_dir, args.c1c2_dir))

    # paper/tables and paper/figures hold ONLY what Overleaf compiles -- the .tex fragment and the
    # .pdf -- so the whole directory can be uploaded without picking through it. Provenance CSVs and
    # the PNG mirror land in paper/build/, which is never uploaded.
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "build").mkdir(parents=True, exist_ok=True)
    write_table(scorecard, args.out_dir / "tables" / "table1_scorecard.tex")
    plot_deltas(scorecard, args.out_dir / "figures" / "fig1_deltas.pdf",
                args.out_dir / "build" / "fig1_deltas.png")
    scorecard.to_csv(args.out_dir / "build" / "table1_data.csv", index=False)

    resolved_over_features = int((scorecard["d_features"] > scorecard["d_features_2se"]).sum())
    print(f"fusion beats features beyond 2*SE on {resolved_over_features} of {len(scorecard)}")
    print("vs C1:", verdict_counts(scorecard["d_c1"], scorecard["d_c1_2se"]))
    print("vs C2:", verdict_counts(scorecard["d_c2"], scorecard["d_c2_2se"]))
    print(f"untrained control negative on {int((scorecard['d_features_untrained'] < 0).sum())}"
          f", max {scorecard['d_features_untrained'].max():+.4f}")

    ablation = dynamics_ablation(args.f1_dir)
    ablation.to_csv(args.out_dir / "build" / "dynamics_ablation.csv", index=False)
    print(f"removing the dynamics term reduces the gain on {int((ablation['drop'] > 0).sum())}"
          f" of {len(ablation)}, beyond 2*SE on {int(ablation['resolved'].sum())}")


if __name__ == "__main__":
    main()
