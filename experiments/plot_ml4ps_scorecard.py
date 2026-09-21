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


def load_scorecard(f1_dir: Path, c1c2_dir: Path, xgb_dir: Path | None = None) -> pd.DataFrame:
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
    if xgb_dir is not None:
        # The same three frozen arms scored with the XGBoost readout (experiments/f1_xgb_control).
        xgb = pd.read_csv(xgb_dir / "f1_absolute.csv")
        xgb = xgb[(xgb["readout"] == readout) & (xgb["readout_family"] == "xgb")]
        xgb_arms = {"xgb_features": ("features", "features_only"), "xgb_mu": (encoder_family, "mu"),
                    "xgb_fusion": (encoder_family, "features_plus_mu")}
        for arm, (family, arm_set) in xgb_arms.items():
            sel = xgb[(xgb["family"] == family) & (xgb["arm_set"] == arm_set)].set_index("task")
            assert set(paper_tasks) <= set(sel.index), f"xgb {arm} is missing tasks"
            scorecard[arm] = sel.loc[paper_tasks, "score_mean"].to_numpy()
            scorecard[f"{arm}_2se"] = sel.loc[paper_tasks, "score_2se"].to_numpy()
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
    Emit Table 1 as a booktabs LaTeX fragment: linear readout | XGBoost readout | end-to-end.

    Bold marks the best point estimate among the three arms under each frozen readout and nothing
    else (Yue Ma, wave 6, 2026-09-17): no significance test lives in this table -- the 2*SE bands are
    Figure 1's job -- and the supervised arms are never bold. Column padding is tightened to 3pt
    because eleven columns at the default 6pt overrun the text width by 43pt.
    """
    groups = [["features", "mu", "fusion"], ["xgb_features", "xgb_mu", "xgb_fusion"]]
    lines = []
    for _, row in scorecard.iterrows():
        cells = []
        for arms in groups:
            scores = [row[arm] for arm in arms]
            best = int(np.argmax(scores))
            for index, score in enumerate(scores):
                text = f"{score:.3f}"
                cells.append(r"\textbf{" + text + "}" if index == best else text)
        cells += [f"{row['c1']:.3f}", f"{row['c2']:.3f}"]

        if np.isnan(row["prevalence"]):
            context = f"{row['n_test']}"
        else:
            context = f"{row['n_test']} ({row['prevalence']:.3f})"
        lines.append(
            f"{task_labels[row['task']]} & {metric_labels[row['metric']]} & {context} & "
            + " & ".join(cells)
            + r" \\"
        )

    arm_heads = r"features & $\mu$ & features\,$\oplus$\,$\mu$"
    header = [
        r"% GENERATED by experiments/plot_ml4ps_scorecard.py -- do not hand-edit",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{frozen linear readout} & \multicolumn{3}{c}{frozen XGBoost readout}"
        r" & \multicolumn{2}{c}{end-to-end supervised} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}",
        rf"task & metric & $n$ (prev.) & {arm_heads} & {arm_heads} & Conv1D & MLP \\",
        r"\midrule",
    ]
    footer = [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(header + lines + footer) + "\n", encoding="utf-8")


def gain_cell(delta: float, band: float | None = None) -> str:
    """A signed gain for the appendix table; bold (sign included) when it clears its own 2*SE band."""
    sign, digits = ("-" if delta < 0 else "+"), f"{abs(delta):.3f}"
    if band is not None and abs(delta) > band:
        return rf"$\boldsymbol{{{sign}}}\mathbf{{{digits}}}$"  # \mathbf alone leaves the sign light
    return f"${{{sign}}}{digits}$"


def retained_cell(full: float, ablated: float) -> str:
    """
    The share of the full mu gain that survives removing the GRU prediction term, per task.

    Defined only where the full gain is a gain: a ratio of two negative numbers would read as
    "most of it retained" on a row where there is nothing to retain, which is how the deleted
    summed-gain claim went wrong in the first place. A negative value means the gain did not
    merely shrink, it changed sign.
    """
    if full <= 0:
        return "--"
    return f"${ablated / full:.2f}$"


def write_readout_table(scorecard: pd.DataFrame, xgb_gain: pd.DataFrame, out_path: Path) -> None:
    """
    Emit the Appendix B table: what each set of 128 columns adds to the 25 features, per readout.

    Linear: the trained mu, its dynamics-free twin and the untrained control -- the two controls that
    left Figure 1a in wave 5 and that the Results text still quotes. XGBoost: the trained mu and the
    untrained control (the dynamics-free twin was never scored under XGBoost). The absolute scores
    live in Table 1, so this table is gains only. The last row counts win / tie / loss at 2*SE.
    """
    assert list(xgb_gain["task"]) == list(scorecard["task"])
    lines = []
    for (_, row), (_, xgb) in zip(scorecard.iterrows(), xgb_gain.iterrows()):
        cells = [
            gain_cell(row["d_features"], row["d_features_2se"]),
            gain_cell(row["d_features_dynoff"], row["d_features_dynoff_2se"]),
            retained_cell(row["d_features"], row["d_features_dynoff"]),
            gain_cell(row["d_features_untrained"]),
            gain_cell(xgb["delta_mean"], xgb["delta_2se"]),
            gain_cell(xgb["untrained_delta"]),
        ]
        lines.append(f"{task_labels[row['task']]} & {metric_labels[row['metric']]} & " + " & ".join(cells) + r" \\")

    def tally(delta: pd.Series, band: pd.Series) -> str:
        counts = verdict_counts(delta, band)
        return f"{counts['ahead']} / {counts['tied']} / {counts['behind']}"

    header = [
        r"% GENERATED by experiments/plot_ml4ps_scorecard.py -- do not hand-edit",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{linear readout} & \multicolumn{2}{c}{XGBoost readout} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r"task & metric & $+\mu$ & $+\mu$, no prediction term & retained & $+$untrained $\mu$ "
        r"& $+\mu$ & $+$untrained $\mu$ \\",
        r"\midrule",
    ]
    footer = [
        r"\midrule",
        "win / tie / loss & & "
        + tally(scorecard["d_features"], scorecard["d_features_2se"]) + " &"
        + " & & & "
        + tally(xgb_gain["delta_mean"], xgb_gain["delta_2se"]) + r" & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.write_text("\n".join(header + lines + footer) + "\n", encoding="utf-8")


def write_unseen_table(unseen_csv: Path, out_path: Path, seen_definition: str = "fit+val") -> None:
    """
    Emit the Appendix F table from experiments/analyze_unseen_rescore.py's output: the fusion gain on
    all test stars and on those never seen in pre-training (absent from its training and validation
    splits). Saved per-star predictions are filtered, nothing is refitted.
    """
    rows = pd.read_csv(unseen_csv)
    rows = rows[rows["seen_definition"] == seen_definition].set_index("task").loc[paper_tasks]
    lines = []
    for task, row in rows.iterrows():
        metric = {"numax_hon": "r2", "rotation_period": "r2", "rgb_vs_heb": "roc_auc"}.get(task, "pr_auc")
        lines.append(
            f"{task_labels[task]} & {metric_labels[metric]} & {int(row['n_test'])} & {int(row['n_unseen'])} & "
            + gain_cell(row["gain_all"], row["gain_all_2se"]) + " & "
            + gain_cell(row["gain_unseen"], row["gain_unseen_2se"]) + r" \\"
        )
    header = [
        r"% GENERATED by experiments/plot_ml4ps_scorecard.py -- do not hand-edit",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"task & metric & test stars & unseen test stars & gain, all & gain, unseen \\",
        r"\midrule",
    ]
    out_path.write_text("\n".join(header + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


def load_xgb_gain(xgb_dir: Path) -> pd.DataFrame:
    """
    The fusion-minus-features gain under the XGBoost readout (experiments/f1_xgb_control), the
    nonlinear companion to panel (a). Paired per seed over the same six encoders, so its 2*SE band
    is read the same way as the linear arm's. Returned in `paper_tasks` order.
    """
    summary = pd.read_csv(xgb_dir / "f1_summary.csv")
    rows = summary[
        (summary["contrast"] == "fusion_minus_features")
        & (summary["readout"] == readout)
        & (summary["readout_family"] == "xgb")
        & summary["task"].isin(paper_tasks)
    ]
    trained = rows[rows["family"] == encoder_family].set_index("task")
    untrained = rows[rows["family"] == "untrained"].set_index("task")
    assert set(trained.index) == set(paper_tasks), f"xgb summary missing {set(paper_tasks) - set(trained.index)}"
    gain = trained.loc[paper_tasks, ["delta_mean", "delta_2se"]].copy()
    gain["untrained_delta"] = untrained.loc[paper_tasks, "delta_mean"]
    return gain.reset_index()


def plot_deltas(scorecard: pd.DataFrame, out_path: Path, png_path: Path,
                xgb_gain: pd.DataFrame | None = None, figsize: tuple[float, float] = (6.5, 2.45),
                font_size: float = 8) -> None:
    """
    Draw Figure 1: the two claims as horizontal delta plots against a common zero line.
    Panel (a) is what the learned latent adds to the engineered features, with the untrained-encoder
    control beside it; panel (b) is the fusion probe against the two end-to-end supervised baselines.

    With `xgb_gain` (Yue Ma, wave 5, 2026-09-16) panel (a) drops the two controls that are not the
    paper's claim -- the dynamics-free twin and the untrained encoder -- and instead shows the same
    gain under the XGBoost readout at alpha 0.3, so the linear (paper) arm leads the eye and the
    reader sees the gain was also tried under a nonlinear readout.
    """
    plt.rcParams.update({"font.family": "serif", "font.size": font_size, "axes.linewidth": 0.6})
    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    positions = np.arange(len(scorecard))[::-1]
    labels = []
    for task in scorecard["task"]:
        labels.append(task_labels[task])

    # Panel (a) carries three encoders against one zero: the shipped recipe, the same recipe with the
    # latent-dynamics term removed, and an untrained encoder. Stacking them here rather than in a
    # second table is what lets the dynamics claim in Results be read off a figure we already print.
    left = axes[0]
    pair_offset = 0.11 # half the gap between the two points of a row; small enough that a pair stays inside its band
    linear_offset = 0.24 if xgb_gain is None else pair_offset
    left.errorbar(
        scorecard["d_features"],
        positions + linear_offset,
        xerr=scorecard["d_features_2se"],
        fmt="o",
        markersize=3.5,
        capsize=1.8,
        linewidth=0.9,
        color="#1f4e79",
        label=r"trained $\mu$, linear readout" if xgb_gain is not None else r"trained $\mu$",
    )
    if xgb_gain is None:
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
    else:
        assert list(xgb_gain["task"]) == list(scorecard["task"])
        left.errorbar(
            xgb_gain["delta_mean"],
            positions - pair_offset,
            xerr=xgb_gain["delta_2se"],
            fmt="o",
            markersize=3.5,
            capsize=1.8,
            linewidth=0.9,
            color="#1f4e79",
            alpha=0.3,
            label=r"trained $\mu$, XGBoost readout",
        )
    left.set_xticks([-0.05, 0.0, 0.05])  # the default five ticks collide at print size
    left.set_title(r"(a) gain over engineered features", fontsize=font_size)
    left.set_xlabel(r"$\Delta$ score,  (features $\oplus\,\mu$) $-$ features")

    right = axes[1]
    right.errorbar(
        scorecard["d_c1"],
        positions + pair_offset,
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
        positions - pair_offset,
        xerr=scorecard["d_c2_2se"],
        fmt="s",
        markersize=3.2,
        capsize=1.8,
        linewidth=0.9,
        color="#c26a2a",
        label="vs. MLP on raw flux",
    )
    right.set_title(r"(b) fusion probe vs. end-to-end supervision", fontsize=font_size)
    right.set_xlabel(r"$\Delta$ score (fusion $-$ baseline)")

    handles = []
    legend_labels = []
    for axis in axes:
        # Alternate rows shaded (Prof. Theissen, 2026-09-19): each band holds the two offset points of
        # one task, and a faint dotted line runs through each row between its two points, so a pair
        # traces back to its label. Heavier dashed lines were tried and crowded the plot.
        for position in positions:
            if position % 2 == 0:
                axis.axhspan(position - 0.5, position + 0.5, color="#000000", alpha=0.06, linewidth=0, zorder=0)
            axis.axhline(position, color="#bbbbbb", linewidth=0.35, linestyle=(0, (1, 3)), zorder=0)
        axis.set_ylim(positions.min() - 0.5, positions.max() + 0.5)
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
    if xgb_gain is None:
        wanted = [r"trained $\mu$", r"$\mu$, dynamics removed", r"untrained $\mu$ (control)",
                  "vs. supervised Conv1D", "vs. MLP on raw flux"]
    else:
        wanted = [r"trained $\mu$, linear readout", r"trained $\mu$, XGBoost readout",
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
        ncol=len(handles), # one row: matplotlib fills a wrapped legend column-major, which interleaves the panels

        frameon=False,
        fontsize=font_size - 1,
    )
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight") # screen copy; LaTeX uses the PDF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-dir", type=Path, default=Path("experiments/f1_fusion_scorecard"))
    parser.add_argument("--c1c2-dir", type=Path, default=Path("experiments/c1c2_supervised"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper"))
    parser.add_argument("--xgb-dir", type=Path, default=Path("experiments/f1_xgb_control"))
    parser.add_argument("--panel-a", choices=["ablation", "xgb"], default="xgb",
                        help="panel (a) companions: the XGBoost readout at alpha 0.3 (the paper, wave 5 on) or "
                             "the dynamics-off + untrained controls (the pre-wave-5 figure)")
    parser.add_argument("--unseen-csv", type=Path,
                        default=Path("experiments/unseen_pretraining/unseen_rescore.csv"))
    parser.add_argument("--fig-size", type=float, nargs=2, default=[5.5, 1.65], metavar=("W", "H"),
                        help="Figure 1 canvas in inches; fonts are absolute, so a smaller canvas printed at the "
                             "same width reads larger")
    parser.add_argument("--font-size", type=float, default=7)
    args = parser.parse_args()

    scorecard = add_deltas(load_scorecard(args.f1_dir, args.c1c2_dir, args.xgb_dir))
    readout_gain = load_xgb_gain(args.xgb_dir)
    xgb_gain = readout_gain if args.panel_a == "xgb" else None
    fig_stem = "fig1_deltas" if xgb_gain is None else "fig1_deltas_xgb"

    # paper/tables and paper/figures hold ONLY what Overleaf compiles -- the .tex fragment and the
    # .pdf -- so the whole directory can be uploaded without picking through it. Provenance CSVs and
    # the PNG mirror land in paper/build/, which is never uploaded.
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "build").mkdir(parents=True, exist_ok=True)
    write_table(scorecard, args.out_dir / "tables" / "table1_scorecard.tex")
    write_readout_table(scorecard, readout_gain, args.out_dir / "tables" / "table_readout.tex")
    write_unseen_table(args.unseen_csv, args.out_dir / "tables" / "table_unseen.tex")
    plot_deltas(scorecard, args.out_dir / "figures" / f"{fig_stem}.pdf",
                args.out_dir / "build" / f"{fig_stem}.png", xgb_gain=xgb_gain,
                figsize=tuple(args.fig_size), font_size=args.font_size)
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
