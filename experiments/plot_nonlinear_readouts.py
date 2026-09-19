"""Fusion gain (features (+) mu minus features) on the 10 paper tasks under three readouts.

Wave-3 (Yue Ma 2026-09-15): the paper's headline is the linear readout; this figure puts the two
gradient-boosted readouts beside it so the reader sees where the gain survives a nonlinear readout on
the same feature sets. Nothing is recomputed: every point is a row of an existing summary CSV.
    linear   experiments/f1_fusion_scorecard/f1_summary.csv     (the paper's Figure 1a)
    gbm      experiments/f1_nonlinear_control/f1_summary.csv    (C3b, sklearn HistGradientBoosting)
    xgb      experiments/f1_xgb_control/f1_summary.csv          (XGBoost, grid + early stopping)
Bars are 2*SE over the six encoder seeds (paired per seed under gbm/xgb); the untrained control is a
single encoder and carries none.

Outputs (experiments/f1_xgb_control/):
    fig_nonlinear_readouts{,_no_untrained,_no_untrained_faded}.{png,pdf}   three figure variants
    table_readouts.tex / table_readouts.png            the write-up table (linear + XGBoost, wave 5)
    table_readouts_all3.tex / table_readouts_all3.png  same with HistGradientBoosting as well
    fig_nonlinear_readouts_data.csv                                         every plotted number

Run (swm env, repo root): python experiments/plot_nonlinear_readouts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

repo_root = Path(__file__).resolve().parents[1]

paper_tasks = ["pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "rgb_vs_heb",
               "numax_hon", "rotation_period", "flare"]
task_labels = {"pulsating": "pulsating", "eb": "eclipsing binary", "rotation": "rotation", "transit": "transit",
               "osc_giant": "oscillating giant", "solar_like_osc": "solar-like osc.", "rgb_vs_heb": "RGB vs. HeB",
               "numax_hon": r"$\nu_{\max}$ ($R^2$)", "rotation_period": r"rotation period ($R^2$)",
               "flare": "flare activity"}
plain_labels = {**{k: v for k, v in task_labels.items()}, "numax_hon": "numax", "rotation_period": "rotation period"}
metric_of = {"numax_hon": "R2", "rotation_period": "R2", "rgb_vs_heb": "ROC-AUC"}
metric_tex = {"R2": "$R^2$", "ROC-AUC": "ROC-AUC", "PR-AUC": "PR-AUC"}
# fixed hue per readout family; blue stays the paper's linear-arm colour
readouts = [("linear", "experiments/f1_fusion_scorecard/f1_summary.csv", "#1f4e79", "linear (paper)"),
            ("gbm", "experiments/f1_nonlinear_control/f1_summary.csv", "#c26a2a", "HistGradientBoosting"),
            ("xgb", "experiments/f1_xgb_control/f1_summary.csv", "#4c9f70", "XGBoost")]
encoder_family = "hann0p3_fbwd"


def load_deltas(path: Path, family: str) -> pd.DataFrame:
    """Fusion-minus-features rows for one readout family at the `mean` pooling, indexed by task."""
    summary = pd.read_csv(path)
    rows = summary[(summary["contrast"] == "fusion_minus_features") & (summary["readout"] == "mean")
                   & (summary["readout_family"] == family) & summary["task"].isin(paper_tasks)]
    trained = rows[rows["family"] == encoder_family].set_index("task")
    untrained = rows[rows["family"] == "untrained"].set_index("task")
    out = pd.DataFrame({"delta": trained["delta_mean"], "delta_2se": trained["delta_2se"],
                        "untrained": untrained["delta_mean"], "features": trained["features_only"]})
    assert set(out.index) == set(paper_tasks), f"{path}: missing tasks {set(paper_tasks) - set(out.index)}"
    return out.loc[paper_tasks]


def draw(tables: dict[str, pd.DataFrame], out_dir: Path, with_untrained: bool,
         fade_nonlinear: bool = False) -> None:
    """One dot-and-bar chart; `with_untrained` adds the x markers of the untrained-encoder control.

    `fade_nonlinear` draws the two boosted readouts at alpha 0.3 so the linear (paper) arm leads the eye
    (Yue Ma 2026-09-16: highlight linear, the others semi-transparent).
    """
    y = np.arange(len(paper_tasks))[::-1]
    offsets = {"linear": 0.25, "gbm": 0.0, "xgb": -0.25}
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for name, _, colour, label in readouts:
        t = tables[name]
        alpha = 0.3 if (fade_nonlinear and name != "linear") else 1.0
        axis.errorbar(t["delta"], y + offsets[name], xerr=t["delta_2se"], fmt="o", markersize=4,
                      color=colour, ecolor=colour, elinewidth=1.2, capsize=2, label=label, alpha=alpha)
        if with_untrained:
            axis.plot(t["untrained"], y + offsets[name], marker="x", linestyle="none", markersize=4.5,
                      color=colour, alpha=0.6,
                      label=f"{label}, untrained encoder" if name == "linear" else None)
    axis.axvline(0, color="black", linewidth=0.7)
    for k in range(len(paper_tasks) - 1):
        axis.axhline(y[k] - 0.5, color="#dddddd", linewidth=0.5)
    axis.set_yticks(y)
    axis.set_yticklabels([task_labels[t] for t in paper_tasks])
    axis.set_xlabel(r"features $\oplus\,\mu$ minus features (PR-AUC, ROC-AUC or $R^2$)")
    title = "Fusion gain under three readouts on the same feature sets; bars are 2 SE over six seeds"
    if with_untrained:
        title += ",\nx marks the untrained-encoder control"
    axis.set_title(title, fontsize=8.5, loc="left")
    axis.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2 if with_untrained else 3,
                frameon=False)
    axis.tick_params(labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    stem = ("fig_nonlinear_readouts" + ("" if with_untrained else "_no_untrained")
            + ("_faded" if fade_nonlinear else ""))
    figure.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    figure.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def tex_cell(delta: float, band: float) -> str:
    """One LaTeX cell: signed gain, bold when it clears its own 2*SE band."""
    body = f"{delta:+.3f}"
    return rf"\textbf{{{body}}}" if abs(delta) > band else body


def counts_of(t: pd.DataFrame) -> str:
    """win / tie / loss under the 2*SE rule (a bar touching zero is a tie)."""
    win = int((t["delta"] > t["delta_2se"]).sum())
    loss = int((t["delta"] < -t["delta_2se"]).sum())
    return f"{win} / {len(t) - win - loss} / {loss}"


def write_table(tables: dict[str, pd.DataFrame], out_dir: Path, names: list[str] | None = None,
                stem: str = "table_readouts") -> None:
    """The write-up table: per task, the engineered-arm score and the fusion gain under each readout.

    LaTeX (booktabs) for the paper and a PNG rendering of the same cells for chat. Bold (LaTeX) or a
    trailing * (PNG) marks a gain beyond 2*SE over the six seeds; the last row counts win / tie / loss.
    `names` picks the readout columns (default all three); Yue Ma wave 5: one boosted readout is
    enough for the table, the two perform alike.
    """
    chosen = [r for r in readouts if names is None or r[0] in names]
    heads = [label for _, _, _, label in chosen]
    eol = r" \\"
    cmid = "".join(rf"\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(chosen)))
    lines = ["% GENERATED by experiments/plot_nonlinear_readouts.py -- do not hand-edit",
             r"\begin{tabular}{ll" + "rr" * len(chosen) + "}", r"\toprule",
             "task & metric & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in heads) + eol,
             cmid,
             " & & " + " & ".join(r"features & $+\mu$ gain" for _ in chosen) + eol, r"\midrule"]
    png_rows = []
    for task in paper_tasks:
        tex_cells, png_cells = [], []
        for name, _, _, _ in chosen:
            t = tables[name].loc[task]
            tex_cells += [f"{t['features']:.3f}", tex_cell(t["delta"], t["delta_2se"])]
            star = "*" if abs(t["delta"]) > t["delta_2se"] else ""
            png_cells += [f"{t['features']:.3f}", f"{t['delta']:+.3f}{star}"]
        metric = metric_of.get(task, "PR-AUC")
        tex_label = task_labels[task].replace(" ($R^2$)", "")
        lines.append(f"{tex_label} & {metric_tex[metric]} & " + " & ".join(tex_cells) + eol)
        png_rows.append([plain_labels[task], metric] + png_cells)
    counts = [counts_of(tables[name]) for name, _, _, _ in chosen]
    lines += [r"\midrule",
              "win / tie / loss & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{c}}}" for c in counts) + eol,
              r"\bottomrule", r"\end{tabular}"]
    (out_dir / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # PNG rendering of the same cells for chat
    short = {"linear (paper)": "linear (paper)", "HistGradientBoosting": "HistGB", "XGBoost": "XGBoost"}
    col_labels = ["task", "metric"] + [f"{short[h]}\n{c}" for h in heads for c in ("features", "+mu gain")]
    png_rows.append(["win / tie / loss", ""] + sum((["", c] for c in counts), []))
    figure, axis = plt.subplots(figsize=(3.6 + 2.1 * len(chosen), 4.6))
    axis.axis("off")
    widths = [0.16, 0.09] + [0.11, 0.11] * len(chosen)
    widths = [w / sum(widths) * 0.98 for w in widths]
    table = axis.table(cellText=png_rows, colLabels=col_labels, loc="center", cellLoc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        text = cell.get_text().get_text()
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        elif text.endswith("*"):
            cell.set_text_props(weight="bold")
        if r == len(png_rows):
            cell.set_facecolor("#f7f7f7")
    axis.set_title("Engineered-feature score and the gain from adding mu, per readout (mean over six seeds).\n"
                   "* = gain beyond 2 SE; last row = win / tie / loss under that rule",
                   fontsize=8.5, loc="left")
    figure.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    tables = {name: load_deltas(repo_root / path, name) for name, path, _, _ in readouts}
    out_dir = repo_root / "experiments" / "f1_xgb_control"
    draw(tables, out_dir, with_untrained=True)
    draw(tables, out_dir, with_untrained=False)  # Yue Ma 2026-09-16: the x markers made the chart too busy
    draw(tables, out_dir, with_untrained=False, fade_nonlinear=True)  # the write-up candidate
    write_table(tables, out_dir, names=["linear", "xgb"])  # Yue Ma wave 5: one boosted readout in the table
    write_table(tables, out_dir, stem="table_readouts_all3")  # the three-readout version, for the record

    # the numbers behind the figure, one row per task with all three readouts side by side
    wide = pd.concat({name: tables[name] for name in tables}, axis=1)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide.to_csv(out_dir / "fig_nonlinear_readouts_data.csv")
    with pd.option_context("display.width", 220, "display.float_format", "{:+.4f}".format):
        print(wide[[c for c in wide.columns if not c.endswith("features")]].to_string())
        print()
        print(wide[[c for c in wide.columns if c.endswith("features")]].to_string())
    for name in tables:
        print(f"{name}: win / tie / loss = {counts_of(tables[name])}")


if __name__ == "__main__":
    main()
