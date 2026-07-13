"""Aggregate the exp02 objective sweep into one ranked, pretrain-once comparison table.

Each variant's skyline run writes experiments/<exp>/results/skyline_gate.csv (one row per task). This
module collects those gate files across every exp02* variant plus the exp01 baseline, and ranks the
variants by the mean trained-untrained PR-AUC gap over the gated tasks (pulsating, eb, rotation) - the
honest self-supervision signal, not absolute PR-AUC, which exp01 showed is capacity-confounded. transit
is carried through as report-only. A variant is flagged as regressing when any gated task's trained-
untrained gap falls materially below the exp01 baseline, so the pretrain-once winner is the one lifting
the aggregate without sacrificing a task.

Run (from repo root, PYTHONPATH=src, swm env):
    python -m swm.eval.sweep_summary
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_TASKS = ("pulsating", "eb", "rotation") # transit is report-only, excluded from the aggregate
BASELINE_EXP = "exp01_window256_seq16"
REGRESSION_MARGIN = 0.01 # a gated task counts as regressed if its gap drops this far below the baseline


def load_gate(exp_dir: Path) -> pd.DataFrame | None:
    """
    Read one experiment's skyline_gate.csv into a per-task frame, or None when the variant has no gate yet.
    The gate file is the per-task decision table skyline writes; a missing file means that variant has not
    been evaluated, so the caller skips it rather than failing the whole sweep.
    """
    gate_path = exp_dir / "results" / "skyline_gate.csv"
    if not gate_path.exists():
        return None
    frame = pd.read_csv(gate_path)
    frame["exp_name"] = exp_dir.name
    return frame


def summarize(root: Path = REPO_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the per-variant ranking and the long per-(variant, task) table from all exp02* + baseline gates.
    Ranking key is the mean trained-untrained gap over GATE_TASKS; each variant also carries its per-task
    gaps, the GBM-on-trained-mu mechanism readout, and a regressed flag versus the exp01 baseline gaps.
    Returns (ranked_summary, long_table).
    """
    exp_dirs = [root / "experiments" / BASELINE_EXP]
    for path in sorted((root / "experiments").glob("exp02*")):
        if path.is_dir():
            exp_dirs.append(path)

    long_parts = []
    for exp_dir in exp_dirs:
        gate = load_gate(exp_dir)
        if gate is not None:
            long_parts.append(gate)
    assert long_parts, "no skyline_gate.csv found for the baseline or any exp02 variant"
    long = pd.concat(long_parts, ignore_index=True)

    baseline = long[long["exp_name"] == BASELINE_EXP]
    baseline_gap = {}
    for row in baseline.itertuples(index=False):
        baseline_gap[row.task] = row.trained_minus_untrained

    summary_rows = []
    for exp_name, group in long.groupby("exp_name"):
        gated = group[group["task"].isin(GATE_TASKS)]
        mean_gap = float(gated["trained_minus_untrained"].mean())
        regressed_tasks = []
        for row in gated.itertuples(index=False):
            if row.task in baseline_gap and row.trained_minus_untrained < baseline_gap[row.task] - REGRESSION_MARGIN:
                regressed_tasks.append(row.task)
        record = {"exp_name": exp_name, "mean_gated_gap": mean_gap, "regressed": ";".join(regressed_tasks)}
        for row in group.itertuples(index=False):
            record[f"gap_{row.task}"] = row.trained_minus_untrained
            record[f"trainmu_gbm_{row.task}"] = row.trained_mu_gbm
        summary_rows.append(record)

    ranked = pd.DataFrame(summary_rows).sort_values("mean_gated_gap", ascending=False).reset_index(drop=True)
    return ranked, long


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Aggregate the exp02 objective sweep")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    ranked, long = summarize(args.root)
    out_path = args.root / "experiments" / "exp02_sweep_summary.csv"
    ranked.to_csv(out_path, index=False)
    log.info(f"wrote sweep ranking to {out_path}")
    log.info(f"ranked variants (by mean trained-untrained gap over {GATE_TASKS}):\n{ranked}")


if __name__ == "__main__":
    main()
