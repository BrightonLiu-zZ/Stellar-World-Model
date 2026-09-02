"""exp10 G10-noregress: does a complementarity lever cost anything on the four v1 tasks?

THE GATE (manifest experiments/configs/exp10_fusion_spine.yaml). Per cell, against
`exp07_hann0p3_fbwd` seeds 0-5, PAIRED BY SEED, over the 4 v1 tasks x {mean, mean_std}: FAIL if any
task regresses beyond 2*SE at EITHER readout. Same form as exp09's G9-noregress, same estimator.

WHY BOTH READOUTS. exp09 wave 7 failed on `pulsating` at ONE readout only, and a single-readout gate
would have passed a cell that the second readout convicted. `mean_resid` is appendix-only and is never
a gate input (it is amplitude-residualized `mean`, not a pooling).

THE PAIRED ESTIMATOR IS KEPT EVEN THOUGH IT IS THE CONSERVATIVE ONE. Measured in exp09: pairing does
not reliably reduce variance here, because the split is identical across cells and there is no shared
nuisance to cancel. Swapping to the unpaired estimator after seeing the result is precisely the move
the VOID rule exists to forbid, so the pre-registered paired estimator stands.

THE A3 CONFOUND IS PRICED, NOT HIDDEN. exp10 cells are read at `best_recon_only` (D-E10.9); the exp07
reference seeds predate that checkpoint and are read at `best_recon_aux`. Measured size of that
difference elsewhere in exp10: mean |d| 0.0006-0.0008, max 0.0047. It is cited on every table rather
than assumed away.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, seconds):
    python experiments/analyze_exp10_noregress.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp10_noregress")

experiments = repo_root / "experiments"
cell_summary = experiments / "exp10_diag_probe_summary.csv"
reference_summary = experiments / "exp09_diag_w6_ref_probe_summary.csv"

reference = "exp07_hann0p3_fbwd"
cells = ["exp10_cond_dec", "exp10_decorr", "exp10_multistep"] # multistep is VOID on G10-valid; labelled
void_cells = {"exp10_multistep"}
tasks = ("pulsating", "eb", "rotation", "transit")
readouts = ("mean", "mean_std")
seeds = [0, 1, 2, 3, 4, 5]


def paired_delta(probe: pd.DataFrame, cell: str, task: str, readout: str) -> dict:
    """Cell minus reference on the seeds BOTH arms carry, differenced within seed before averaging."""
    def series(name: str) -> pd.Series:
        rows = probe[(probe["cell"] == name) & (probe["task"] == task) & (probe["pooling"] == readout)]
        rows = rows[rows["seed"].isin(seeds)]
        return rows.set_index("seed")["pr_auc"].sort_index()

    delta = (series(cell) - series(reference)).dropna()
    assert len(delta) == len(seeds), f"{cell}/{task}/{readout}: paired on {len(delta)} seeds, want {len(seeds)}"
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {"cell": cell, "task": task, "readout": readout, "n_seeds": int(len(delta)),
            "delta_mean": float(delta.mean()), "delta_sd": float(delta.std(ddof=1)), "se": se,
            "regresses": bool(delta.mean() < -2 * se),
            "per_seed": ", ".join(f"{v}" for v in delta.to_numpy())}


def main() -> int:
    assert cell_summary.exists(), f"missing {cell_summary}; run the exp10 diagnostics mu+stars stages first"
    probe = pd.concat([pd.read_csv(cell_summary), pd.read_csv(reference_summary)], ignore_index=True)
    probe = probe.drop_duplicates(["cell", "seed", "pooling", "task"])

    rows = []
    for cell in cells:
        for readout in readouts:
            for task in tasks:
                rows.append(paired_delta(probe, cell, task, readout))
    result = pd.DataFrame(rows)

    verdicts = []
    for cell in cells:
        block = result[result["cell"] == cell]
        bad = block[block["regresses"]]
        verdicts.append({
            "cell": cell,
            "regressing_rows": "; ".join(f"{r.task}@{r.readout}" for r in bad.itertuples()) or "none",
            "worst_delta": float(block["delta_mean"].min()),
            "G10_noregress": "VOID (not evidence)" if cell in void_cells else
                             ("PASS" if len(bad) == 0 else "FAIL"),
        })
    gate = pd.DataFrame(verdicts)

    log.info("paired v1 deltas vs the frozen recipe (positive = the lever helps):\n"
             + result.drop(columns=["per_seed"]).to_string(index=False))
    log.info("G10-noregress:\n" + gate.to_string(index=False))
    result.to_csv(experiments / "exp10_noregress.csv", index=False)
    gate.to_csv(experiments / "exp10_noregress_gate.csv", index=False)
    log.info("wrote experiments/exp10_noregress.csv and experiments/exp10_noregress_gate.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
