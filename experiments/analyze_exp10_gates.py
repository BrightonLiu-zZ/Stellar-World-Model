"""exp10 G10-valid: is each cell's seed a legitimate run at all, scored BEFORE any probe is read.

THE VOID RULE, and why it is first. A collapsed seed does not merely score badly -- it inflates every
paired standard error it enters (wave-5 K3 measured 3-8x), which turns the 2*SE survival rule
PERMISSIVE exactly where a cell is failing. So validity is decided on training-side evidence alone,
and a void row's probe verdict is not evidence in either direction.

THE THREE CLAUSES (manifest experiments/configs/exp10_fusion_spine.yaml, gates block):
  (i)   dose = lambda_dyn * train/dyn / train/recon, averaged over the LAST 10 EPOCHS; the cell mean
        must land in [0.6, 1.4] and no seed may sit below 0.3. Never read early-epoch (F13/F25a: the
        ep-7 pilot read produced a spurious dose shortfall that cost exp05/06 a wrong conclusion).
  (ii)  no seed with val/recon > 1.10 or val/kl_total < 0.1. Both numbers are MEASURED, not assumed:
        the wave-6 collapsed seed read 1.201 / 0.000 against 0.81-0.87 for every healthy seed.
  (iii) median selected epoch >= beta_warmup_epochs + 5 = 15, i.e. the checkpoint is not a warmup
        transient. Selection is on val/monitor_recon_only, the primary read for exp10 (D-E10.9).

n_active_units is NOT consulted. It was retired in exp09: the hinge cells' extra units were dilution,
not capacity (KL/unit 0.03-0.06 against 0.20-0.38 elsewhere), and across 14 cells rho(eb, n_active) was
+0.21 at p=0.46.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, reads the dumped W&B curves):
    python experiments/analyze_exp10_gates.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp10_gates")

curves_dir = repo_root / "experiments" / "exp10_forensics" / "curves_exp10"
reference_curves = repo_root / "experiments" / "exp09_forensics" / "curves_exp09"

cells = {"exp10_cond_dec": 60.0, "exp10_decorr": 60.0, "exp10_multistep": 20.0} # cell -> lambda_dyn
reference = ("exp07_hann0p3_fbwd", 60.0)
seeds = [0, 1, 2, 3, 4, 5]
dose_band = (0.6, 1.4)
dose_seed_floor = 0.3
recon_ceiling = 1.10
kl_floor = 0.1
warmup = 10
min_selected_epoch = warmup + 5
last_n = 10 # epochs averaged for the dose


def curve(cell: str, seed: int) -> pd.DataFrame:
    """One run's per-epoch history, from whichever curve dump holds it."""
    for directory in (curves_dir, reference_curves):
        path = directory / f"{cell}_B_seed{seed}.csv"
        if path.exists():
            return pd.read_csv(path).sort_values("epoch").reset_index(drop=True)
    raise FileNotFoundError(f"no curve dump for {cell} seed {seed}")


def seed_row(cell: str, seed: int, lambda_dyn: float) -> dict:
    """Every G10-valid quantity for one run, plus the verdict its clauses imply.

    The selector differs by arm, and that is the A3 confound the manifest already prices rather than a
    choice made here: exp10 cells are read at `val/monitor_recon_only` (D-E10.9), while the exp07
    reference predates that channel entirely and is read at `val/monitor_recon_aux`. The column used is
    recorded per row so no table can quietly mix the two.
    """
    history = curve(cell, seed)
    tail = history.tail(last_n)
    dose = lambda_dyn * float(tail["train/dyn"].mean()) / float(tail["train/recon"].mean())
    post_warmup = history[history["epoch"] >= warmup]
    selector = "val/monitor_recon_only"
    if selector not in history.columns:
        selector = "val/monitor_recon_aux"
    selected = int(post_warmup.loc[post_warmup[selector].idxmin(), "epoch"])
    val_recon = float(history.loc[history["epoch"] == selected, "val/recon"].iloc[0])
    kl_total = float(history.loc[history["epoch"] == selected, "val/kl_total"].iloc[0])
    return {
        "cell": cell, "seed": seed, "epochs_run": int(history["epoch"].max()) + 1,
        "selector": selector, "dose": dose, "val_recon": val_recon, "kl_total": kl_total,
        "selected_epoch": selected,
        "dose_ok": dose >= dose_seed_floor,
        "collapse_ok": val_recon <= recon_ceiling and kl_total >= kl_floor,
    }


def main() -> int:
    rows = []
    for cell, lambda_dyn in cells.items():
        for seed in seeds:
            rows.append(seed_row(cell, seed, lambda_dyn))
    for seed in seeds:
        rows.append(seed_row(reference[0], seed, reference[1]))
    per_seed = pd.DataFrame(rows)

    summary = []
    for cell in per_seed["cell"].unique():
        block = per_seed[per_seed["cell"] == cell]
        dose_mean = float(block["dose"].mean())
        clause_i = dose_band[0] <= dose_mean <= dose_band[1] and bool(block["dose_ok"].all())
        clause_ii = bool(block["collapse_ok"].all())
        clause_iii = float(block["selected_epoch"].median()) >= min_selected_epoch
        summary.append({
            "cell": cell,
            "dose_mean": dose_mean, "dose_min": float(block["dose"].min()),
            "dose_max": float(block["dose"].max()), "dose_sd": float(block["dose"].std(ddof=1)),
            "val_recon_max": float(block["val_recon"].max()),
            "kl_total_min": float(block["kl_total"].min()),
            "selected_epoch_median": float(block["selected_epoch"].median()),
            "clause_i_dose": clause_i, "clause_ii_collapse": clause_ii, "clause_iii_selection": clause_iii,
            "G10_valid": clause_i and clause_ii and clause_iii,
        })
    gate = pd.DataFrame(summary)

    log.info("per-seed (scatter printed beside every aggregate, D-E10.12):\n"
             + per_seed.to_string(index=False))
    log.info("G10-valid:\n" + gate.to_string(index=False))
    per_seed.to_csv(repo_root / "experiments" / "exp10_valid_per_seed.csv", index=False)
    gate.to_csv(repo_root / "experiments" / "exp10_valid_gate.csv", index=False)
    log.info("wrote experiments/exp10_valid_per_seed.csv and experiments/exp10_valid_gate.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
