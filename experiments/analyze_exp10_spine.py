"""exp10 G10-spine: does a complementarity lever widen the fusion claim under a NONLINEAR readout?

THE GATE, verbatim from the manifest (experiments/configs/exp10_fusion_spine.yaml):
    C3b fan on the cell's mu -- `features (+) mu` minus `features` under GBM at readout `mean`,
    6 seeds paired including the GBM random_states; survive = > 2*SE AND |delta| >= 0.01; small-n
    rows also need a draw/bootstrap CI excluding 0; PASS iff the count exceeds 4 of 11 AND transit
    is among the survivors. `flare` is reported, never claimed (L1).

WHY THE EFFECT FLOOR EXISTS. The bare 2*SE rule minted survivors at deltas of 0.001-0.004 twice during
the forensics (F-A and F-F each did it; F-F's swapped-in row was numax_hon at +0.0015 on an R^2 of
0.92). A rule that promotes an effect three orders of magnitude below the metric's own scale is not
measuring an effect, so D-E10.2 pairs the significance test with a floor of 0.01.

WHY SMALL-N ROWS NEED MORE. F-E measured the DRAW spread against the ENCODER-SEED spread on
`rgb_vs_heb`, `rotation_period` and `ijspeert` and found it 5-15x larger. A 6-seed standard error on
those rows is therefore under-dispersed by roughly an order of magnitude, so a survivor there is not
counted on the seed-SE alone. The bootstrap over the paired per-seed deltas computed here is a
NECESSARY condition, not a sufficient one; any small-n row that survives it is flagged for the full
draw CI rather than silently counted.

THE BASELINE THIS IS SCORED AGAINST is the published C3b count of 4/11 on the shipping recipe, which
the `hann0p3_fbwd` row reproduces here from the same caches - if that row does not come back at 4, the
fan is not measuring what the gate was written about.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, seconds, after the F1 fan):
    python experiments/analyze_exp10_spine.py
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
log = logging.getLogger("exp10_spine")

experiments = repo_root / "experiments"
summary_path = experiments / "exp10_spine" / "f1_summary.csv"

cells = ["exp10_cond_dec", "exp10_decorr", "exp10_multistep"]
void_cells = {"exp10_multistep"} # VOID on G10-valid (dose 0.236); scored, never counted
reference = "hann0p3_fbwd"
effect_floor = 0.01
survivor_bar = 4 # the published C3b count; PASS needs strictly more
required_survivor = "transit"
small_n_tasks = {"rgb_vs_heb", "rotation_period", "ijspeert"} # F-E: draw spread 5-15x the seed spread
bootstrap_draws = 20000
rng_seed = 0


def bootstrap_ci(deltas: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Percentile CI of the mean paired delta, resampling the SEEDS with replacement.
    On six seeds this is a coarse instrument, which is exactly why it is used as a necessary condition
    on the small-n rows rather than as the gate itself.
    """
    rng = np.random.default_rng(rng_seed)
    means = np.zeros(bootstrap_draws)
    for i in range(bootstrap_draws):
        means[i] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> int:
    assert summary_path.exists(), f"missing {summary_path}; run the F1 fan into experiments/exp10_spine first"
    summary = pd.read_csv(summary_path)
    fan = summary[(summary["contrast"] == "fusion_minus_features") & (summary["readout"] == "mean")
                  & (summary["readout_family"] == "gbm") & summary["reportable"]]

    rows = []
    for family in [reference, *cells]:
        block = fan[fan["family"] == family]
        for row in block.itertuples(index=False):
            deltas = np.array([getattr(row, f"delta_s{s}") for s in range(int(row.n_seeds))])
            beats = bool(row.delta_mean > row.delta_2se)
            clears_floor = bool(abs(row.delta_mean) >= effect_floor)
            lo, hi = bootstrap_ci(deltas)
            ci_excludes_zero = bool(lo > 0 or hi < 0)
            is_small_n = row.task in small_n_tasks
            survives = beats and clears_floor and (ci_excludes_zero if is_small_n else True)
            rows.append({
                "family": family, "task": row.task, "metric": row.metric, "n_test": int(row.n_test),
                "delta_mean": float(row.delta_mean), "delta_2se": float(row.delta_2se),
                "beats_2se": beats, "clears_floor": clears_floor,
                "small_n": is_small_n, "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": ci_excludes_zero,
                "survives": survives,
                "per_seed": ", ".join(f"{d}" for d in deltas),
            })
    detail = pd.DataFrame(rows)

    verdicts = []
    for family in [reference, *cells]:
        block = detail[detail["family"] == family]
        survivors = block[block["survives"]]["task"].tolist()
        flagged = block[block["survives"] & block["small_n"]]["task"].tolist()
        passes = len(survivors) > survivor_bar and required_survivor in survivors
        verdicts.append({
            "family": family, "n_tasks": int(len(block)), "n_survivors": int(len(survivors)),
            "survivors": ", ".join(sorted(survivors)) or "none",
            "transit_retained": required_survivor in survivors,
            "small_n_survivors_needing_draw_ci": ", ".join(sorted(flagged)) or "none",
            "G10_spine": "VOID (not evidence)" if family in void_cells else
                         ("PASS" if passes else "FAIL"),
        })
    gate = pd.DataFrame(verdicts)

    log.info("per-task fusion-minus-features under GBM at readout mean:\n"
             + detail.drop(columns=["per_seed"]).to_string(index=False))
    log.info("G10-spine:\n" + gate.to_string(index=False))
    detail.to_csv(experiments / "exp10_spine_gate.csv", index=False)
    gate.to_csv(experiments / "exp10_spine_verdict.csv", index=False)
    log.info("wrote experiments/exp10_spine_gate.csv and experiments/exp10_spine_verdict.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
