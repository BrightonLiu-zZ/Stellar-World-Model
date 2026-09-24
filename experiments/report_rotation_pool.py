"""Print the pre-registered reading of the pool-3 rotation rescore (design s7 + sign-off A2).

Reads experiments/rotation_pool/{absolute,summary}.csv only; computes nothing new. For each
(population, task) at the headline cell -- linear family, `mean` readout, hann0p3_fbwd, 6 seeds -- it
prints the three absolute scores, the fusion gain with its 2 SE, the untrained control, the paired
trained-minus-untrained contrast, the fbwd-minus-off contrast, and whether the row COUNTS under the
signed rule. Writes the same table to experiments/rotation_pool/verdict.csv.

    python experiments/report_rotation_pool.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

out_dir = Path(__file__).resolve().parents[1] / "experiments" / "rotation_pool"
CELL = "hann0p3_fbwd"


def pick(frame: pd.DataFrame, **where) -> pd.Series | None:
    """The single row matching every key, or None when the cell was not scored."""
    mask = pd.Series(True, index=frame.index)
    for key, value in where.items():
        mask &= frame[key] == value
    hit = frame[mask]
    assert len(hit) <= 1, f"{where} matches {len(hit)} rows"
    return hit.iloc[0] if len(hit) else None


def main() -> int:
    absolute = pd.read_csv(out_dir / "absolute.csv")
    summary = pd.read_csv(out_dir / "summary.csv")
    rows = []
    for (block, task), _ in absolute.groupby(["block", "task"], sort=False):
        head = {"block": block, "task": task, "readout": "mean", "readout_family": "linear"}
        score = {}
        for arm_set, family in [("features_only", "features"), ("mu", CELL), ("features_plus_mu", CELL)]:
            hit = pick(absolute, **head, arm_set=arm_set, family=family)
            score[arm_set] = hit["score_mean"] if hit is not None else float("nan")
        gain = pick(summary, **head, contrast="fusion_minus_features", family=CELL)
        floor = pick(summary, **head, contrast="fusion_minus_features", family="untrained")
        paired = pick(summary, **head, contrast="fusion_trained_minus_untrained", family=CELL)
        ablate = pick(summary, **head, contrast="fusion_fbwd_minus_off")
        if gain is None:
            continue
        row = {"population": block, "task": task, "n_test": int(gain["n_test"]),
               "features": score["features_only"], "mu": score["mu"], "fusion": score["features_plus_mu"],
               "gain": gain["delta_mean"], "gain_2se": gain["delta_2se"], "gain_beats_2se": bool(gain["beats_2se"])}
        if floor is not None:
            # delta_s0 is init 0, the paper's single untrained arm (footing F2); the mean is over 6 inits
            row.update({"untrained_gain_i0": floor.get("delta_s0", float("nan")),
                        "untrained_gain": floor["delta_mean"], "untrained_2se": floor["delta_2se"],
                        "untrained_beats_2se": bool(floor["beats_2se"])})
        if paired is not None:
            row.update({"trained_minus_untrained": paired["delta_mean"], "tmu_2se": paired["delta_2se"],
                        "tmu_beats_2se": bool(paired["beats_2se"])})
        if ablate is not None:
            row.update({"fbwd_minus_off": ablate["delta_mean"], "fmo_2se": ablate["delta_2se"]})
        # the signed rule: base gain > 2 SE, and for pool-3 rows the untrained control must not also
        # clear 2 SE while trained-minus-untrained must (A2). Pool 1 keeps the ML4PS rule.
        counts = row["gain_beats_2se"]
        if block != "pool1" and floor is not None and paired is not None:
            counts = counts and not row["untrained_beats_2se"] and row["tmu_beats_2se"]
        row["counts"] = bool(counts)
        rows.append(row)
    verdict = pd.DataFrame(rows)
    verdict.to_csv(out_dir / "verdict.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(verdict.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
