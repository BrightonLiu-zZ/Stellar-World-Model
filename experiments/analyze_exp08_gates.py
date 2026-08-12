"""Pre-exp08 CHK-1: the downstream menu on the frozen recipe, against two PRE-REGISTERED gates.

Open question Q1 asks whether the Hann taper's measured cost in the pulsator band (13% worse at
65-260 microHz under a referee neither model trained against, F19) shows up in the probes that read
that band. But the mu caches this scores answer a second question the handoff did not name, and it is
the load-bearing one:

  GATE 0 (transfer)   Does the FROZEN recipe beat an untrained encoder on the asteroseismic block at
                      all? Every asteroseismic number in the project -- numax_hon R2 +0.286, osc_giant
                      +0.098, solar_like_osc, rgb_vs_heb -- was measured on exp05 arms: a different
                      recipe at a 60-epoch budget. F25(a) shows a short-budget run is not a prefix of a
                      long one and that no capacity number survives a budget change, and exp07 is
                      ep100. So the ML4PS scorecard currently quotes a transfer result for a recipe
                      that no longer exists. If Gate 0 fails, Gate 1 is irrelevant: no aux-window
                      redesign fixes an encoder that does not transfer, and exp08 is neither of the
                      branches the handoff pre-supposes.

  GATE 1 (taper cost) Paired hann0p3 - comb0p3, same arm, same seed.

Pre-registration, fixed before any number is read:
  primary       numax_hon (R2). It reads 65-260 microHz directly -- the exact band F19 measured the
                taper degrading -- and being a regression score it carries no base-rate pathology.
  replication   osc_giant, solar_like_osc, rgb_vs_heb, reported with the same gate but NOT as
                independent gates (that would be seven-way multiplicity on 6 seeds).
  arm rule      fbwd decides, because it is the frozen recipe. off is an independent replication and is
                never pooled with it (F21).
  reported      rotation_period, ijspeert, flare are reported and not gated; flare is the project's
                stated null (CLAUDE.md), never a win.
  not reported  numax_hatt, dnu_hatt, prot_kounkel, ijspeert_excl_villanova are computed because the
                scorecard emits them for free, and carried with reportable=False per ADR-0010.

Untrained arms carry SIX random inits (untrained, untrained_s1..s5) rather than the historical single
seed-0 init, so the reference contributes its own variance to every SE instead of the zero F17 warns
about. Trained-vs-untrained is an unpaired two-sample comparison (init seed i has no correspondence to
training seed i), so SE_diff = sqrt(SE_trained^2 + SE_untrained^2); hann-vs-comb IS paired by seed and
uses the SE of the per-seed differences.

Reads experiments/exp08_prechecks/new_task_scorecard.csv (whatever arms it holds).
Writes experiments/exp08_prechecks/gates.csv + menu.csv. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp08_gates.py
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp08_gates")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "exp08_prechecks"

ADR0010_MENU = ["numax_hon", "rotation_period", "osc_giant", "solar_like_osc", "rgb_vs_heb",
                "ijspeert", "flare"]
PRIMARY = "numax_hon"
REPLICATION = ["osc_giant", "solar_like_osc", "rgb_vs_heb"]   # the asteroseismic block minus the primary
ASTEROSEISMIC = [PRIMARY] + REPLICATION
FROZEN, BASELINE = "hann0p3", "comb0p3"
DECIDING_ARM = "fbwd"
# <recipe>_<arm>[_<extra>]_s<seed>; `extra` carries exp05-style lambda tags (comb_fbwd_c1p0_s0) and is
# folded back into the recipe label so two cells of one arm never collapse into one row.
_ARM_RE = re.compile(r"^(?P<recipe>.+?)_(?P<arm>fbwd|fwd|multi|off)(?:_(?P<extra>[^_]+))?_s(?P<seed>\d+)$")


def parse_arm(arm: str) -> dict:
    """Split an arm name into recipe / dynamics arm / seed; untrained arms carry an init seed instead."""
    if arm.startswith("untrained"):
        seed = 0 if arm == "untrained" else int(arm.rsplit("_s", 1)[1])
        return {"recipe": "untrained", "arm": "untrained", "seed": seed, "kind": "untrained"}
    m = _ARM_RE.match(arm)
    assert m is not None, f"cannot parse arm name {arm!r}"
    recipe = m["recipe"] if m["extra"] is None else f"{m['recipe']}_{m['extra']}"
    return {"recipe": recipe, "arm": m["arm"], "seed": int(m["seed"]), "kind": "trained"}


def load_menu(path: Path, pooling: str = "mean") -> pd.DataFrame:
    """Scorecard rows annotated with recipe/arm/seed, one comparable `score` column, ADR-0010 flag.

    The scorecard reports a different metric per task shape -- PR-AUC for detection, ROC-AUC for the
    balanced contrastive task, R2 for regression -- so the comparable column has to be assembled from
    `shape` rather than picked once. Pooling is FIXED to `mean`: F26 shows the val-declared MIL winner
    changes identity across seeds of one cell, so selecting a pooling per cell would put operator
    selection noise inside the gate. MIL on the frozen recipe is Q6 and is decided separately.
    """
    frame = pd.read_csv(path)
    if "pooling" in frame.columns:
        frame = frame[frame["pooling"] == pooling].copy()
    metric_of = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
    assert set(frame["shape"]).issubset(metric_of), f"unknown task shape in {sorted(set(frame['shape']))}"
    frame["metric"] = frame["shape"].map(metric_of)
    frame["score"] = [row[row["metric"]] for _, row in frame.iterrows()]
    meta = pd.DataFrame([parse_arm(a) for a in frame["arm"]])
    # the scorecard's own `arm` column holds the ARM NAME; `arm` below is the DYNAMICS arm. Rename the
    # source column rather than the derived one, so gates.csv keeps `arm` meaning fbwd/off everywhere.
    frame = frame.rename(columns={"arm": "arm_name"})
    frame = pd.concat([frame.reset_index(drop=True), meta], axis=1)
    assert not frame.columns.duplicated().any(), \
        f"duplicate columns after the metadata join: {sorted(frame.columns[frame.columns.duplicated()])}"
    frame["reportable"] = frame["task"].isin(ADR0010_MENU)
    assert frame["score"].notna().all(), "a scorecard row has no score under its shape's metric"
    return frame


def unpaired_gate(trained: np.ndarray, untrained: np.ndarray) -> dict:
    """Two-sample delta with both sides' seed spread entering the SE (F17: never only one side's)."""
    se_t = trained.std(ddof=1) / np.sqrt(len(trained)) if len(trained) > 1 else np.nan
    se_u = untrained.std(ddof=1) / np.sqrt(len(untrained)) if len(untrained) > 1 else np.nan
    delta = float(trained.mean() - untrained.mean())
    se = float(np.sqrt(se_t ** 2 + se_u ** 2))
    return {"delta": delta, "se": se, "n_trained": len(trained), "n_untrained": len(untrained),
            "passes": bool(delta > 2 * se)}


def paired_gate(a: pd.Series, b: pd.Series) -> dict:
    """Paired delta over the seeds both arms share; SE from the per-seed differences."""
    common = sorted(set(a.index) & set(b.index))
    diffs = np.array([a[s] - b[s] for s in common], dtype=float)
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    return {"delta": float(diffs.mean()), "se": se, "n_pairs": len(diffs),
            "fails": bool(diffs.mean() < -2 * se)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered Gate 0 / Gate 1 on the downstream menu.")
    ap.add_argument("--scorecard", default=None, help="Default: experiments/exp08_prechecks/new_task_scorecard.csv")
    ap.add_argument("--out-dir", default=None, help="Default: experiments/exp08_prechecks")
    ap.add_argument("--pooling", default="mean", help="Fixed pooling operator; never val-selected (F26).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard = Path(args.scorecard) if args.scorecard else out_dir / "new_task_scorecard.csv"
    menu = load_menu(scorecard, args.pooling)
    menu.to_csv(out_dir / "menu.csv", index=False)
    log.info(f"{len(menu)} scorecard rows, {menu['task'].nunique()} tasks, "
             f"{menu['arm_name'].nunique()} arms, {menu['recipe'].nunique()} recipes")

    rows = []
    for task in sorted(menu["task"].unique()):
        task_rows = menu[menu["task"] == task]
        untrained = task_rows[task_rows["kind"] == "untrained"]["score"].to_numpy()
        for recipe in sorted(set(task_rows[task_rows["kind"] == "trained"]["recipe"])):
            for arm in ["fbwd", "off"]:
                sel = task_rows[(task_rows["recipe"] == recipe) & (task_rows["arm"] == arm)]
                if sel.empty:
                    continue
                gate0 = unpaired_gate(sel["score"].to_numpy(), untrained) if len(untrained) else {}
                rows.append({"task": task, "recipe": recipe, "arm": arm,
                             "reportable": task in ADR0010_MENU,
                             "role": ("primary" if task == PRIMARY else
                                      "replication" if task in REPLICATION else "reported"),
                             "score_mean": float(sel["score"].mean()),
                             "score_sd": float(sel["score"].std(ddof=1)) if len(sel) > 1 else np.nan,
                             "n_seeds": int(len(sel)),
                             "untrained_mean": float(untrained.mean()) if len(untrained) else np.nan,
                             **{f"gate0_{k}": v for k, v in gate0.items()}})
        # Gate 1: paired frozen - baseline, within arm
        for arm in ["fbwd", "off"]:
            a = task_rows[(task_rows["recipe"] == FROZEN) & (task_rows["arm"] == arm)].set_index("seed")["score"]
            b = task_rows[(task_rows["recipe"] == BASELINE) & (task_rows["arm"] == arm)].set_index("seed")["score"]
            if a.empty or b.empty:
                continue
            g1 = paired_gate(a, b)
            rows.append({"task": task, "recipe": f"{FROZEN}_minus_{BASELINE}", "arm": arm,
                         "reportable": task in ADR0010_MENU,
                         "role": ("primary" if task == PRIMARY else
                                  "replication" if task in REPLICATION else "reported"),
                         **{f"gate1_{k}": v for k, v in g1.items()}})

    gates = pd.DataFrame(rows)
    gates.to_csv(out_dir / "gates.csv", index=False)
    log.info(f"wrote {out_dir/'gates.csv'} and {out_dir/'menu.csv'}")

    # ---- the pre-registered verdict, printed in the order the branch rule reads it
    print("\n" + "=" * 78)
    primary_g0 = gates[(gates.task == PRIMARY) & (gates.recipe == FROZEN) & (gates.arm == DECIDING_ARM)]
    primary_g1 = gates[(gates.task == PRIMARY) & (gates.recipe == f"{FROZEN}_minus_{BASELINE}")
                       & (gates.arm == DECIDING_ARM)]
    if not primary_g0.empty and "gate0_passes" in primary_g0:
        r = primary_g0.iloc[0]
        print(f"GATE 0 (transfer, {FROZEN}_{DECIDING_ARM} vs untrained, {PRIMARY}): "
              f"delta {r['gate0_delta']:+.4f} +/- {r['gate0_se']:.4f} -> "
              f"{'PASS' if r['gate0_passes'] else 'FAIL'}")
    if not primary_g1.empty and "gate1_fails" in primary_g1:
        r = primary_g1.iloc[0]
        print(f"GATE 1 (taper cost, {FROZEN}-{BASELINE}, {DECIDING_ARM} arm, {PRIMARY}): "
              f"delta {r['gate1_delta']:+.4f} +/- {r['gate1_se']:.4f} -> "
              f"{'FAIL (taper costs)' if r['gate1_fails'] else 'HOLD'}")
    print("=" * 78)

    print("\nADR-0010 menu, absolute score (mean over seeds), deciding arm:")
    show = gates[(gates.reportable) & (gates.arm == DECIDING_ARM) & gates.score_mean.notna()]
    print(show.pivot_table(index="task", columns="recipe", values="score_mean").round(4).to_string())
    # seed counts differ by recipe: exp07 extended only the winner and its baseline to 6 seeds, so
    # lpsd0p3 carries 4. Printed next to the scores so a 4-seed column is never read as a 6-seed one.
    print("\nseeds behind each column (NOT equal across recipes):")
    print(show.pivot_table(index="task", columns="recipe", values="n_seeds").astype("Int64").to_string())

    print("\nGate 1 paired deltas, both arms (negative = taper costs):")
    g1 = gates[gates.recipe == f"{FROZEN}_minus_{BASELINE}"]
    if not g1.empty:
        print(g1.pivot_table(index=["task", "role"], columns="arm", values="gate1_delta").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
