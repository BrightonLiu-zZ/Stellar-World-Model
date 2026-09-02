"""C1/C2 -- score the two supervised arms and place them beside the F1 scorecard.

Roadmap rows C1 (D20: supervised Conv1D, the raw-CNN baseline merged with the Y13c supervised
ceiling) and C2 (Y13b: the independent model-family check). Reads the per-run artifacts that
`swm.train.supervised` wrote and emits the three tables the D18 scorecard needs, plus the pilot gate.

    c1c2_probe.csv     one row per (arm, task, seed): the single test score of that run
    c1c2_absolute.csv  per (arm, task): mean, sd, 2*SE over seeds -- the SELECTION quantity (F4)
    c1c2_summary.csv   cross-arm contrasts against F1's linear arms, plus `recovery fraction`

ESTIMATOR CONVENTIONS THAT BIND HERE (each has a history in this project):
  - EVERY cross-arm delta is UNPAIRED and says so. A supervised seed is an init/shuffle seed; an
    encoder seed is a pretraining seed. Pairing them by index would manufacture a correlation that
    does not exist, which is the estimator error C3b caught in F1 running the other way (there the
    fix was TO pair, because there the seeds meant the same thing). SE = sqrt(se_a^2 + se_b^2).
  - `recovery fraction` = (probe - floor) / (supervised - floor), floor metric-native: prevalence for
    PR-AUC, 0.5 for ROC-AUC, 0.0 for R2. Per task, NEVER averaged across the 11 -- they are three
    different metrics on populations spanning 150 to 3,429 test stars. Where the probe beats the
    supervised arm the row is marked `ssl_exceeds_ceiling` rather than reported as ">100%".
    The metric-native floor is used instead of CONTEXT.md's older `supervised-gap fraction`
    (untrained-mu floor) because that floor is a frozen LINEAR readout while this ceiling is
    end-to-end: the ratio would mix capacities, and the gap metric is on record as not
    capacity-invariant. CONTEXT.md carries both terms with that scope written down.
  - every PR-AUC row names its prevalence (R8-F1).
  - runs carrying a `flags` value (selected_first_epoch / selected_at_cap / small_n) keep it on every
    row they touch. A flagged cell is reported, not silently averaged in.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, seconds):
    python experiments/analyze_c1c2_supervised.py --gate     # after the seed-0 pilot wave
    python experiments/analyze_c1c2_supervised.py            # after the full queue
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("c1c2")

MANIFEST = repo_root / "experiments" / "configs" / "c1c2_supervised_baselines.yaml"
F1_ABSOLUTE = repo_root / "experiments" / "f1_fusion_scorecard" / "f1_absolute.csv"
# The F1 arms the supervised rows are placed against. `mean` / `linear` is the D18 headline readout,
# and rule 5 says a table naming a fusion number must name the mu-only and feats-only arms beside it.
F1_REFERENCE = {"features_linear": ("features", "features_only"),
                "mu_linear": ("hann0p3_fbwd", "mu"),
                "fusion_linear": ("hann0p3_fbwd", "features_plus_mu")}
EB_FEATURES_LINEAR = 0.742 # F1 published; the pilot gate's one absolute anchor


def load_runs(root: Path) -> pd.DataFrame:
    """Collect every finished run's result.json into one frame, newest layout only."""
    rows = []
    for path in sorted(root.glob("runs/*/*/seed*/result.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except OSError as err:
            log.error(f"cannot read {path}: {err}")
            raise
    assert rows, f"no result.json under {root / 'runs'}; has the queue run?"
    frame = pd.DataFrame(rows)
    frame["flags"] = frame["flags"].fillna("")
    return frame


def absolute_rows(probe: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, task) absolute score over seeds. Absolute score is the selection quantity (F4)."""
    rows = []
    for (arm, task), group in probe.groupby(["arm", "task"], sort=False):
        values = group["score"].to_numpy(dtype=float)
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        flags = set()
        for value in group["flags"]:
            for flag in str(value).split(";"):
                if flag:
                    flags.add(flag)
        first = group.iloc[0]
        n_pos = int(first["n_test_pos"])
        rows.append({"arm": arm, "roadmap_row": first["roadmap_row"], "block": first["block"],
                     "task": task, "shape": first["shape"], "metric": first["metric"],
                     "n_seeds": len(values), "score_mean": float(values.mean()), "score_sd": sd,
                     "score_2se": 2 * sd / np.sqrt(len(values)) if len(values) > 1 else 0.0,
                     "floor": float(first["floor"]),
                     "n_train": int(first["n_train"]), "n_test": int(first["n_test"]),
                     "n_test_pos": n_pos,
                     "prevalence": (n_pos / int(first["n_test"])) if n_pos >= 0 else np.nan,
                     "median_selected_epoch": float(group["selected_epoch"].median()),
                     "mean_minutes": float(group["minutes"].mean()),
                     "flags": ";".join(sorted(flags))})
    return pd.DataFrame(rows)


def f1_reference(path: Path) -> pd.DataFrame:
    """F1's linear arms at readout `mean`, reduced to one row per (task, reference arm)."""
    frame = pd.read_csv(path)
    frame = frame[(frame["readout"] == "mean") & (frame["readout_family"] == "linear")]
    rows = []
    for name, (family, arm_set) in F1_REFERENCE.items():
        subset = frame[(frame["family"] == family) & (frame["arm_set"] == arm_set)]
        for _, row in subset.iterrows():
            rows.append({"task": row["task"], "reference": name, "ref_mean": float(row["score_mean"]),
                         "ref_2se": float(row["score_2se"]), "ref_n_seeds": int(row["n_seeds"])})
    return pd.DataFrame(rows)


def recovery(probe_mean: float, probe_2se: float, sup_mean: float, sup_2se: float,
             floor: float) -> dict:
    """`recovery fraction` with a conservative interval from both arms' 2*SE, and its own guard rails.

    Undefined when the supervised ceiling is at or below the metric-native floor (the denominator has
    no meaning), and reported as an exceedance rather than a percentage when the probe is above the
    ceiling -- a ratio over 1 invites being read as "SSL is 130% of supervised", which is not a claim
    anybody can defend.
    """
    span = sup_mean - floor
    if span <= 0:
        return {"recovery": np.nan, "recovery_lo": np.nan, "recovery_hi": np.nan,
                "recovery_note": "undefined: supervised ceiling at or below the metric floor"}
    point = (probe_mean - floor) / span
    lo_span = max(sup_mean + sup_2se - floor, 1e-9)
    hi_span = max(sup_mean - sup_2se - floor, 1e-9)
    lo = (probe_mean - probe_2se - floor) / lo_span
    hi = (probe_mean + probe_2se - floor) / hi_span
    note = ""
    if point > 1.0:
        note = "ssl_exceeds_ceiling"
    return {"recovery": point, "recovery_lo": lo, "recovery_hi": hi, "recovery_note": note}


def ceiling_context(absolute: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Best supervised arm per task, used to ANNOTATE the recovery rows rather than to redefine them.

    C1 (`conv_supervised`) stays the designated Ceiling B: swapping the denominator to whichever arm
    won, after seeing which arm won, is the post-hoc estimator swap this project's VOID rule forbids.
    But where C2 scores higher, a recovery ratio computed against C1 understates the labelled ceiling,
    and the row has to say so on its own face -- otherwise `rgb_vs_heb` prints 9.26x with nothing
    beside it to explain that the denominator is a 755-star model that barely clears its floor.
    """
    best = {}
    for task, group in absolute.groupby("task", sort=False):
        top = group.loc[group["score_mean"].idxmax()]
        best[task] = (str(top["arm"]), float(top["score_mean"]))
    return best


def summarize(absolute: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Unpaired supervised-vs-F1 contrasts and the recovery fraction, one row per (arm, task, ref)."""
    rows = []
    best = ceiling_context(absolute)
    for _, sup in absolute.iterrows():
        refs = reference[reference["task"] == sup["task"]]
        for _, ref in refs.iterrows():
            delta = float(ref["ref_mean"]) - float(sup["score_mean"]) # positive = the F1 arm is ahead
            two_se = float(np.sqrt(ref["ref_2se"] ** 2 + sup["score_2se"] ** 2))
            row = {"arm": sup["arm"], "roadmap_row": sup["roadmap_row"], "block": sup["block"],
                   "task": sup["task"], "shape": sup["shape"], "metric": sup["metric"],
                   "reference": ref["reference"], "reference_score": float(ref["ref_mean"]),
                   "supervised_score": float(sup["score_mean"]), "floor": float(sup["floor"]),
                   "delta_ref_minus_supervised": delta, "delta_2se": two_se,
                   "beats_2se": bool(abs(delta) > two_se),
                   "spread_note": "unpaired: supervised seeds are init seeds, encoder seeds are "
                                  "pretraining seeds",
                   "n_test": int(sup["n_test"]), "n_test_pos": int(sup["n_test_pos"]),
                   "prevalence": sup["prevalence"], "flags": sup["flags"]}
            row.update(recovery(float(ref["ref_mean"]), float(ref["ref_2se"]),
                                float(sup["score_mean"]), float(sup["score_2se"]), float(sup["floor"])))
            best_arm, best_score = best[sup["task"]]
            row["best_supervised_arm"] = best_arm
            row["best_supervised_score"] = best_score
            notes = []
            if best_arm != sup["arm"]:
                notes.append(f"{best_arm} reaches {best_score:.3f} on this task, so a ratio against "
                             f"{sup['arm']} understates the labelled ceiling")
            span = float(sup["score_mean"]) - float(sup["floor"])
            ref_span = float(ref["ref_mean"]) - float(sup["floor"])
            if span > 0 and ref_span > 0 and span < 0.25 * ref_span:
                notes.append("denominator span is under a quarter of the probe's own span; the ratio "
                             "is numerically unstable and should not be quoted as a percentage")
            row["ceiling_caveat"] = " | ".join(notes)
            rows.append(row)
    return pd.DataFrame(rows)


def run_gate(manifest: dict, probe: pd.DataFrame) -> bool:
    """Score the pre-registered pilot gate. Every clause is printed, PASS or FAIL, with its number."""
    pilot = manifest["queue"]["pilot"]
    wave = probe[(probe["arm"] == pilot["arm"]) & (probe["seed"] == int(pilot["seed"]))]
    small_n = set()
    for task in manifest["tasks"]:
        if task.get("small_n"):
            small_n.add(task["name"])
    max_epochs = int(manifest["base"]["max_epochs"])

    print(f"\nPILOT GATE -- arm {pilot['arm']}, seed {pilot['seed']}, {len(wave)} of "
          f"{len(manifest['tasks'])} tasks present\n")
    view = wave[["task", "metric", "score", "floor", "selected_epoch", "epochs_ran", "n_train",
                 "minutes", "flags"]].sort_values("task")
    print(view.to_string(index=False))

    clauses = []
    above = wave[wave["score"] <= wave["floor"]]
    clauses.append(("every task strictly above its metric-native floor", above.empty,
                    "all above" if above.empty else f"at or below floor: {above['task'].tolist()}"))

    eb = wave[wave["task"] == "eb"]
    if eb.empty:
        clauses.append((f"eb pr_auc >= {EB_FEATURES_LINEAR}", False, "eb not run"))
    else:
        value = float(eb["score"].iloc[0])
        clauses.append((f"eb pr_auc >= {EB_FEATURES_LINEAR}", value >= EB_FEATURES_LINEAR, f"{value}"))

    numax = wave[wave["task"] == "numax_hon"]
    if numax.empty:
        clauses.append(("numax_hon r2 >= 0.0", False, "numax_hon not run"))
    else:
        value = float(numax["score"].iloc[0])
        clauses.append(("numax_hon r2 >= 0.0", value >= 0.0, f"{value}"))

    gated = wave[~wave["task"].isin(small_n)]
    bad_epoch = gated[(gated["selected_epoch"] == 0) | (gated["selected_epoch"] == max_epochs - 1)]
    clauses.append(("no run selecting at epoch 0 or at the cap (small_n cells exempt)", bad_epoch.empty,
                    "none" if bad_epoch.empty else f"{bad_epoch['task'].tolist()}"))

    print("")
    passed = True
    for name, ok, detail in clauses:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  --  {detail}")
        passed = passed and ok
    print(f"\n  remaining clause is manual: inspect the train/val curves for the W13 criterion "
          f"(decent size, stable loss, no overfit).")
    print(f"\nPILOT GATE {'PASS' if passed else 'FAIL'} on the automated clauses\n")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the C1/C2 supervised baselines.")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--root", default=None, help="default: the manifest's paths.root")
    ap.add_argument("--gate", action="store_true", help="score the pre-registered pilot gate and stop")
    args = ap.parse_args()

    try:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except OSError as err:
        log.error(f"cannot read manifest {args.manifest}: {err}")
        raise
    root = Path(args.root) if args.root else repo_root / manifest["paths"]["root"]

    probe = load_runs(root)
    probe.to_csv(root / "c1c2_probe.csv", index=False)
    log.info(f"wrote {root / 'c1c2_probe.csv'} ({len(probe)} runs)")

    if args.gate:
        return 0 if run_gate(manifest, probe) else 1

    absolute = absolute_rows(probe)
    absolute.to_csv(root / "c1c2_absolute.csv", index=False)
    summary = summarize(absolute, f1_reference(F1_ABSOLUTE))
    summary.to_csv(root / "c1c2_summary.csv", index=False)
    log.info(f"wrote c1c2_absolute.csv ({len(absolute)} rows) and c1c2_summary.csv ({len(summary)} rows)")

    print("\nabsolute test score per arm (mean over seeds), readout: end-to-end supervised, "
          "arm set: {conv_supervised, mlp_raw}:")
    print(absolute.pivot_table(index=["block", "task"], columns="arm", values="score_mean")
          .round(4).to_string())
    print("\nrecovery fraction vs the fusion arm (features (+) mu, linear, readout `mean`), per task:")
    view = summary[(summary["reference"] == "fusion_linear") & (summary["arm"] == "conv_supervised")]
    print(view[["task", "supervised_score", "reference_score", "recovery", "recovery_lo",
                "recovery_hi", "recovery_note", "flags"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
