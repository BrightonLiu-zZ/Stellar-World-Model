"""exp09 wave 6: the encoder decision (roadmap D17) and the mechanism multiplier, scored as written.

Wave 6 does not ask exp09's question. exp09's question -- does closing the exploitable spectral channel
restore selectability -- is answered and stays answered: G9-select fails, the void survives, and no cell
cleared the pre-registered G9-artifact bar of 1.5x. This script scores the DOWNSTREAM question the
ML4PS paper needs settled: which encoder ships.

TWO GATES, BOTH NAMED IN EVERY TABLE, NEVER SUBSTITUTED FOR EACH OTHER:
  G9-artifact  <= 1.5x  pre-registered 2026-08-15, governs exp09's MECHANISM verdicts. Untouched.
  G17-artifact <= 2.0x  declared 2026-08-25 BEFORE any wave-6 run, governs the ENCODER CHOICE only.
Recording it this way is the whole point. A threshold dated before the runs is a decision rule; the
same threshold dated after them is gate-shopping. See the gates block of
experiments/configs/exp09_loss_exploit_ladder.yaml for the derivation of 2.0.

THE SWITCH RULE (roadmap D17, pre-registered). Every clause must hold:
  (1) G17-artifact passes on the candidate, and
  (2) the user signs off VISUALLY on a fresh random sample of reconstructions -- the co-equal clause,
      never satisfied by the number alone (the F18 lesson), and NOT scored here, and
  (3) G9-noregress passes at BOTH readouts (`mean` and `mean_std`), and
  (4) >= 2 of the 4 v1 tasks beat exp07_hann0p3_fbwd by more than 2*SE at `mean`, on VALID rows.
G9-dose is applied BEFORE the probe is read, as always: a void row's probe verdict is not evidence in
either direction (wave 5 K3 -- a collapsing seed inflates every paired SE 3-8x, so the 2*SE rule turns
permissive exactly where a cell is failing).

The estimator is PAIRED BY SEED. Measured 2026-08-25, pairing does not reliably reduce variance here
(rho -0.660 on eb; sd(paired)/sd(unpaired) 1.28) because the split is identical across cells and there
is no shared nuisance to cancel -- unpaired, eb would already clear at 6 seeds with no new runs. That
is precisely the estimator swap the VOID rule exists to forbid, so the paired estimator stays: it is
pre-registered, it is used in every published exp09 number, and it happens to be the conservative one.
The honest way to run a 12-seed paired test is to give the reference 12 seeds, which wave 6 did.

Run (repo root, swm env, PYTHONPATH=src), after the artifact and probe fans:
    python experiments/analyze_exp09_wave6_gates.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
CURVES = EXP / "exp09_forensics" / "curves_exp09"

TASKS = ("pulsating", "eb", "rotation", "transit")
READOUTS = ("mean", "mean_std")          # D2 ladder: `mean` headline, `mean_std` second. `mean_resid`
                                         # is appendix-only and is never a gate input.
REF = "exp07_hann0p3_fbwd"
CANDIDATE = "exp09_dpss_impulse_w0p025"
HANN_LADDER = ("exp09_hann_w0p025", "exp09_hann_w0p10")
OFF_ARM = "exp09_dpss_impulse_w0p025_off"

G9_BAR, G17_BAR = 1.5, 2.0
DOSE_BAND = (0.6, 1.4)
# Measured 2026-08-26 on the wave-6 curves, not assumed: seed 6 of the candidate finished at 0 active
# units, KL 0.000, dose 0.000 and val/recon 1.201 against ~0.816 for every healthy seed -- a total
# posterior collapse. Named here so the sensitivity block cannot silently drift to a different seed.
COLLAPSED_SEEDS = (6,)
LAMBDA_DYN = {CANDIDATE: 60.0, REF: 60.0, "exp09_hann_w0p025": 60.0, "exp09_hann_w0p10": 60.0,
              OFF_ARM: 0.0}   # lambda 0 => dose is 0/recon by construction, so G9-dose is inapplicable

# Published wave-1..5 anchors, read from the ladder's own CSV rather than retyped, so a re-run of the
# earlier waves propagates here instead of silently disagreeing with a hardcoded copy.
FLOOR_CELL = "exp09_aux_none"            # the measured no-pressure null of the severity metric
TWIN = {"exp09_hann_w0p025": "exp09_dpss_impulse_w0p025",   # weight-matched dpss+kurtosis twins: each
        "exp09_hann_w0p10": "exp09_dpss_impulse_w0p10"}     # hann cell is a ONE-FACTOR mechanism contrast
W30 = {"hann": "exp07_hann0p3_fbwd", "both": "exp09_aux_dpss_impulse"}  # the same contrast at w=0.3


def load_severity() -> pd.DataFrame:
    """Wave-6 severity runs, plus the published waves 1-5 runs the anchors and twins live in."""
    frames = [pd.read_csv(EXP / f"exp09_impulse_w6_{tag}_runs.csv") for tag in ("cand", "ref", "new")]
    published = pd.read_csv(EXP / "exp09_impulse_runs.csv")
    published["wave"] = "1-5"
    out = pd.concat(frames, ignore_index=True)
    out["wave"] = "6"
    return pd.concat([out, published], ignore_index=True)


def load_probe() -> pd.DataFrame:
    """Wave-6 probe summaries from the three checkpoint arms (each needed its own mu cache dir)."""
    frames = []
    for tag in ("ro", "ref", "new"):
        frames.append(pd.read_csv(EXP / f"exp09_diag_w6_{tag}_probe_summary.csv"))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["cell", "seed", "pooling", "task"])


def curve(cell: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(CURVES / f"{cell}_B_seed{seed}.csv")


def dose_and_activity(cell: str, seeds: list[int]) -> dict:
    """
    Per-seed training dose (lambda * dyn / recon over the last 10 epochs) and active-unit count.

    Dose is the G9-dose gate's quantity and the VOID rule's trigger: a run that finishes with 0 active
    units has bought its clean spectrum by collapsing, and its probe row is not evidence either way.
    """
    lam = LAMBDA_DYN[cell]
    dose, n_active = [], []
    for seed in seeds:
        c = curve(cell, seed)
        dose.append(lam * c.tail(10)["train/dyn"].mean() / c.tail(10)["train/recon"].mean())
        n_active.append(c.tail(10)["val/n_active_units"].mean())
    applicable = lam > 0
    return {"dose": float(np.mean(dose)), "dose_min": float(np.min(dose)),
            "n_active": float(np.mean(n_active)), "n_act_min": float(np.min(n_active)),
            "G9_dose": ("n/a (lambda 0)" if not applicable else
                        "PASS" if all(DOSE_BAND[0] <= d <= DOSE_BAND[1] for d in dose) else "FAIL"),
            "void": bool(applicable and (min(dose) <= 0.0 or min(n_active) <= 0.0))}


def paired_delta(probe: pd.DataFrame, cell: str, task: str, pooling: str, seeds: list[int]) -> dict:
    """Candidate minus reference, paired within seed, over the seeds BOTH arms actually carry."""
    def series(c):
        rows = probe[(probe.cell == c) & (probe.task == task) & (probe.pooling == pooling)]
        return rows[rows.seed.isin(seeds)].set_index("seed").pr_auc
    d = (series(cell) - series(REF)).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"delta": float(d.mean()), "se": se, "n": int(len(d)),
            "ratio": float(d.mean() / (2 * se)) if se > 0 else np.nan}


def noregress(probe: pd.DataFrame, cell: str, pooling: str, seeds: list[int]) -> list[str]:
    """Tasks regressing against the frozen recipe by more than 2*SE. Empty list == the gate passes."""
    bad = []
    for task in TASKS:
        r = paired_delta(probe, cell, task, pooling, seeds)
        if r["delta"] < -2 * r["se"]:
            bad.append(task)
    return bad


def reproduction_checks(sev: pd.DataFrame, probe: pd.DataFrame) -> None:
    """
    The footing. Both re-derive a published number from the wave-6 re-run; a drift here means the
    downstream verdict is not measuring what the published gates were decided on.
    """
    print("=== REPRODUCTION (must hold before anything new is read)")
    a = sev[(sev.wave == "6") & (sev.cell == CANDIDATE) & (sev.seed < 6)].set_index("seed").max_ratio
    b = sev[(sev.wave == "1-5") & (sev.cell == CANDIDATE)].set_index("seed").max_ratio
    diff = float((a - b).abs().max())
    print(f"  severity, {CANDIDATE} seeds 0-5 re-run vs published : max abs diff {diff:.2e} "
          f"-> {'OK' if diff < 1e-9 else 'DRIFT'}")

    gap = pd.read_csv(EXP / "exp07_aux_gap_6seed.csv")
    gap = gap[(gap.arm == "trained") & (gap.cell == REF)]
    merged = probe[(probe.cell == REF) & (probe.seed < 6)].merge(
        gap, on=["cell", "seed", "pooling", "task"], suffixes=("_new", "_pub"))
    diff = float((merged.pr_auc_new - merged.pr_auc_pub).abs().max())
    print(f"  probe,    {REF} seeds 0-5 re-run vs exp07_aux_gap_6seed : {len(merged)} rows, "
          f"max abs diff {diff:.2e} -> {'OK' if diff < 1e-9 else 'DRIFT'}")


def severity_table(sev: pd.DataFrame) -> pd.DataFrame:
    """Every wave-6 cell against BOTH gates, with the position of the maximum kept in view."""
    rows = []
    for cell in (CANDIDATE, OFF_ARM, *HANN_LADDER, REF):
        g = sev[(sev.wave == "6") & (sev.cell == cell)]
        rows.append({
            "cell": cell.replace("exp09_", "").replace("exp07_", ""),
            "n": len(g), "severity": g.max_ratio.mean(), "sev_sd": g.max_ratio.std(),
            # A planted impulse sits at ONE address every seed; noise wanders. The spread of max_pos is
            # therefore a second, independent read on whether the number is measuring an artifact.
            "max_pos_span": f"{int(g.max_pos.min())}-{int(g.max_pos.max())}",
            "centre": g.centre_ratio.mean(), "edge": g.edge_ratio.mean(),
            "G9_artifact(1.5)": "PASS" if g.max_ratio.mean() <= G9_BAR else "fail",
            "G17_artifact(2.0)": "PASS" if g.max_ratio.mean() <= G17_BAR else "fail",
        })
    return pd.DataFrame(rows).set_index("cell")


def multiplier_table(sev: pd.DataFrame) -> pd.DataFrame:
    """
    P11 part 2: how much of the artifact the dpss+kurtosis machinery removes, at each pressure.

    Reported on the FLOOR-SUBTRACTED excess as well as the raw ratio, because the metric's null is
    1.172 and not 1.0: a raw ratio of severities silently credits the mechanism with the floor both
    cells share. Both are printed -- ratio statistics name their numerator and denominator.
    """
    floor = sev[(sev.wave == "1-5") & (sev.cell == FLOOR_CELL)].max_ratio.mean()

    def sev_of(cell):
        g = sev[(sev.wave == "6") & (sev.cell == cell)]
        return float(g.max_ratio.mean()) if len(g) else float(
            sev[(sev.wave == "1-5") & (sev.cell == cell)].max_ratio.mean())

    rows = []
    for weight, hann_cell, both_cell in [(0.025, "exp09_hann_w0p025", TWIN["exp09_hann_w0p025"]),
                                         (0.10, "exp09_hann_w0p10", TWIN["exp09_hann_w0p10"]),
                                         (0.30, W30["hann"], W30["both"])]:
        h, b = sev_of(hann_cell), sev_of(both_cell)
        rows.append({"weight": weight, "hann_only": h, "dpss+kurtosis": b,
                     "mult_raw": h / b, "mult_excess": (h - floor) / (b - floor)})
    out = pd.DataFrame(rows).set_index("weight")
    print(f"\n  (floor = {FLOOR_CELL} severity {floor:.3f}, the metric's measured no-pressure null)")
    return out


def main() -> int:
    sev, probe = load_severity(), load_probe()
    reproduction_checks(sev, probe)

    print("\n=== WAVE 6 SEVERITY | G9-artifact 1.5x pre-registered (mechanism)"
          "\n                    | G17-artifact 2.0x forward-only (encoder choice)")
    print(severity_table(sev).round(3).to_string())

    print("\n=== G9-DOSE, applied BEFORE any probe row is read (VOID rule)")
    seeds12, seeds6 = list(range(12)), list(range(6))
    dose_rows = []
    for cell in (CANDIDATE, REF, *HANN_LADDER, OFF_ARM):
        seeds = seeds12 if cell in (CANDIDATE, REF) else seeds6
        dose_rows.append({"cell": cell.replace("exp09_", "").replace("exp07_", ""),
                          **dose_and_activity(cell, seeds)})
    doses = pd.DataFrame(dose_rows).set_index("cell")
    print(doses.round(3).to_string())
    void = [c for c, r in doses.iterrows() if r["void"]]
    print(f"  VOID rows: {void if void else 'none -- every wave-6 row is valid and readable'}")

    cand_void = bool(doses.loc[CANDIDATE.replace("exp09_", ""), "void"])

    print(f"\n=== SWITCH RULE, clause (4): paired v1 deltas vs {REF} at `mean`, as delta/(2*SE)")
    print("    (>= 2 of 4 tasks above 1.00 required; 6-seed column is the pre-extension footing)")
    rows = []
    for task in TASKS:
        r6 = paired_delta(probe, CANDIDATE, task, "mean", seeds6)
        r12 = paired_delta(probe, CANDIDATE, task, "mean", seeds12)
        rows.append({"task": task, "delta_6": r6["delta"], "ratio_6": r6["ratio"],
                     "delta_12": r12["delta"], "2SE_12": 2 * r12["se"], "ratio_12": r12["ratio"],
                     "clears": "YES" if r12["ratio"] > 1.0 else "no"})
    switch = pd.DataFrame(rows).set_index("task")
    print(switch.round(4).to_string())
    n_clear = int((switch.ratio_12 > 1.0).sum())
    n_up = int((switch.delta_12 > 0).sum())

    print("\n=== SWITCH RULE, clause (3): G9-noregress at BOTH readouts, 12 seeds")
    nore = {}
    for pooling in READOUTS:
        bad = noregress(probe, CANDIDATE, pooling, seeds12)
        nore[pooling] = bad
        print(f"  {pooling:9s}: {'PASS' if not bad else 'FAIL: ' + ','.join(bad)}")

    print("\n=== P11 part 2: the mechanism multiplier at three pressures")
    print(multiplier_table(sev).round(3).to_string())

    if cand_void:
        sensitivity(sev, probe, doses)

    print("\n=== VERDICT")
    cand_sev = sev[(sev.wave == "6") & (sev.cell == CANDIDATE)].max_ratio.mean()
    clause1 = cand_sev <= G17_BAR
    clause3 = all(not v for v in nore.values())
    clause4 = n_clear >= 2
    print(f"  (1) G17-artifact  : severity {cand_sev:.3f} <= {G17_BAR} -> {'PASS' if clause1 else 'FAIL'}")
    print("  (2) visual sign-off: NOT SCORED HERE -- the user's call, and the number never passes alone")
    if cand_void:
        # The VOID rule is applied BEFORE the probe, so clauses (3) and (4) are NOT READ. Printing them
        # above is deliberate -- suppressing them entirely would leave a reader unable to check the
        # rule was followed -- but they are not evidence in either direction and cannot fire a switch.
        print(f"  (3) G9-noregress   : NOT READ -- {CANDIDATE} is VOID on G9-dose "
              f"(printed above as {'PASS' if clause3 else 'FAIL'}, for audit only)")
        print(f"  (4) >=2 of 4 v1    : NOT READ -- same reason "
              f"(printed above as {n_clear}/4, for audit only)")
        print("\n  SWITCH DOES NOT FIRE -> branch P11-C on the encoder question: a seed collapsed and"
              "\n  voided the row on G9-dose. Severity did NOT rise above 2.0 -- this is the OTHER"
              "\n  P11-C trigger, and the row is void before the probe is ever read.")
    else:
        fires = clause1 and clause3 and clause4
        print(f"  (3) G9-noregress   : {'PASS' if clause3 else 'FAIL'} (both readouts)")
        print(f"  (4) >=2 of 4 v1    : {n_clear}/4 clear 2*SE ({n_up}/4 improved) -> "
              f"{'PASS' if clause4 else 'FAIL'}")
        print(f"\n  SWITCH {'FIRES (pending the visual sign-off)' if fires else 'DOES NOT FIRE'} -> "
              f"branch {'P11-A' if fires else 'P11-B'} on the encoder question.")
    print("  hann0p3 SHIPS. NO further seeds either way: 12 was the pre-registered final n"
          " (optional stopping).")
    return 0


def sensitivity(sev: pd.DataFrame, probe: pd.DataFrame, doses: pd.DataFrame) -> None:
    """
    What the collapsed seed is and is not responsible for. REPORTED, NEVER DECIDED ON.

    Dropping the collapsed seed and re-reading the probe is exactly the estimator swap the VOID rule
    forbids: the seed is selected on an outcome (collapse) that is correlated with the probe, after
    seeing the result, and no pre-registration authorised it. It is computed here for one reason only
    -- to establish whether the verdict is HOSTAGE to that choice. If the switch fails under both
    estimators, the reader never has to adjudicate the void question at all.
    """
    kept = [s for s in range(12) if s not in COLLAPSED_SEEDS]
    print(f"\n=== SENSITIVITY: seeds {sorted(COLLAPSED_SEEDS)} dropped from BOTH arms, n={len(kept)}")
    print("    NOT A DECISION INPUT -- post-hoc seed selection on a probe-correlated outcome is the")
    print("    estimator swap the VOID rule exists to forbid. Reported to test whether the verdict")
    print("    depends on it.")
    g = sev[(sev.wave == "6") & (sev.cell == CANDIDATE)]
    print(f"  severity: all 12 {g.max_ratio.mean():.3f} | excl collapsed "
          f"{g[~g.seed.isin(COLLAPSED_SEEDS)].max_ratio.mean():.3f} -> G17 holds either way "
          "(the collapsed seed reconstructs a constant, so its flat error profile sits near the "
          "no-pressure floor and pulls the mean DOWN -- it FLATTERS the artifact gate, not the reverse)")
    rows = []
    for pooling in READOUTS:
        for task in TASKS:
            r = paired_delta(probe, CANDIDATE, task, pooling, kept)
            rows.append({"readout": pooling, "task": task, "delta": r["delta"],
                         "2SE": 2 * r["se"], "ratio": r["ratio"]})
    table = pd.DataFrame(rows).set_index(["readout", "task"])
    print(table.round(4).to_string())
    n_clear = int((table.xs("mean").ratio > 1.0).sum())
    bad_ms = [t for t, r in table.xs("mean_std").iterrows() if r.ratio < -1.0]
    print(f"  even here: clause (4) {n_clear}/4 clear at `mean`; clause (3) at `mean_std` "
          f"{'PASS' if not bad_ms else 'FAILS on ' + ','.join(bad_ms)}")
    print("  => the switch fails under BOTH estimators, so the verdict does not turn on the void call.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    raise SystemExit(main())
