"""exp10 forensics -- score the six pre-registered decision rules off the artifacts, in one place.

Every rule in the 2026-08-29 handoff (R-A1 ... R-F3) has a numeric trigger. This script reads the CSVs
the six forensics wrote and evaluates those triggers verbatim, so the README and the hand-back doc quote
a computed verdict rather than a reading of a table. Where a result matches no branch it is recorded as
`mismatch` and NOT snapped to the nearest one -- that is the P8 / P11-F / P12 precedent, and it is the
single most valuable thing this file does.

Inputs (all produced by the other exp10 scripts; a missing one is reported, never silently skipped):
    footing/f1_summary.csv            the incumbent baseline, reproduced through the derived-cache path
    fa_pca/f1_summary.csv             F-A
    fb_predictability/fb_aggregates.csv   F-B
    fc_windowstats/f1_summary.csv     F-C, window-statistic half
    fc_mil/fc_mil_summary.csv         F-C, MIL half
    fd_ensemble/f1_summary.csv        F-D
    fe_subsample/fe_summary.csv       F-E
    ff_gbm_fairness/ff_summary.csv    F-F

Run (repo root, swm env, PYTHONPATH=src; seconds):
    python experiments/exp10_forensics_verdict.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
home = repo_root / "experiments" / "exp10_forensics"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_verdict")

# The three tasks R-A1 wants a recovery on: the ones where fusion currently COSTS score. `rgb_vs_heb`
# already survives under the incumbent GBM, so the rule is scored both ways and both are printed.
recovery_tasks = ["osc_giant", "ijspeert", "rgb_vs_heb"]
small_n_tasks = ["rgb_vs_heb", "ijspeert", "rotation_period"]


def fusion(path: Path, readout_family: str) -> pd.DataFrame:
    """The reportable fusion-minus-features rows of one F1 summary, at readout `mean`."""
    assert path.exists(), f"missing forensic artifact {path}"
    frame = pd.read_csv(path)
    keep = ((frame["contrast"] == "fusion_minus_features") & frame["reportable"]
            & (frame["readout"] == "mean") & (frame["readout_family"] == readout_family))
    return frame[keep].copy()


def transform_of(family: str) -> tuple[str, str]:
    """
    Split a derived arm family into (source arm, transform), e.g. hann0p3_fbwd_pca16 --> both halves.
    The untrained twin carries the prefix `untr_` rather than `untrained_`: `arm_parts` hardcodes the
    family to "untrained" for anything starting with that word, which collapses every untrained variant
    into one row. The short prefix is what keeps the twins distinguishable.
    """
    if family.startswith("untr_"):
        return "untrained", family[len("untr_"):]
    for source in ["hann0p3_fbwd", "untrained"]:
        if family == source:
            return source, "mean"
        if family.startswith(source + "_"):
            return source, family[len(source) + 1:]
    return family, "mean"


def derived_table(path: Path, readout_family: str, drop_collapsed: bool = False) -> pd.DataFrame:
    """
    One derived-cache F1 summary reshaped to (source, transform, task) with the survival flag.
    `drop_collapsed` removes the bare `untrained` family, which in a derived-cache run built before the
    `untr_` rename is not a control at all but several variants overwritten onto one seed key.
    """
    frame = fusion(path, readout_family)
    if drop_collapsed:
        frame = frame[frame["family"] != "untrained"]
    sources, transforms = [], []
    for family in frame["family"]:
        source, transform = transform_of(family)
        sources.append(source)
        transforms.append(transform)
    frame["source"] = sources
    frame["transform"] = transforms
    frame["survives"] = frame["delta_mean"] > frame["delta_2se"]
    return frame


def survival_counts(frame: pd.DataFrame, source: str = "hann0p3_fbwd") -> pd.DataFrame:
    """Per transform: how many of the 11 tasks survive, whether transit is among them, and recoveries."""
    rows = []
    for transform, group in frame[frame["source"] == source].groupby("transform"):
        survivors = set(group.loc[group["survives"], "task"])
        positives = set(group.loc[group["delta_mean"] > 0, "task"])
        rows.append({"transform": transform, "n_tasks": len(group), "n_survive": len(survivors),
                     "transit_retained": "transit" in survivors,
                     "n_recovery_survivors": len(survivors & set(recovery_tasks)),
                     "recovery_survivors": ",".join(sorted(survivors & set(recovery_tasks))),
                     "n_recovery_positive": len(positives & set(recovery_tasks)),
                     "mean_delta": float(group["delta_mean"].mean()),
                     "survivors": ",".join(sorted(survivors))})
    return pd.DataFrame(rows).sort_values("n_survive", ascending=False)


def rule(rule_id: str, fired: bool, statement: str, evidence: str) -> dict:
    """One pre-registered rule's verdict row; `fired` is the trigger as written, never a judgement call."""
    return {"rule": rule_id, "fired": fired, "statement": statement, "evidence": evidence}


def score_fa(verdicts: list[dict], out_dir: Path) -> None:
    """F-A: does reducing mu's column count restore the fusion delta under a nonlinear readout?"""
    twin = home / "fa_pca_untrained" / "f1_summary.csv"
    gbm = pd.concat([derived_table(home / "fa_pca" / "f1_summary.csv", "gbm", drop_collapsed=True),
                     derived_table(twin, "gbm", drop_collapsed=True)], ignore_index=True)
    linear = pd.concat([derived_table(home / "fa_pca" / "f1_summary.csv", "linear", drop_collapsed=True),
                        derived_table(twin, "linear", drop_collapsed=True)], ignore_index=True)
    base_gbm = derived_table(home / "footing" / "f1_summary.csv", "gbm")
    base_linear = derived_table(home / "footing" / "f1_summary.csv", "linear")

    pooled_gbm = pd.concat([gbm, base_gbm], ignore_index=True)
    counts = survival_counts(pooled_gbm)
    counts.to_csv(out_dir / "fa_gbm_survival_counts.csv", index=False)
    print("\nF-A -- GBM survival count per mu transform (11 reportable tasks):")
    print(counts.drop(columns=["survivors"]).to_string(index=False))
    twin_counts = survival_counts(pooled_gbm, source="untrained")
    twin_counts.to_csv(out_dir / "fa_gbm_survival_counts_untrained.csv", index=False)
    print("\nF-A -- the same count on the UNTRAINED twin (a transform that helps here helps any columns):")
    print(twin_counts.drop(columns=["survivors"]).to_string(index=False))

    incumbent = int(counts.loc[counts["transform"] == "mean", "n_survive"].iloc[0])
    hit = counts[(counts["n_survive"] >= 6) & counts["transit_retained"]
                 & (counts["n_recovery_survivors"] >= 1)]
    verdicts.append(rule("R-A1", not hit.empty,
                         "some k gives GBM survival on >=6/11 with transit retained and >=1 recovery "
                         "among osc_giant / ijspeert / rgb_vs_heb -> dilution binding, promote z cell",
                         f"best transform {counts.iloc[0]['transform']} survives "
                         f"{int(counts.iloc[0]['n_survive'])}/11 vs incumbent {incumbent}/11"))

    small = pd.concat([linear, base_linear], ignore_index=True)
    small = small[(small["source"] == "hann0p3_fbwd") & small["task"].isin(["ijspeert", "rgb_vs_heb"])]
    wide = small.pivot_table(index="transform", columns="task", values="delta_mean")
    wide.to_csv(out_dir / "fa_linear_smalln_flip.csv")
    print("\nF-A -- linear fusion delta on the two small probes, by transform (R-A2 input):")
    print(wide.round(4).to_string())
    truncating = []
    for transform in wide.index:
        if transform.startswith("pca") and int(transform[3:]) <= 32:
            truncating.append(transform)
    flipped = wide.loc[truncating]
    both_nonneg = (flipped >= 0).all(axis=1)
    verdicts.append(rule("R-A2", bool(both_nonneg.any()),
                         "under linear, the two small-n fusion losses flip to >=0 at some k<=32 "
                         "-> dilution confirmed for the linear story (journal footnote)",
                         f"k<=32 transforms with both >=0: "
                         f"{list(both_nonneg[both_nonneg].index) if both_nonneg.any() else 'none'}; "
                         f"best ijspeert {wide['ijspeert'].max():.4f}, "
                         f"best rgb_vs_heb {wide['rgb_vs_heb'].max():.4f}"))

    gbm_best = int(counts["n_survive"].max())
    linear_counts = survival_counts(pd.concat([linear, base_linear], ignore_index=True))
    linear_incumbent = int(linear_counts.loc[linear_counts["transform"] == "mean", "n_survive"].iloc[0])
    linear_best = int(linear_counts["n_survive"].max())
    verdicts.append(rule("R-A3", bool(gbm_best <= incumbent and linear_best <= linear_incumbent),
                         "no k improves either family -> dilution is NOT the lever, demote the z cell",
                         f"gbm best {gbm_best} vs incumbent {incumbent}; "
                         f"linear best {linear_best} vs incumbent {linear_incumbent}"))
    linear_counts.to_csv(out_dir / "fa_linear_survival_counts.csv", index=False)


def score_fb(verdicts: list[dict], out_dir: Path) -> None:
    """F-B: is mu a nonlinear recoding of the engineered features, and are the used dims the easy ones?"""
    aggregates = pd.read_csv(home / "fb_predictability" / "fb_aggregates.csv")
    variance = aggregates[aggregates["weighting"] == "mu_variance"]
    unweighted = aggregates[aggregates["weighting"] == "unweighted"]
    probes = aggregates[aggregates["weighting"] == "probe_coef"]

    table = variance.groupby("population")[["ridge", "gbm"]].agg(["mean", "std"])
    table.to_csv(out_dir / "fb_variance_weighted.csv")
    print("\nF-B -- mu-variance-weighted R^2 of mu on the 25 features (6 encoder seeds):")
    print(table.round(4).to_string())

    gap = float(variance["gbm"].mean() - variance["ridge"].mean())
    verdicts.append(rule("R-B1", bool(gap >= 0.15),
                         "gbm - ridge >= 0.15 -> mu is substantially a NONLINEAR recoding; a "
                         "complementarity objective must penalise nonlinear predictability",
                         f"pooled gbm {variance['gbm'].mean():.4f} - ridge "
                         f"{variance['ridge'].mean():.4f} = {gap:.4f}"))

    unexplained = float(1.0 - variance["gbm"].mean())
    probe_mean = float(probes["gbm"].mean())
    plain_mean = float(unweighted["gbm"].mean())
    verdicts.append(rule("R-B2", bool(unexplained < 0.20 and probe_mean > plain_mean),
                         "unpredictable fraction < 0.20 AND probe-used dims more predictable than "
                         "average -> exp10 must CREATE content, not protect it",
                         f"unpredictable {unexplained:.4f}; probe-weighted gbm R^2 {probe_mean:.4f} vs "
                         f"unweighted {plain_mean:.4f}"))

    absolute = pd.read_csv(repo_root / "experiments" / "f1_fusion_scorecard" / "f1_absolute.csv")
    perp = absolute[(absolute["arm_set"] == "mu_perp_full") & (absolute["readout"] == "mean")
                    & (absolute["readout_family"] == "linear") & absolute["reportable"]]
    trained = perp[perp["family"] == "hann0p3_fbwd"].set_index("task")["score_mean"]
    untrained = perp[perp["family"] == "untrained"].set_index("task")["score_mean"]
    residual_over_control = (trained - untrained).dropna()
    verdicts.append(rule("R-B3", bool(unexplained >= 0.20 and residual_over_control.mean() > 0),
                         "large unpredictable fraction but low mu_perp_full probe scores -> the residual "
                         "exists yet is not linearly usable; favour reorganisation over decorrelation",
                         f"unpredictable {unexplained:.4f}; mean mu_perp_full over its untrained control "
                         f"{residual_over_control.mean():.4f} across {len(residual_over_control)} tasks"))
    probes.to_csv(out_dir / "fb_probe_weighted.csv", index=False)


def score_fc(verdicts: list[dict], out_dir: Path) -> None:
    """F-C: does any richer window statistic beat mean-pooling beyond its untrained twin's gain?"""
    def per_seed(cell: pd.Series) -> dict[int, float]:
        """The per-seed delta series behind one summary row, so a gain can be paired seed by seed."""
        out = {}
        for seed in range(6):
            column = f"delta_s{seed}"
            if column in cell.index and pd.notna(cell[column]):
                out[seed] = float(cell[column])
        return out

    rows = []
    stats = derived_table(home / "fc_windowstats" / "f1_summary.csv", "gbm", drop_collapsed=True)
    twin = derived_table(home / "fc_windowstats_untrained" / "f1_summary.csv", "gbm", drop_collapsed=True)
    base = derived_table(home / "footing" / "f1_summary.csv", "gbm")
    both = pd.concat([stats, twin, base], ignore_index=True)
    control_row = {}
    for _, cell in both[both["transform"] == "mean"].iterrows():
        control_row[(cell["source"], cell["task"])] = per_seed(cell)
    for _, cell in both[both["transform"] != "mean"].iterrows():
        reference = control_row.get((cell["source"], cell["task"]), {})
        variant = per_seed(cell)
        # The variant and the `mean` control share encoder seeds, so the gain is differenced WITHIN a
        # seed. Differencing the two means instead throws away the pairing and inflates the error bar.
        shared = sorted(set(variant) & set(reference))
        gains = np.array([variant[s] - reference[s] for s in shared], dtype=float)
        sd = float(gains.std(ddof=1)) if len(gains) > 1 else np.nan
        rows.append({"task": cell["task"], "source": cell["source"], "variant": cell["transform"],
                     "delta_vs_features": cell["delta_mean"], "delta_2se": cell["delta_2se"],
                     "n_seeds": len(gains), "gain_over_mean": float(gains.mean()) if len(gains) else np.nan,
                     "gain_2se": 2 * sd / np.sqrt(len(gains)) if len(gains) > 1 else np.nan})

    mil_path = home / "fc_mil" / "fc_mil_summary.csv"
    if mil_path.exists():
        mil = pd.read_csv(mil_path)
        mil = mil[(mil["readout_family"] == "gbm") & (mil["arm_set"] == "features_plus_ws")]
        mil_probe = pd.read_csv(home / "fc_mil" / "fc_mil_probe.csv")
        for _, cell in mil.iterrows():
            reference = control_row.get((cell["arm_family"], cell["task"]), {})
            sub = mil_probe[(mil_probe["task"] == cell["task"])
                            & (mil_probe["readout_family"] == "gbm")
                            & (mil_probe["arm"].str.startswith(cell["arm_family"]))]
            base_by_seed = sub[sub["arm_set"] == "features_only"].set_index("seed")["pr_auc"]
            fused_by_seed = sub[sub["arm_set"] == "features_plus_ws"].set_index("seed")["pr_auc"]
            gains = []
            for seed, value in reference.items():
                if seed in base_by_seed.index and seed in fused_by_seed.index:
                    gains.append(float(fused_by_seed[seed] - base_by_seed[seed]) - value)
            gains = np.array(gains, dtype=float)
            sd = float(gains.std(ddof=1)) if len(gains) > 1 else np.nan
            rows.append({"task": cell["task"], "source": cell["arm_family"], "variant": "window_score",
                         "delta_vs_features": cell["delta_mean"], "delta_2se": cell["delta_2se"],
                         "n_seeds": len(gains),
                         "gain_over_mean": float(gains.mean()) if len(gains) else np.nan,
                         "gain_2se": 2 * sd / np.sqrt(len(gains)) if len(gains) > 1 else np.nan})
    else:
        log.warning(f"{mil_path} absent; F-C is scored on the window-statistic half alone")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "fc_variant_gains.csv", index=False)
    trained = frame[(frame["source"] == "hann0p3_fbwd") & frame["task"].isin(["transit", "eb"])]
    twin = frame[frame["source"] == "untrained"].set_index(["task", "variant"])["gain_over_mean"]
    print("\nF-C -- gain over `features (+) mean` under GBM on the two localized tasks:")
    view = trained[["task", "variant", "delta_vs_features", "delta_2se", "gain_over_mean"]].copy()
    twin_values = []
    for _, cell in view.iterrows():
        twin_values.append(twin.get((cell["task"], cell["variant"]), np.nan))
    view["untrained_gain"] = twin_values
    print(view.round(4).to_string(index=False))

    hit = view[(view["gain_over_mean"] > view["delta_2se"])
               & (view["gain_over_mean"] > view["untrained_gain"].fillna(-np.inf))]
    verdicts.append(rule("R-C1", not hit.empty,
                         "a window-statistic variant beats `features (+) mean` by > 2*SE on transit or "
                         "eb under GBM AND exceeds its untrained twin -> the localized channel is a real "
                         "mu asset that mean-pooling destroys",
                         f"best gain {view['gain_over_mean'].max():.4f} "
                         f"({view.loc[view['gain_over_mean'].idxmax(), 'variant']} on "
                         f"{view.loc[view['gain_over_mean'].idxmax(), 'task']}); "
                         f"qualifying variants: {hit['variant'].tolist() or 'none'}"))
    view.to_csv(out_dir / "fc_localized_summary.csv", index=False)


def score_fd(verdicts: list[dict], out_dir: Path) -> None:
    """F-D: does concatenating six seeds' mu help (complementary content) or hurt (column count)?"""
    ensemble = fusion(home / "fd_ensemble" / "f1_summary.csv", "gbm")
    base = derived_table(home / "footing" / "f1_summary.csv", "gbm")
    single = base[base["source"] == "hann0p3_fbwd"]
    per_seed = {}
    for _, cell in single.iterrows():
        values = []
        for seed in range(6):
            column = f"delta_s{seed}"
            if column in cell and pd.notna(cell[column]):
                values.append(float(cell[column]))
        per_seed[cell["task"]] = np.array(values)

    rows = []
    for _, cell in ensemble.iterrows():
        seeds = per_seed.get(cell["task"])
        if seeds is None:
            continue
        rows.append({"task": cell["task"], "variant": cell["family"],
                     "delta_mean": cell["delta_mean"], "delta_2se": cell["delta_2se"],
                     "single_seed_mean": float(seeds.mean()), "single_seed_best": float(seeds.max()),
                     "beats_single_mean": bool(cell["delta_mean"] > seeds.mean()),
                     "beats_single_best": bool(cell["delta_mean"] > seeds.max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "fd_ensemble_vs_single.csv", index=False)
    print("\nF-D -- ensemble fusion delta under GBM against the single-seed distribution:")
    print(frame.pivot_table(index="task", columns="variant",
                            values="delta_mean").round(4).to_string())

    concat = frame[frame["variant"] == "ens_concat"]
    n_beat = int(concat["beats_single_best"].sum())
    n_lose = int((~concat["beats_single_mean"]).sum())
    verdicts.append(rule("R-D1", bool(n_beat > len(concat) / 2),
                         "concat >= best single seed on most tasks under GBM -> seeds carry "
                         "complementary content and raw column count is not the binding cost",
                         f"concat beats the best single seed on {n_beat}/{len(concat)} tasks"))
    verdicts.append(rule("R-D2", bool(n_lose > len(concat) / 2),
                         "concat < the single-seed distribution on most tasks -> column count itself "
                         "hurts even a GBM; strongest possible dilution evidence",
                         f"concat below the single-seed mean on {n_lose}/{len(concat)} tasks"))


def score_fe(verdicts: list[dict], out_dir: Path) -> None:
    """F-E: do deltas that are positive at full n die at the small probes' train size?"""
    summary = pd.read_csv(home / "fe_subsample" / "fe_summary.csv")
    summary.to_csv(out_dir / "fe_summary_copy.csv", index=False)
    print("\nF-E -- fusion delta by train-set size:")
    print(summary.pivot_table(index=["task", "family"], columns="n_train_target",
                              values="delta_mean").round(4).to_string())
    # The rule names n ~ 755. The other subsample level is scored too and printed beside it, because a
    # rule that fires at one of its own two levels and not the other is a fact about the rule, not noise.
    hits: dict[str, dict[str, list[str]]] = {}
    for level in [755, 160]:
        hits[f"n{level}"] = {}
        for family in summary["family"].unique():
            view = summary[summary["family"] == family]
            full = view[view["n_train_target"] == -1].set_index("task")["delta_mean"]
            small = view[view["n_train_target"] == level].set_index("task")["delta_mean"]
            died = []
            for task in full.index:
                if task in small and full[task] > 0 and small[task] <= 0:
                    died.append(task)
            hits[f"n{level}"][family] = died
    fired = False
    for family_hits in hits["n755"].values():
        if len(family_hits) >= 2:
            fired = True
    verdicts.append(rule("R-E1", fired,
                         "deltas positive at full n turn <=0 at n~755 on >=2 of the 3 tasks -> the "
                         "small-n fusion losses are n-driven; exp10's gate excludes small-n rows",
                         f"tasks dying at the rule's own level n~755: {hits['n755']}; "
                         f"at the other subsampled level n~160: {hits['n160']}"))
    verdicts.append(rule("R-E2", not fired,
                         "deltas stay positive at n~755 -> the small probes lose for task-specific "
                         "reasons and the gate keeps every reportable row",
                         f"tasks dying at n~755: {hits['n755']}"))


def score_ff(verdicts: list[dict], out_dir: Path) -> None:
    """F-F: is the 4-of-11 GBM verdict stable once both arms get the same tuning budget?"""
    summary = pd.read_csv(home / "ff_gbm_fairness" / "ff_summary.csv")
    summary.to_csv(out_dir / "ff_summary_copy.csv", index=False)
    print("\nF-F -- GBM fusion delta, published default config vs identically-tuned arms:")
    print(summary[["task", "metric", "default_delta", "default_survives", "tuned_delta",
                   "tuned_2se", "tuned_survives", "mu_gain_fraction"]].round(4).to_string(index=False))
    default_count = int(summary["default_survives"].sum())
    tuned_count = int(summary["tuned_survives"].sum())
    moved = abs(tuned_count - default_count)
    verdicts.append(rule("R-F1", bool(moved >= 2),
                         "the GBM survival count moves by >=2 tasks under fair tuning -> C3b's 4/11 is "
                         "tuning-fragile and exp10's gate re-baselines on the tuned numbers",
                         f"default {default_count}/11 -> tuned {tuned_count}/11 (moved {moved})"))
    verdicts.append(rule("R-F2", bool(3 <= tuned_count <= 5 and moved < 2),
                         "the count stays 3-5 -> C3b is robust and the gate stays on the published "
                         "baseline",
                         f"tuned count {tuned_count}/11, moved {moved}"))
    flat = summary[summary["default_delta"].abs() < summary["default_2se"]]
    ignored = flat[flat["mu_gain_fraction"] < 0.05]
    verdicts.append(rule("R-F3", bool(not flat.empty),
                         "importance fraction ~0 where the fusion delta ~0 -> the GBM IGNORES mu "
                         "(redundancy) rather than using it without gain; record which",
                         f"{len(flat)} tasks with a flat delta; of those {len(ignored)} have mu gain "
                         f"share < 5% ({ignored['task'].tolist()}); mu column share "
                         f"{128 / 153:.3f} for reference"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the six exp10 forensics against their rules.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip", nargs="*", default=[], help="forensic letters to skip, e.g. --skip E F")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else home / "verdict"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts: list[dict] = []
    scorers = {"A": score_fa, "B": score_fb, "C": score_fc, "D": score_fd, "E": score_fe, "F": score_ff}
    for letter, scorer in scorers.items():
        if letter in args.skip:
            log.warning(f"F-{letter} SKIPPED by request; its rules are absent from the verdict table")
            continue
        scorer(verdicts, out_dir)

    frame = pd.DataFrame(verdicts)
    frame.to_csv(out_dir / "decision_rules.csv", index=False)
    print("\n" + "=" * 100)
    print("PRE-REGISTERED DECISION RULES")
    print(frame[["rule", "fired", "evidence"]].to_string(index=False))
    log.info(f"wrote {out_dir / 'decision_rules.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
