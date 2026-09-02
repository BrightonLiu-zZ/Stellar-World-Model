"""L1 -- re-measure the `flare` probe on a label whose two classes are both defensible.

Roadmap 2026-08-25 row **L1**, decision **D21**, extended by the 2026-08-26 grilling. D21 as written
fixes only the negative class; the measurements below say both classes are broken, in opposite ways,
and that the two failures interact. So this scores a 2x2 rather than a single corrected number.

THE POSITIVE AXIS -- event coverage. `flare_ever = 1` means flatwrm2 saw a flare somewhere in sectors
1-69. Folding its Table-3 peak times onto the cadences we actually packed says only 19.4% of our 2,022
flare positives have a flare inside the segment the eval encodes (32.5% over all their packed windows).
So 81% of the positive class shows the model no flare.

THE NEGATIVE AXIS -- who was searched. Seli et al. (2025) Eq. (1) only searched light curves with
sigma_ratio > 0.4, so `flare_ever = 0` mixes searched-and-clean stars with never-searched ones.
`swm.eval.flare_search_universe` reconstructs that cut from our own packed windows.

WHY THEY INTERACT, WHICH IS THE POINT OF THE 2x2. The cut runs the same way as the label: every flare
positive necessarily cleared 0.4, while 10,000 of the 20,838 negatives are the `quiet` pool, defined
as matched in NO variability catalog. A probe can therefore score on `flare` by detecting variability
that flatwrm2's own pre-cut wrote into the label. Restricting negatives to searched-and-clean forces
them through the SAME gate the positives passed, which is the matched control for exactly that.

PRE-REGISTERED BEFORE ANY CELL WAS SCORED (user-approved 2026-08-26):
  headline cell = `matched`; `as_published` is reported beside it, never instead of it.
  fusion delta > 2*SE  --> flare stays a scoped win, negative-set provenance named.
  fusion delta within 2*SE --> flare is a null, printable only with the D21 route-(e) visual pass.
  fusion delta < -2*SE --> reported as a negative result.
  reconstruction validation fails --> cells 2 and 4 are VOID; fall back to route (e) alone.
  Across cells the prevalence changes by design, so paired deltas keep their SIGN, not their
  magnitude (R8-F1). Absolute PR-AUC is not comparable between cells and is reported per cell only.

AMENDMENT, made before any cell was scored: restricting positives to on-screen flares leaves ~59 test
positives, which is SMALL by the project's standing rule (n_test < 400 or n_pos < 100). A 6-seed SE
measures how much the ENCODER varies and is silent about how much a 59-positive split varies, so under
seed spread alone an underpowered cell would read as the pre-registered "null" -- the one verdict that
must not be reachable by accident. The fusion rows therefore also carry R1's paired star-level
bootstrap, and `claimable` requires BOTH gates. This makes the win branch harder to reach, never
easier, which is the only direction a post-hoc estimator change is allowed to run.

VALIDATION GATE, also pre-registered: a light curve flatwrm2 demonstrably searched must clear 0.4
under a faithful reconstruction. Threshold >= 0.90. `sigma_ratio_sector` (the longer span, closer to
Seli's per-light-curve statistic) is preferred; `sigma_ratio_segment` is the fallback; if neither
clears the gate the route is void. The test set is fixed at the (star, sector) level -- see
`validate` for why a star-level version of this gate is arithmetically incapable of passing.

This script never writes a mu cache, only reads them, so the exp09 cache-key trap does not apply.

Run (repo root, swm env, PYTHONPATH=src; CPU-only):
    PYTHONUNBUFFERED=1 python experiments/analyze_l1_flare_negatives.py
    python experiments/analyze_l1_flare_negatives.py --arms hann0p3_fbwd_s0 untrained
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse F1/R1's estimator rather than fork it

from analyze_exp08_menu_channel import (align_features, arm_parts, load_mu_cache, metric_value)  # noqa: E402
from analyze_f1_fusion_scorecard import arm_tables, paired_delta  # noqa: E402
from swm.eval.new_task_ceiling import cached_pool_features  # noqa: E402
from swm.eval.new_task_scorecard import label_frame, score_detection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("l1_flare")

splits = ["train", "test"]
sigma_ratio_threshold = 0.4 # Seli Eq. (1)
validation_floor = 0.90 # pre-registered: share of known positives a faithful reconstruction must recover
default_arms = ([f"hann0p3_fbwd_s{s}" for s in range(6)]
                + [f"hann0p3_off_s{s}" for s in range(6)] + ["untrained"])
# (cell, positives restricted to on-screen flares, negatives restricted to searched-and-clean)
cells = [("as_published", False, False),
         ("neg_searched", False, True),
         ("pos_covered", True, False),
         ("matched", True, True)]


def searched_stars(universe: pd.DataFrame, column: str) -> set[int]:
    """
    Stars flatwrm2 would have searched, as a star-level rollup of the per-(tic, sector) reconstruction.
    A detection in ANY sector sets `flare_ever = 1`, so the negative must mirror that: a star counts as
    searched if ANY of its sector-69-or-earlier light curves cleared the cut.
    """
    passing = universe[universe[column] > sigma_ratio_threshold]
    return set(passing["tic_id"].astype(int))


def covered_stars(window_labels: pd.DataFrame) -> set[int]:
    """
    Flare positives with at least one catalogued flare inside a window the eval actually encodes.
    Built from the same fold `swm.eval.flare_window_labels` used for the Level-B probe, so the
    positive class here is the Level-B population lifted to star level rather than a new derivation.
    """
    per_star = window_labels.groupby("tic_id")["label"].max()
    return set(per_star[per_star > 0].index.astype(int))


def validate(catalog: pd.DataFrame, universe: pd.DataFrame, pool_tics: set[int]) -> tuple[str | None, pd.DataFrame]:
    """
    Decide whether the reconstruction is trustworthy, under the gate fixed before any score was read.
    The test set is every (star, sector) pair where the catalog records a flare AND we hold that
    sector's packed data: flatwrm2 demonstrably searched that light curve, so a faithful reconstruction
    must put it above 0.4.
    This is deliberately a PER-LIGHT-CURVE test, matching the level Seli's Eq. (1) is defined at. A
    star-level version would be wrong twice over: 332 pool positives have no S<=69 data at all and
    could never clear the cut, capping the achievable rate at 0.836 and voiding the route for a reason
    unrelated to fidelity; and a star flagged from a sector we do not hold says nothing about the
    sectors we do.
    Returns the chosen column (None if the route is void) and the audit table.
    """
    truth = catalog[["TIC", "sector"]].drop_duplicates()
    truth = truth[truth["TIC"].isin(pool_tics)]
    truth = truth.rename(columns={"TIC": "tic_id"})
    joined = truth.merge(universe, on=["tic_id", "sector"], how="inner")
    assert len(joined) > 0, "no (star, sector) pair is both catalogued and packed; cannot validate"
    rows = []
    for column in ["sigma_ratio_sector", "sigma_ratio_segment"]:
        recovered = float((joined[column] > sigma_ratio_threshold).mean())
        searched = searched_stars(universe, column)
        rows.append({"column": column, "n_searched_lightcurves": len(joined),
                     "n_recovered": int((joined[column] > sigma_ratio_threshold).sum()),
                     "recovered_frac": recovered,
                     "pool_star_pass_rate": len(searched & pool_tics) / len(pool_tics),
                     "lightcurve_pass_rate": float((universe[column] > sigma_ratio_threshold).mean()),
                     "passes_gate": recovered >= validation_floor})
    audit = pd.DataFrame(rows)
    for column in ["sigma_ratio_sector", "sigma_ratio_segment"]:
        row = audit[audit["column"] == column].iloc[0]
        if row["passes_gate"]:
            return column, audit
    return None, audit


published_fan = 6 # F1's flare delta is a 6-seed mean; a shorter fan is a different quantity, not a failure


def reproduction_check(summary: pd.DataFrame) -> None:
    """
    Footing check: the `as_published` cell must reproduce F1's published flare row.
    That cell applies no mask at all, so it IS the F1 measurement reached by a different script; if it
    does not agree, the masks are not the only thing that changed and no corrected cell can be trusted.
    The engineered arm is deterministic, so `features_only` must agree essentially exactly; the fusion
    delta is a 6-seed mean and is only comparable when the full fan ran.
    """
    path = repo_root / "experiments" / "f1_fusion_scorecard" / "f1_summary.csv"
    if not path.exists():
        log.warning("f1_summary.csv absent; the cross-script reproduction control is SKIPPED")
        return
    published = pd.read_csv(path)
    published = published[(published["task"] == "flare") & (published["readout"] == "mean")
                          & (published["contrast"] == "fusion_minus_features")
                          & (published["readout_family"] == "linear")
                          & (published["family"] == "hann0p3_fbwd")]
    if published.empty:
        log.warning("no published flare row in f1_summary.csv; reproduction control SKIPPED")
        return
    mine = summary[(summary["cell"] == "as_published") & (summary["family"] == "hann0p3_fbwd")
                   & (summary["contrast"] == "fusion_minus_features")]
    assert not mine.empty, "the as_published cell produced no fusion row"
    base_diff = abs(float(mine["features_only"].iloc[0]) - float(published["features_only"].iloc[0]))
    log.info(f"REPRO features_only: {float(mine['features_only'].iloc[0])} vs published "
             f"{float(published['features_only'].iloc[0])} (|diff| {base_diff})")
    assert base_diff < 1e-6, f"engineered arm does not reproduce F1: |diff| {base_diff}"
    if int(mine["n_seeds"].iloc[0]) != published_fan:
        log.warning(f"REPRO delta SKIPPED: {int(mine['n_seeds'].iloc[0])} seeds present, the published "
                    f"{float(published['delta_mean'].iloc[0])} is a {published_fan}-seed mean. A short "
                    f"fan is a different quantity; asserting on it would fire on every smoke run.")
        return
    delta_diff = abs(float(mine["delta_mean"].iloc[0]) - float(published["delta_mean"].iloc[0]))
    log.info(f"REPRO fusion delta: {float(mine['delta_mean'].iloc[0])} vs published "
             f"{float(published['delta_mean'].iloc[0])} (|diff| {delta_diff})")
    assert delta_diff < 5e-4, f"as_published does not reproduce F1's flare delta: |diff| {delta_diff}"


def cell_population(tics: list[int], labels: pd.DataFrame, covered: set[int], searched: set[int] | None,
                    restrict_positives: bool, restrict_negatives: bool) -> np.ndarray:
    """
    Boolean keep-mask over one split's stars for one cell of the 2x2.
    An uncovered positive is DROPPED, never relabelled negative: it is a flare star whose flare we did
    not pack, which is not evidence of absence. An unsearched negative is likewise dropped rather than
    kept, because absence from a catalog that never looked carries no information.
    """
    keep = []
    for tic in tics:
        is_positive = bool(labels["flare"].get(tic, 0) == 1)
        if is_positive:
            if restrict_positives and tic not in covered:
                keep.append(False)
            else:
                keep.append(True)
        else:
            if restrict_negatives and tic not in searched:
                keep.append(False)
            else:
                keep.append(True)
    return np.array(keep, dtype=bool)


def subset_mu(mu: dict, masks: dict[str, np.ndarray]) -> dict:
    """Restrict a cached mu table to one cell's population, keeping the per-star window blocks intact."""
    out = {}
    for split in splits:
        tics, blocks = mu[split]
        mask = masks[split]
        kept_tics = []
        kept_blocks = []
        for i, tic in enumerate(tics):
            if mask[i]:
                kept_tics.append(tic)
                kept_blocks.append(blocks[i])
        out[split] = (kept_tics, kept_blocks)
    return out


def bootstrap_cell(dumps: pd.DataFrame, cell: str, family: str, seeds: list[int], n_boot: int,
                   rng: np.random.Generator) -> tuple[float, float]:
    """
    Paired star-level 95% CI on the seed-averaged fusion delta for one cell of the 2x2.
    The seed spread measures how much the ENCODER varies; it says nothing about how much a 59-positive
    test split varies, and the restricted cells are exactly where that second source dominates. The
    engineered arm is resampled on the SAME stars as every fusion seed, so the CI is the sampling noise
    of the star population rather than independent noise per readout.
    """
    base = dumps[(dumps["cell"] == cell) & (dumps["arm_set"] == "features_only")]
    if base.empty:
        return float("nan"), float("nan")
    base = base.sort_values("tic_id").reset_index(drop=True)
    fusion = {}
    for seed in seeds:
        sub = dumps[(dumps["cell"] == cell) & (dumps["arm_set"] == "features_plus_mu")
                    & (dumps["family"] == family) & (dumps["seed"] == seed)]
        if sub.empty:
            return float("nan"), float("nan")
        sub = sub.sort_values("tic_id").reset_index(drop=True)
        assert np.array_equal(sub["tic_id"].to_numpy(), base["tic_id"].to_numpy()), \
            f"{cell}: arm sets disagree on test-set membership"
        fusion[seed] = sub

    n = len(base)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        y = base["y"].to_numpy()[idx]
        m_base = metric_value("detection", y, base["score"].to_numpy()[idx])
        per_seed = []
        for seed in seeds:
            per_seed.append(metric_value("detection", y, fusion[seed]["score"].to_numpy()[idx]) - m_base)
        deltas[b] = np.nanmean(per_seed)
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def summarize(probe: pd.DataFrame, dumps: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """
    Fusion and mu-only deltas against the engineered arm, per cell and per dynamics arm, plus the
    fbwd-minus-off contrast. Mirrors F1's conventions: paired per seed, never pooled across arms (F21),
    prevalence on every row (R8-F1), and the engineered arm is seedless under a linear readout so a
    fusion delta's spread is the mu side's alone.
    Every restricted cell is SMALL by the project's standing rule (n_test < 400 or n_pos < 100), so the
    fusion rows also carry a paired star bootstrap and `claimable` requires both gates.
    """
    rng = np.random.default_rng(0) # an eval-side CI, so a fixed seed is the reproducible choice
    rows = []
    fusion_by_family: dict[str, dict[str, dict[int, float]]] = {}
    for cell, group in probe.groupby("cell", sort=False):
        base = group[group["arm_set"] == "features_only"]["pr_auc"]
        assert len(base) > 0, f"{cell}: no engineered arm scored"
        features_only = float(base.mean())
        n_test = int(group["n_test"].max())
        n_pos = int(group["n_test_pos"].max())
        small = bool(n_test < 400 or n_pos < 100)
        meta = {"cell": cell, "n_test": n_test, "n_test_pos": n_pos, "prevalence": n_pos / n_test,
                "features_only": features_only, "small_probe": small}
        for family, fam in group.groupby("family", sort=False):
            if family == "features":
                continue
            for arm_set, contrast in [("features_plus_mu", "fusion_minus_features"),
                                      ("mu", "mu_minus_features")]:
                per_seed = {}
                for _, row in fam[fam["arm_set"] == arm_set].iterrows():
                    per_seed[int(row["seed"])] = float(row["pr_auc"]) - features_only
                if not per_seed:
                    continue
                stats = paired_delta(per_seed)
                if contrast == "fusion_minus_features":
                    fusion_by_family.setdefault(family, {})[cell] = per_seed
                    lo, hi = bootstrap_cell(dumps, cell, family, sorted(per_seed), n_boot, rng)
                    stats["boot_lo"] = lo
                    stats["boot_hi"] = hi
                    stats["confirm_bootstrap"] = bool(np.isfinite(lo) and lo > 0)
                    seed_gate = bool(stats["delta_mean"] > stats["delta_2se"])
                    stats["claimable"] = bool(seed_gate and (not small or stats["confirm_bootstrap"]))
                rows.append({**meta, "contrast": contrast, "family": family, **stats})

    fbwd = None
    off = None
    for family in fusion_by_family:
        if family.endswith("_fbwd"):
            fbwd = family
        if family.endswith("_off"):
            off = family
    if fbwd is not None and off is not None:
        for cell, a in fusion_by_family[fbwd].items():
            b = fusion_by_family[off].get(cell)
            if not b:
                continue
            shared = {}
            for seed in sorted(set(a) & set(b)):
                shared[seed] = a[seed] - b[seed]
            rows.append({"cell": cell, "contrast": "fusion_fbwd_minus_off",
                         "family": f"{fbwd}_vs_off", **paired_delta(shared)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="L1: the flare probe on corrected positive and negative classes.")
    ap.add_argument("--arms", nargs="+", default=default_arms)
    ap.add_argument("--cache-dir", default="experiments/exp08_menu_channel/mu_cache")
    ap.add_argument("--universe", default="labels/qc/flare_search_universe.parquet")
    ap.add_argument("--window-labels", default="labels/qc/flare_window_labels_pool.parquet")
    ap.add_argument("--flare-catalog", default="data/Table3_flare_catalog.csv",
                    help="Seli+2025 Table 3; supplies the (star, sector) pairs the validation gate uses")
    ap.add_argument("--out-dir", default="experiments/l1_flare_negatives")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="paired star bootstrap resamples; every restricted cell is a small probe")
    args = ap.parse_args()

    cache_dir = repo_root / args.cache_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_frame()
    universe = pd.read_parquet(repo_root / args.universe)
    window_labels = pd.read_parquet(repo_root / args.window_labels)
    covered = covered_stars(window_labels)

    log.info("loading pool feature table (~15 s, no output until it finishes)")
    pool_feats = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                      repo_root / "processed" / "sequences", None)

    reference = load_mu_cache(cache_dir / f"{args.arms[0]}.npz")
    pool_tics = set()
    for split in splits:
        pool_tics.update(int(t) for t in reference[split][0])

    catalog = pd.read_csv(repo_root / args.flare_catalog)
    chosen, audit = validate(catalog, universe, pool_tics)
    audit.to_csv(out_dir / "l1_validation.csv", index=False)
    print("\nreconstruction validation (pre-registered floor "
          f"{validation_floor} of demonstrably-searched light curves must clear "
          f"sigma_ratio > {sigma_ratio_threshold}):")
    print(audit.to_string(index=False))
    if chosen is None:
        log.error("VALIDATION FAILED under both spans: route (g) is VOID. Cells `neg_searched` and "
                  "`matched` are not computed; fall back to D21 route (e) alone.")
        return 1
    log.info(f"validation passed on `{chosen}`; using it for the searched-and-clean negative set")
    searched = searched_stars(universe, chosen)

    probe_rows = []
    dump_rows = []
    jobs = []
    for arm in args.arms:
        for cell, restrict_pos, restrict_neg in cells:
            jobs.append((arm, cell, restrict_pos, restrict_neg))

    features_done = set()
    for arm, cell, restrict_pos, restrict_neg in tqdm(jobs, desc="arm x cell", total=len(jobs)):
        family, seed = arm_parts(arm)
        mu = load_mu_cache(cache_dir / f"{arm}.npz")
        masks = {}
        for split in splits:
            masks[split] = cell_population(mu[split][0], labels, covered, searched,
                                           restrict_pos, restrict_neg)
        mu_cell = subset_mu(mu, masks)
        feats_cell = align_features(pool_feats, mu_cell)
        tables = arm_tables(mu_cell, feats_cell, "mean", splits)
        for arm_set, table in tables.items():
            if arm_set == "features_only":
                if cell in features_done:
                    continue
                features_done.add(cell)
                label = ("features", "features", -1)
            else:
                label = (arm, family, seed)
            sink = []
            for row in score_detection(table, labels, "flare", poolings=("mean",), sink=sink):
                probe_rows.append({"cell": cell, "arm": label[0], "family": label[1], "seed": label[2],
                                   "arm_set": arm_set, **row})
            for dump in sink:
                for tic, y, score in zip(dump["tics"], dump["y"], dump["scores"]):
                    dump_rows.append({"cell": cell, "arm_set": arm_set, "family": label[1],
                                      "seed": label[2], "tic_id": int(tic), "y": int(y),
                                      "score": float(score)})

    probe = pd.DataFrame(probe_rows)
    probe["mu_cache"] = str(cache_dir)
    probe["sigma_ratio_column"] = chosen
    probe.to_csv(out_dir / "l1_probe.csv", index=False)
    dumps = pd.DataFrame(dump_rows)
    dumps.to_parquet(out_dir / "l1_star_scores.parquet", index=False)
    summary = summarize(probe, dumps, args.n_boot)
    summary.to_csv(out_dir / "l1_summary.csv", index=False)
    reproduction_check(summary)
    log.info(f"wrote {out_dir}/l1_{{probe,summary,validation}}.csv "
             f"({len(probe)} probe rows, {len(summary)} summary rows)")

    counts = probe[probe["arm_set"] == "features_only"][["cell", "n_test", "n_test_pos"]].drop_duplicates()
    counts["n_test_neg"] = counts["n_test"] - counts["n_test_pos"]
    counts["prevalence"] = counts["n_test_pos"] / counts["n_test"]
    print("\nwhat-if counters, test split (R8-F1: deltas keep their sign, not their magnitude):")
    print(counts.to_string(index=False))

    head = summary[(summary["contrast"] == "fusion_minus_features")
                   & (summary["family"] == "hann0p3_fbwd")]
    if not head.empty:
        print("\nfusion delta (features_plus_mu - features_only), hann0p3_fbwd, readout `mean`:")
        print(head[["cell", "n_test_pos", "prevalence", "features_only", "delta_mean", "delta_2se",
                    "boot_lo", "boot_hi", "claimable"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
