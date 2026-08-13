"""R1 (roadmap 2026-08-11) -- the fusion claim, measured on the ADR-0010 downstream menu.

The ML4PS framing rests on `features (+) mu` beating `features` alone: SSL carries something the 25
engineered T'DA-style features do not. That claim is measured only on the four v1 tasks
(`exp07_channel_probe.csv`, `exp08_signature_channel_probe.csv`) and is printed beside the seven-probe
downstream menu, which has never seen a fusion readout at all (`exp08_ladder_menu.csv` and
`exp08_prechecks/new_task_scorecard.csv` carry plain probes only). This script closes that gap.

Five readouts per probe, mirroring the v1 `channel_probes()` so the two tables are column-comparable:
    mu                 mean-pooled frozen-encoder latent, the plain probe (also the extraction assert)
    mu_resid_amp       mu with the periodicity-free amplitude basis projected out
    mu_resid_full      mu with all 25 engineered features projected out -- what mu holds that they do not
    features_only      the engineered basis alone, seed-independent and arm-independent by construction
    features_plus_mu   the fusion cell

Every readout is packed as one row per star and pushed through `new_task_scorecard`'s own scorers, so
each probe's keep-mask (numax floor, prot 5.7 d cap, rgb State filter, Villanova exclusion) is
byte-identical to the frozen-mu scorecard rather than re-derived here. Pooling is `mean` throughout,
the menu convention; no MIL anywhere (R-Q6).

Estimator (matching analyze_new_task_exp05.py):
  fusion delta   features_plus_mu - features_only, per seed, per ARM, never pooled across arms (F21).
                 features_only carries no seed index, so the spread is the mu side's alone -- stated
                 rather than dressed up as a two-sided SE.
  arm contrast   delta_fbwd - delta_off, paired per seed. features_only cancels exactly, so this one
                 does carry both arms' spreads (F17).
  bootstrap      paired star-level 95% CI on probes where sampling noise dominates seed noise
                 (n_test < 400 or n_pos < 100); the SAME resampled star index scores every arm.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, needs the mu caches built first):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp08_menu_channel.py
    python experiments/analyze_exp08_menu_channel.py --arms hann0p3_fbwd_s0 hann0p3_off_s0 untrained \
        --n-boot 200 --out-dir experiments/exp08_menu_channel_smoke
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from tqdm.auto import tqdm

from swm.eval.features import FEATURE_NAMES
from swm.eval.new_task_ceiling import cached_pool_features, cached_subset_features
from swm.eval.new_task_scorecard import (DETECTION, REGRESSION, label_frame, score_contrastive,
                                         score_detection, score_ijspeert_from_mu,
                                         score_regression_task, score_rotation_period_from_mu)
from swm.eval.readout_sweep import pool_stars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp08_menu_channel")

repo_root = Path(__file__).resolve().parents[1]
SPLITS = ["train", "test"]  # the scorers read no other split; skipping val halves the residualisation
AMP_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"]  # exp06 amplitude basis, periodogram-free
READOUTS = ["mu", "mu_resid_amp", "mu_resid_full", "features_only", "features_plus_mu"]
# ADR-0010: one probe per physical quantity. The other four tasks the scorers emit are kept in the CSV
# (numax_hatt is the documented robustness footnote) but never enter the verdict.
REPORTABLE = {"numax_hon", "rotation_period", "osc_giant", "solar_like_osc", "rgb_vs_heb", "ijspeert",
              "flare"}
HEADLINE = {"detection": "pr_auc", "contrastive": "roc_auc", "regression": "r2"}
DEFAULT_ARMS = ([f"hann0p3_fbwd_s{s}" for s in range(6)]
                + [f"hann0p3_off_s{s}" for s in range(6)] + ["untrained"])
BOOT_N_MAX = 400  # below this test-set size the seed SE alone is not an honest error bar
BOOT_POS_MAX = 100  # ...and a wide test set with very few positives needs it just as much
_ARM_RE = re.compile(r"^(?P<family>.+?)_s(?P<seed>\d+)$")


def arm_parts(arm: str) -> tuple[str, int]:
    """Split an arm name into (family, seed); the single-init untrained reference is family `untrained`, seed 0."""
    if arm.startswith("untrained"):
        match = _ARM_RE.match(arm)
        if match is None:
            return "untrained", 0
        return "untrained", int(match.group("seed"))
    match = _ARM_RE.match(arm)
    assert match is not None, f"cannot read a family/seed from arm name {arm!r}"
    return match.group("family"), int(match.group("seed"))


def load_mu_cache(cache_path: Path) -> dict:
    """Read one arm's mu cache into {split: (tics, per-star window blocks)}, train and test only.

    `new_task_scorecard.load_cache` insists on a `val` split, which the pool caches carry and the
    v1-subset caches do not; the scorers read neither, so this reads exactly the two splits they use.
    """
    payload = np.load(cache_path, allow_pickle=False)
    result = {}
    for split in SPLITS:
        flat = payload[f"{split}_mu"]
        tics = payload[f"{split}_tics"].tolist()
        blocks = []
        start = 0
        for count in payload[f"{split}_counts"]:
            blocks.append(flat[start:start + int(count)])
            start += int(count)
        result[split] = (tics, blocks)
    return result


def mean_pool(mu: dict) -> dict:
    """Reduce each star's window-mu block to its mean, kept in the (tics, blocks) layout the scorers want."""
    result = {}
    for split in SPLITS:
        tics, blocks = mu[split]
        pooled = pool_stars(blocks, "mean")
        rows = []
        for row in pooled:
            rows.append(row.reshape(1, -1))
        result[split] = (tics, rows)
    return result


def stacked(table: dict, split: str) -> np.ndarray:
    """The (n_star, n_dim) matrix behind one split's 1-row blocks."""
    return np.concatenate(table[split][1], axis=0)


def as_blocks(tics: list[int], values: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Wrap a (n_star, n_dim) matrix back into the per-star 1-row blocks the scorers pool over."""
    blocks = []
    for row in values.astype(np.float32):
        blocks.append(row.reshape(1, -1))
    return tics, blocks


def align_features(feats: dict, mu: dict) -> dict:
    """Reorder the feature table onto each split's mu star list, failing loud if the populations differ.

    Both tables come from the same first-segment replay with the same absmax guard, so a mismatch means
    one of the two caches is stale rather than a legitimate population difference.
    """
    result = {}
    for split in SPLITS:
        mu_tics = mu[split][0]
        feat_tics, feat_blocks = feats[split]
        lookup = {}
        for i, tic in enumerate(feat_tics):
            lookup[tic] = i
        missing = []
        rows = []
        for tic in mu_tics:
            if tic not in lookup:
                missing.append(tic)
                continue
            rows.append(feat_blocks[lookup[tic]])
        assert not missing, f"{split}: {len(missing)} mu stars absent from the feature table (first {missing[:3]})"
        result[split] = (list(mu_tics), rows)
    return result


def residualize(mu: dict, feats: dict, basis: list[str]) -> dict:
    """Project a feature sub-basis out of mu, fitting the linear map on the TRAIN split only."""
    keep = []
    for name in basis:
        keep.append(FEATURE_NAMES.index(name))
    fitter = LinearRegression().fit(stacked(feats, "train")[:, keep], stacked(mu, "train"))
    result = {}
    for split in SPLITS:
        values = stacked(mu, split) - fitter.predict(stacked(feats, split)[:, keep])
        result[split] = as_blocks(mu[split][0], values)
    return result


def concat_tables(left: dict, right: dict) -> dict:
    """Column-concatenate two aligned 1-row-per-star tables (the fusion cell: features then mu)."""
    result = {}
    for split in SPLITS:
        values = np.concatenate([stacked(left, split), stacked(right, split)], axis=1)
        result[split] = as_blocks(left[split][0], values)
    return result


def score_menu(table: dict, subset_table: dict, labels: pd.DataFrame, sink: list) -> list[dict]:
    """Every menu probe for one readout: pool-borne tasks off `table`, subset-borne tasks off `subset_table`."""
    rows = []
    for name, column in DETECTION:
        for cell in score_detection(table, labels, column, poolings=("mean",), sink=sink):
            rows.append({"task": name, "shape": "detection", **cell})
    for cell in score_contrastive(table, labels, sink=sink):
        rows.append({"task": "rgb_vs_heb", "shape": "contrastive", **cell})
    for name, column, log_target in REGRESSION:
        for cell in score_regression_task(table, labels, column, log_target, sink=sink):
            rows.append({"task": name, "shape": "regression", **cell})
    for cell in score_rotation_period_from_mu(subset_table, sink=sink):
        rows.append({"task": "rotation_period", "shape": "regression", **cell})
    for cell in score_ijspeert_from_mu(subset_table, sink=sink):
        rows.append(cell)
    return rows


def readout_tables(mu: dict, feats: dict) -> dict[str, dict]:
    """The five readouts for one arm on one pool, all sharing that pool's star order."""
    tables = {}
    tables["mu"] = mu
    tables["mu_resid_amp"] = residualize(mu, feats, AMP_COLS)
    tables["mu_resid_full"] = residualize(mu, feats, list(FEATURE_NAMES))
    tables["features_only"] = feats
    tables["features_plus_mu"] = concat_tables(feats, mu)
    return tables


def metric_value(shape: str, y: np.ndarray, score: np.ndarray) -> float:
    """One probe's headline metric, recomputed from raw predictions (used inside the bootstrap)."""
    if shape == "regression":
        return float(r2_score(y, score))
    if y.min() == y.max():
        return np.nan  # a resample with one class carries no information; dropped by nanpercentile
    if shape == "contrastive":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def bootstrap_fusion(dumps: pd.DataFrame, task: str, shape: str, family: str, seeds: list[int],
                     n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Paired star-level 95% CI on the seed-averaged fusion delta for one probe and one arm family.

    The engineered-feature readout is scored on the SAME resampled stars as every fusion seed, so the CI
    reflects sampling noise in the star population rather than independent noise per readout.
    """
    base = dumps[(dumps["task"] == task) & (dumps["readout"] == "features_only")]
    if base.empty:
        return np.nan, np.nan
    base = base.sort_values("tic_id").reset_index(drop=True)
    fusion = {}
    for seed in seeds:
        sub = dumps[(dumps["task"] == task) & (dumps["readout"] == "features_plus_mu")
                    & (dumps["family"] == family) & (dumps["seed"] == seed)]
        if sub.empty:
            return np.nan, np.nan
        sub = sub.sort_values("tic_id").reset_index(drop=True)
        assert np.array_equal(sub["tic_id"].to_numpy(), base["tic_id"].to_numpy()), \
            f"{task}: readouts disagree on test-set membership"
        fusion[seed] = sub

    n = len(base)
    deltas = np.empty(n_boot)
    for b in tqdm(range(n_boot), desc=f"boot[{family}/{task}]", total=n_boot, leave=False):
        idx = rng.integers(0, n, n)
        y = base["y"].to_numpy()[idx]
        m_base = metric_value(shape, y, base["score"].to_numpy()[idx])
        per_seed = []
        for seed in seeds:
            per_seed.append(metric_value(shape, y, fusion[seed]["score"].to_numpy()[idx]) - m_base)
        deltas[b] = np.nanmean(per_seed)
    return float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def paired_delta(values: dict[int, float]) -> dict:
    """Mean, SD and 2*SE of a per-seed delta series, with the per-seed values kept for the CSV."""
    deltas = np.array(list(values.values()), dtype=float)
    sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else np.nan
    two_se = 2 * sd / np.sqrt(len(deltas)) if len(deltas) > 1 else np.nan
    row = {"n_seeds": len(deltas), "delta_mean": float(deltas.mean()), "delta_sd": sd,
           "delta_2se": two_se,
           "confirm_seed_gate": bool(len(deltas) > 1 and deltas.mean() > two_se)}
    for seed, value in values.items():
        row[f"delta_s{seed}"] = value
    return row


def summarize(probe: pd.DataFrame, dumps: pd.DataFrame | None, n_boot: int) -> pd.DataFrame:
    """Fusion deltas per arm family, the mu-vs-features standing comparison, and the arm contrast."""
    rng = np.random.default_rng(0)  # fixed: an inference statistic, not a display sample
    rows = []
    fusion_by_family: dict[str, dict[tuple[str, str], dict[int, float]]] = {}
    for (task, shape), group in probe.groupby(["task", "shape"], sort=False):
        metric = HEADLINE[shape]
        n_test = int(group["n_test"].max())
        if group["n_test_pos"].notna().any():
            n_pos = int(group["n_test_pos"].max())
        else:
            n_pos = n_test
        small = (n_test < BOOT_N_MAX) or (n_pos < BOOT_POS_MAX)
        base = group[group["readout"] == "features_only"][metric]
        assert len(base) > 0, f"{task}: no features_only row"
        features_only = float(base.iloc[0])

        for family, family_group in group.groupby("family", sort=False):
            if family == "features":
                continue
            fusion, mu_only = {}, {}
            for _, cell in family_group.iterrows():
                if cell["readout"] == "features_plus_mu":
                    fusion[int(cell["seed"])] = float(cell[metric]) - features_only
                elif cell["readout"] == "mu":
                    mu_only[int(cell["seed"])] = float(cell[metric]) - features_only
            if not fusion:
                continue
            fusion_by_family.setdefault(family, {})[(task, shape)] = fusion

            row = {"task": task, "shape": shape, "metric": metric, "contrast": "fusion_minus_features",
                   "family": family, "n_test": n_test, "n_test_pos": n_pos,
                   "features_only": features_only, "reportable": task in REPORTABLE, **paired_delta(fusion)}
            lo, hi = (np.nan, np.nan)
            if dumps is not None and small:
                lo, hi = bootstrap_fusion(dumps, task, shape, family, sorted(fusion), n_boot, rng)
            row["boot_lo"], row["boot_hi"] = lo, hi
            row["confirm_bootstrap"] = bool(np.isfinite(lo) and lo > 0)
            row["claimable"] = bool(row["confirm_seed_gate"] and (not small or row["confirm_bootstrap"]))
            rows.append(row)

            rows.append({"task": task, "shape": shape, "metric": metric, "contrast": "mu_minus_features",
                         "family": family, "n_test": n_test, "n_test_pos": n_pos,
                         "features_only": features_only, "reportable": task in REPORTABLE,
                         "boot_lo": np.nan, "boot_hi": np.nan, "confirm_bootstrap": False,
                         "claimable": False, **paired_delta(mu_only)})

    # The arm contrast: features_only cancels exactly, so this is the one statistic carrying both arms'
    # seed spreads (F17). Only defined where the two arms share seed indices.
    if "hann0p3_fbwd" in fusion_by_family and "hann0p3_off" in fusion_by_family:
        for key, fbwd in fusion_by_family["hann0p3_fbwd"].items():
            off = fusion_by_family["hann0p3_off"].get(key)
            if not off:
                continue
            task, shape = key
            shared = {}
            for seed in sorted(set(fbwd) & set(off)):
                shared[seed] = fbwd[seed] - off[seed]
            rows.append({"task": task, "shape": shape, "metric": HEADLINE[shape],
                         "contrast": "fusion_fbwd_minus_off", "family": "hann0p3_fbwd_vs_off",
                         "reportable": task in REPORTABLE, "boot_lo": np.nan, "boot_hi": np.nan,
                         "confirm_bootstrap": False, "claimable": False, **paired_delta(shared)})
    return pd.DataFrame(rows)


def reproduction_check(probe: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Assert the freshly extracted mu and the cached features reproduce the artifacts already on disk.

    The mu rows pin the extraction to the right checkpoints (the documented --ckpt-dir trap: a wrong
    cell's weights under a correct arm label is invisible downstream). The features_only rows pin the
    cached feature table to the A1 ceiling it was originally computed for.
    """
    rows = []
    scorecard_path = repo_root / "experiments" / "exp08_prechecks" / "new_task_scorecard.csv"
    if scorecard_path.exists():
        reference = pd.read_csv(scorecard_path)
        reference = reference[reference["pooling"] == "mean"]
        for _, cell in probe[probe["readout"] == "mu"].iterrows():
            match = reference[(reference["arm"] == cell["arm"]) & (reference["task"] == cell["task"])]
            if match.empty:
                continue
            metric = HEADLINE[cell["shape"]]
            rows.append({"check": "mu_vs_prechecks_scorecard", "arm": cell["arm"], "task": cell["task"],
                         "metric": metric, "new": float(cell[metric]),
                         "reference": float(match[metric].iloc[-1])})

    ceiling_path = repo_root / "experiments" / "exp08_prechecks" / "ceiling_A1A2.csv"
    if ceiling_path.exists():
        reference = pd.read_csv(ceiling_path)
        reference = reference[reference["arm"] == "A1_feats"]
        for _, cell in probe[probe["readout"] == "features_only"].iterrows():
            match = reference[reference["task"] == cell["task"]]
            if match.empty:
                continue
            metric = HEADLINE[cell["shape"]]
            rows.append({"check": "features_vs_A1_ceiling", "arm": "features", "task": cell["task"],
                         "metric": metric, "new": float(cell[metric]),
                         "reference": float(match[metric].iloc[-1])})

    result = pd.DataFrame(rows)
    if result.empty:
        log.warning("no reference rows found; reproduction control skipped")
        return result
    result["abs_diff"] = (result["new"] - result["reference"]).abs()
    worst = result.sort_values("abs_diff", ascending=False).head(5)
    log.info("reproduction control, five largest deviations\n" + worst.round(6).to_string(index=False))
    assert result["abs_diff"].max() < tol, \
        f"reproduction control failed: max |diff| = {result['abs_diff'].max()} > {tol}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Fusion readouts on the ADR-0010 downstream menu.")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--cache-dir", default=None, help="Default: experiments/exp08_menu_channel/mu_cache")
    ap.add_argument("--subset-cache-dir", default=None,
                    help="Default: experiments/exp08_menu_channel/subset_mu_cache")
    ap.add_argument("--out-dir", default=None, help="Default: experiments/exp08_menu_channel")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples on small probes.")
    ap.add_argument("--repro-tol", type=float, default=1e-6,
                    help="Max |diff| allowed against the prechecks scorecard and the A1 ceiling.")
    args = ap.parse_args()

    home = repo_root / "experiments" / "exp08_menu_channel"
    cache_dir = Path(args.cache_dir) if args.cache_dir else home / "mu_cache"
    subset_cache_dir = Path(args.subset_cache_dir) if args.subset_cache_dir else home / "subset_mu_cache"
    out_dir = Path(args.out_dir) if args.out_dir else home
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_frame()
    pool_feats = cached_pool_features(repo_root / "processed" / "subset" / "new_task_pool.parquet",
                                      repo_root / "processed" / "sequences", None)
    subset_feats = cached_subset_features(repo_root / "experiments" / "exp01_window256_seq16" / "packed")

    probe_rows = []
    dump_frames = []
    features_scored = False
    star_order: dict[str, list[int]] = {}
    for arm in tqdm(args.arms, desc="arms", total=len(args.arms)):
        family, seed = arm_parts(arm)
        mu = mean_pool(load_mu_cache(cache_dir / f"{arm}.npz"))
        subset_mu = mean_pool(load_mu_cache(subset_cache_dir / f"{arm}.npz"))
        # Every arm replays the same pool, so a differing star list means a stale cache; features_only is
        # scored once and would otherwise sit on a population the other arms do not share.
        for name, table in (("pool", mu), ("subset", subset_mu)):
            for split in SPLITS:
                key = f"{name}_{split}"
                if key not in star_order:
                    star_order[key] = list(table[split][0])
                assert star_order[key] == list(table[split][0]), f"{arm}: {key} star list differs"
        feats = align_features(pool_feats, mu)
        subset_aligned = align_features(subset_feats, subset_mu)
        tables = readout_tables(mu, feats)
        subset_tables = readout_tables(subset_mu, subset_aligned)
        for readout in READOUTS:
            # features_only carries no arm and no seed; scored once, under the family label `features`.
            if readout == "features_only" and features_scored:
                continue
            if readout == "features_only":
                features_scored = True
            sink = []
            rows = score_menu(tables[readout], subset_tables[readout], labels, sink)
            if readout == "features_only":
                arm_label, family_label, seed_label = "features", "features", -1
            else:
                arm_label, family_label, seed_label = arm, family, seed
            for row in rows:
                probe_rows.append({"arm": arm_label, "family": family_label, "seed": seed_label,
                                   "readout": readout, **row})
            for entry in sink:
                dump_frames.append(pd.DataFrame({"arm": arm_label, "family": family_label,
                                                 "seed": seed_label, "readout": readout,
                                                 "task": entry["task"], "pooling": entry["pooling"],
                                                 "tic_id": entry["tics"], "y": entry["y"],
                                                 "score": entry["scores"]}))
        log.info(f"scored arm {arm}")

    probe = pd.DataFrame(probe_rows)
    probe["reportable"] = probe["task"].isin(REPORTABLE)
    probe["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    probe.to_csv(out_dir / "menu_channel_probe.csv", index=False)

    dumps = pd.concat(dump_frames, ignore_index=True)
    dumps.to_parquet(out_dir / "menu_channel_star_scores.parquet", index=False)

    checks = reproduction_check(probe, args.repro_tol)
    if not checks.empty:
        checks.to_csv(out_dir / "menu_channel_repro.csv", index=False)

    summary = summarize(probe, dumps, args.n_boot)
    summary["run_id"] = probe["run_id"].iloc[0]
    summary.to_csv(out_dir / "menu_channel_summary.csv", index=False)
    log.info(f"wrote {out_dir}/menu_channel_{{probe,summary,star_scores,repro}}.*")

    headline = summary[(summary["contrast"] == "fusion_minus_features") & summary["reportable"]]
    print("\nfusion delta (features_plus_mu - features_only), ADR-0010 menu, pooling=mean:")
    print(headline.pivot_table(index="task", columns="family", values="delta_mean").round(4).to_string())
    print("\nclaimable cells (seed gate, plus bootstrap where the probe is small):")
    print(headline[["task", "family", "delta_mean", "delta_2se", "boot_lo", "boot_hi", "claimable"]]
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
