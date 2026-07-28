"""
Consolidate the MIL pooling sweep and emit the headline results table (plan 2026-07-25).

The sweep ran as four invocations writing four CSVs, because concurrent runs of swm.eval.mil_sweep
each do read-existing --> concat --> write and would silently clobber one another. This module merges
them back into the canonical long table and derives one compact key-results file so the writeup and
the notebook read the same numbers.

Blocks written to mil_pooling_results.csv (a `block` column selects between them):
  winner        per (task, bag arm): the val-declared winning operator, its test PR-AUC, the
                capacity-matched untrained arm under the SAME operator, the gap and its 2*SE over
                seeds, and the mean-pooling baseline it must be judged against.
  witness_rate  ws_lse test PR-AUC at the mean-like and max-like ends of the temperature grid, and
                the relative change between them. This is the witness-rate measurement: strongly
                positive for localized tasks, negative for global ones.
  control       bagsize_only (logistic on log window count alone) against the base rate, per scope.
                At the first-segment scope every bag holds 16-20 windows so this must sit at base
                rate; at all-segment scope it does not, which is what the K-matched arm exists for.
  kmatch        the coverage-versus-bag-size decomposition: the same operator at first-segment,
                K-matched-16 (16 windows drawn from across ALL segments), and full all-segment.
  tier3         learned pooling (ABMIL, DSMIL) against the best zero-parameter operator.

Operator selection is on ABSOLUTE val PR-AUC, never on the gap: the gap is not invariant to readout
capacity, and ranking by it would crown ABMIL, which inflates its gap by collapsing on the untrained
encoder rather than by reading the trained one better (grill decision 2026-07-26).

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.mil_report
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

repo_root = Path(__file__).resolve().parents[3]
mil_dir = repo_root / "experiments" / "mil_pooling"
part_names = ("mil_sweep.csv", "mil_sweep_all.csv", "mil_sweep_ablate.csv", "mil_sweep_kmatch.csv")
headline_cell = "exp05_comb_fbwd_c1p0"
control_cell = "exp05_comb_off"
tasks_order = ("transit", "eb", "pulsating", "rotation")


def merge_parts(out_name: str = "mil_sweep.csv") -> pd.DataFrame:
    """
    Concatenate every sweep part CSV into the canonical long table and delete the extra parts.
    Rows keep their own run_id and git_sha, so merging preserves the audit trail rather than
    flattening it; de-duplication is on the full cell key plus run_id.
    """
    frames = []
    for name in part_names:
        path = mil_dir / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
        log.info(f"{name}: {len(frame)} rows")
    assert len(frames) > 0, f"no sweep CSVs under {mil_dir}"
    merged = pd.concat(frames, ignore_index=True)
    keys = ["exp_name", "seed", "arm_kind", "bag_scope", "kmatch", "kmatch_draw",
            "family", "pooling", "param", "task", "run_id"]
    present = []
    for key in keys:
        if key in merged.columns:
            present.append(key)
    before = len(merged)
    merged = merged.drop_duplicates(subset=present)
    if len(merged) < before:
        log.info(f"dropped {before - len(merged)} duplicate rows")
    out_path = mil_dir / out_name
    merged.to_csv(out_path, index=False)
    for name in part_names:
        if name == out_name:
            continue
        path = mil_dir / name
        if path.exists():
            path.unlink() # merged in; leaving it would let the notebook read a partial view again
    log.info(f"wrote {out_path} ({len(merged)} rows)")
    return merged


def load_all(pool: str = "v1") -> pd.DataFrame:
    """
    Read one pool's long table (plus the learned-pooling table for v1) into a frame with a bag-arm column.
    The two pools never share a CSV: they are different star populations with different splits, so a
    merged table would silently average across them.
    """
    # v1 parts are merged into one canonical file; the new-task pool keeps its parts split by
    # metric and scope, so glob them rather than listing names a later run could invalidate.
    if pool == "v1":
        paths = [mil_dir / "mil_sweep.csv"]
    else:
        paths = sorted(mil_dir.glob("mil_sweep_new_task*.csv"))
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    assert len(frames) > 0, f"no sweep CSV for pool {pool!r} under {mil_dir}"
    sweep = pd.concat(frames, ignore_index=True)
    learned_path = mil_dir / "mil_learned.csv"
    if pool == "v1" and learned_path.exists():
        learned = pd.read_csv(learned_path)
        learned["kmatch"] = 0
        learned["kmatch_draw"] = 0
        sweep = pd.concat([sweep, learned], ignore_index=True)
    sweep["param"] = sweep["param"].fillna(-1.0)
    sweep["kmatch"] = sweep["kmatch"].fillna(0).astype(int)
    sweep["bag_arm"] = np.where(sweep["kmatch"] > 0, "kmatch16", sweep["bag_scope"])
    # Rows written before the regression probes existed carry only pr_auc_*; the generic score_*
    # columns are what every block reads, so both metrics can share one code path and one table.
    if "metric" not in sweep.columns:
        sweep["metric"] = "pr_auc"
    sweep["metric"] = sweep["metric"].fillna("pr_auc")
    for generic, legacy in [("score_val", "pr_auc_val"), ("score_test", "pr_auc_test")]:
        if generic not in sweep.columns:
            sweep[generic] = np.nan
        sweep[generic] = sweep[generic].fillna(sweep[legacy])
    return sweep


def winner_block(data: pd.DataFrame) -> pd.DataFrame:
    """
    Per (task, bag arm): pick the operator with the highest seed-averaged VAL PR-AUC, then report its
    test PR-AUC against the untrained arm under the same operator (untrained tuning its own
    hyperparameter on val) and against mean pooling, the protocol every earlier table used.
    """
    trained = data[(data.arm_kind == "trained") & (data.exp_name == headline_cell)]
    untrained = data[data.arm_kind == "untrained"]
    # Tier-3 learned heads are diagnostic-only (they need the unsigned ADR-0008 exception), so they
    # are never eligible to be declared the winner even when they top the val column.
    candidates = trained[(trained.pooling != "bagsize_only") & (trained.family != "learned")]
    val_mean = candidates.groupby(["bag_arm", "task", "family", "pooling", "param"])["score_val"].mean().reset_index()
    winners = val_mean.sort_values("score_val").groupby(["bag_arm", "task"]).tail(1)
    untrained_val = untrained.groupby(["bag_arm", "task", "pooling", "param"])["score_val"].mean().reset_index()
    rows = []
    for row in winners.itertuples(index=False):
        cell = trained[(trained.bag_arm == row.bag_arm) & (trained.task == row.task)
                       & (trained.pooling == row.pooling) & (trained.param == row.param)]
        per_seed = cell.groupby("seed")["score_test"].mean().to_numpy()
        ref = untrained_val[(untrained_val.bag_arm == row.bag_arm) & (untrained_val.task == row.task)
                            & (untrained_val.pooling == row.pooling)]
        if len(ref) == 0:
            continue
        ref_param = ref.sort_values("score_val").param.iloc[-1]
        ref_test = untrained[(untrained.bag_arm == row.bag_arm) & (untrained.task == row.task)
                             & (untrained.pooling == row.pooling)
                             & (untrained.param == ref_param)].score_test.mean()
        base = trained[(trained.bag_arm == row.bag_arm) & (trained.task == row.task)
                       & (trained.pooling == "mean")].score_test.mean()
        gaps = per_seed - ref_test
        two_se = 2 * gaps.std(ddof=1) / np.sqrt(len(gaps))
        rows.append({"block": "winner", "task": row.task, "bag_arm": row.bag_arm,
                     "metric": cell.metric.iloc[0],
                     "pooling": row.pooling, "param": row.param, "n_seeds": len(per_seed),
                     "score_val": row.score_val, "score_test": per_seed.mean(),
                     "test_sd": per_seed.std(ddof=1), "untrained_test": ref_test,
                     "gap": gaps.mean(), "two_se": two_se, "confirmed": bool(gaps.mean() > two_se),
                     "mean_pool_test": base, "gain_over_mean": per_seed.mean() - base,
                     "base_rate": cell.base_rate_test.mean() if "base_rate_test" in cell else np.nan,
                     "n_test_pos": cell.n_test_pos.mean() if "n_test_pos" in cell else np.nan,
                     "n_test": cell.n_test.mean()})
    return pd.DataFrame(rows)


def witness_rate_block(data: pd.DataFrame) -> pd.DataFrame:
    """
    The witness-rate measurement: how much a task gains by moving ws_lse from mean-like to max-like.
    A localized signal (few witness windows) gains a lot; a global one loses slightly.
    """
    trained = data[(data.arm_kind == "trained") & (data.exp_name == headline_cell)
                   & (data.pooling == "ws_lse")]
    rows = []
    for (bag_arm, task), group in trained.groupby(["bag_arm", "task"]):
        curve = group.groupby("param")["pr_auc_test"].mean()
        cold = curve.loc[curve.index.min()]
        hot_param = curve.index.max()
        hot = curve.loc[hot_param]
        best_param = curve.idxmax()
        rows.append({"block": "witness_rate", "task": task, "bag_arm": bag_arm,
                     "pr_auc_test_beta_min": cold, "pr_auc_test_beta_max": hot,
                     "beta_star_test": best_param, "pr_auc_test": curve.max(),
                     "rel_gain_mean_to_max": (hot - cold) / cold})
    return pd.DataFrame(rows)


def control_block(data: pd.DataFrame) -> pd.DataFrame:
    """bagsize_only against the base rate: how much a probe can score knowing only how many windows a star has."""
    control = data[(data.pooling == "bagsize_only") & (data.arm_kind == "trained")
                   & (data.exp_name == headline_cell)]
    rows = []
    for (bag_arm, task), group in control.groupby(["bag_arm", "task"]):
        metric = group.metric.iloc[0]
        row = {"block": "control", "task": task, "bag_arm": bag_arm, "pooling": "bagsize_only",
               "metric": metric, "score_test": group.score_test.mean()}
        if metric == "pr_auc": # the null for a detection probe is the base rate, for R2 it is zero
            base = group.base_rate_test.mean()
            row["base_rate"] = base
            row["ratio_to_base"] = group.score_test.mean() / base
        rows.append(row)
    return pd.DataFrame(rows)


def kmatch_block(data: pd.DataFrame,
                 poolings: tuple[str, ...] = ("mean", "moments", "mean_std", "mean_skew", "rff_meanmap"),
                 ) -> pd.DataFrame:
    """
    Coverage versus bag size. Only the K-matched-16 arm is confound-free: every star has at least 16
    windows, so at K0=16 bag size carries no information at all. The all-segment column is therefore
    an upper bound that still contains the observed-more-often selection effect.
    """
    trained = data[(data.arm_kind == "trained") & (data.exp_name == headline_cell)
                   & (data.pooling.isin(poolings))]
    pivot = trained.groupby(["task", "pooling", "bag_arm"])["pr_auc_test"].mean().unstack()
    rows = []
    for (task, pooling), row in pivot.iterrows():
        if "first" not in row.index or "kmatch16" not in row.index:
            continue
        if pd.isna(row.get("first")) or pd.isna(row.get("kmatch16")):
            continue # operator was not run at every arm, so the decomposition is undefined for it
        rows.append({"block": "kmatch", "task": task, "pooling": pooling,
                     "first_seg_k16": row.get("first"), "kmatched_k16": row.get("kmatch16"),
                     "all_seg_k62": row.get("all"),
                     "kmatch_minus_first": row.get("kmatch16") - row.get("first"),
                     "all_minus_kmatch": row.get("all") - row.get("kmatch16")})
    return pd.DataFrame(rows)


def tier3_block(data: pd.DataFrame) -> pd.DataFrame:
    """Learned pooling against the best zero-parameter operator, plus the gap each would claim."""
    first = data[data.bag_arm == "first"]
    trained = first[(first.arm_kind == "trained") & (first.exp_name == headline_cell)]
    untrained = first[first.arm_kind == "untrained"]
    rows = []
    for task in tasks_order:
        simple = trained[(trained.task == task) & (trained.family != "learned")
                         & (trained.pooling != "bagsize_only")]
        if len(simple) == 0:
            continue
        best = simple.groupby(["pooling", "param"])["score_test"].mean().idxmax()
        for pooling, param, label in [(best[0], best[1], "best_simple"),
                                      ("abmil", -1.0, "learned"), ("dsmil", -1.0, "learned")]:
            cell = trained[(trained.task == task) & (trained.pooling == pooling) & (trained.param == param)]
            if len(cell) == 0:
                continue
            ref = untrained[(untrained.task == task) & (untrained.pooling == pooling)
                            & (untrained.param == param)].score_test.mean()
            rows.append({"block": "tier3", "task": task, "tier": label, "pooling": pooling,
                         "score_test": cell.score_test.mean(), "test_sd": cell.score_test.std(ddof=1),
                         "untrained_test": ref, "gap": cell.score_test.mean() - ref})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="consolidate the MIL sweep and write the key-results table")
    parser.add_argument("--no-merge", action="store_true", help="skip merging the part CSVs")
    parser.add_argument("--pool", default="v1", choices=["v1", "new_task"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = "mil_pooling_results.csv" if args.pool == "v1" else "mil_pooling_results_new_task.csv"

    if not args.no_merge and args.pool == "v1":
        merge_parts()
    data = load_all(args.pool)
    blocks = [winner_block(data), witness_rate_block(data), control_block(data), kmatch_block(data)]
    if args.pool == "v1": # Tier-3 learned heads were only ever run on the v1 subset
        blocks.append(tier3_block(data))
    result = pd.concat(blocks, ignore_index=True)
    ordered = ["block", "task", "bag_arm", "metric", "tier", "pooling", "param", "n_seeds",
               "score_val", "score_test", "pr_auc_test", "test_sd", "untrained_test", "gap", "two_se",
               "confirmed", "mean_pool_test", "gain_over_mean", "base_rate", "ratio_to_base", "n_test", "n_test_pos",
               "pr_auc_test_beta_min", "pr_auc_test_beta_max", "beta_star_test", "rel_gain_mean_to_max",
               "first_seg_k16", "kmatched_k16", "all_seg_k62", "kmatch_minus_first", "all_minus_kmatch"]
    cols = []
    for col in ordered:
        if col in result.columns:
            cols.append(col)
    result = result[cols]
    out_path = mil_dir / args.out
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(result)} rows)")
    for block in ["winner", "witness_rate", "control", "kmatch", "tier3"]:
        part = result[result.block == block].dropna(axis=1, how="all")
        if len(part) == 0:
            continue
        print(f"\n=== {block} ===")
        print(part.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
