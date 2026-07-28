"""Unit tests for the MIL headline blocks (swm.eval.mil_report, plan 2026-07-25).

These exist because a real bug shipped twice: adding the R2 regression probes renamed `pr_auc_test`
to the metric-agnostic `score_test` in some blocks but not others, and the notebook that consumes the
table only failed when a human re-ran it. The contract worth pinning is therefore the OUTPUT SCHEMA,
not the arithmetic: every block must expose the generic score column the notebook reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swm.eval.mil_report import (control_block, kmatch_block, tier3_block, winner_block,
                                 witness_rate_block)

headline_cell = "exp05_comb_fbwd_c1p0"


def _frame() -> pd.DataFrame:
    """A miniature sweep table covering both arms, three bag arms, and both operator families."""
    rows = []
    rng = np.random.default_rng(0)
    for arm_kind, exp_name, offset in [("trained", headline_cell, 0.10), ("untrained", "untrained", 0.0)]:
        for bag_arm in ["first", "kmatch16", "all"]:
            for pooling, family, param in [("mean", "feature", -1.0), ("moments", "feature", -1.0),
                                           ("mean_std", "feature", -1.0), ("rff_meanmap", "feature", 512.0),
                                           ("ws_lse", "score", 0.5), ("ws_lse", "score", 50.0),
                                           ("bagsize_only", "feature", -1.0), ("abmil", "learned", -1.0)]:
                for seed in range(4):
                    base = 0.5 + offset + rng.normal(scale=0.01)
                    rows.append({
                        "exp_name": exp_name, "seed": seed, "arm_kind": arm_kind,
                        "bag_scope": "all" if bag_arm == "kmatch16" else bag_arm,
                        "kmatch": 16 if bag_arm == "kmatch16" else 0,
                        "family": family, "pooling": pooling, "param": param, "task": "transit",
                        "metric": "pr_auc", "pr_auc_val": base, "pr_auc_test": base,
                        "score_val": base, "score_test": base,
                        "base_rate_test": 0.06, "n_test_pos": 122, "n_test": 2021,
                    })
    frame = pd.DataFrame(rows)
    frame["bag_arm"] = np.where(frame.kmatch > 0, "kmatch16", frame.bag_scope)
    return frame


def test_long_blocks_expose_the_generic_score_column():
    """The notebook reads `score_test`; a block emitting only `pr_auc_test` breaks it silently."""
    data = _frame()
    for name, block in [("winner", winner_block(data)), ("control", control_block(data)),
                        ("tier3", tier3_block(data))]:
        assert len(block) > 0, f"{name} block produced no rows"
        assert "score_test" in block.columns, f"{name} block is missing score_test"


def test_kmatch_block_exposes_its_per_arm_columns():
    """kmatch is deliberately wide: one column per bag arm plus the two deltas, so it has no score_test."""
    block = kmatch_block(_frame())
    assert len(block) > 0
    assert {"first_seg_k16", "kmatched_k16", "all_seg_k62",
            "kmatch_minus_first", "all_minus_kmatch"} <= set(block.columns)


def test_winner_block_never_selects_a_learned_head():
    """Tier-3 heads need the unsigned ADR-0008 exception, so they are diagnostic only."""
    data = _frame()
    boosted = data.pooling == "abmil"
    data.loc[boosted, ["score_val", "pr_auc_val"]] = 0.99 # make it the best on val by a mile
    winners = winner_block(data)
    assert len(winners) > 0
    assert "abmil" not in set(winners.pooling)


def test_winner_block_reports_the_gap_against_the_same_operator():
    data = _frame()
    winners = winner_block(data)
    row = winners.iloc[0]
    assert np.isclose(row["gap"], row["score_test"] - row["untrained_test"])
    assert np.isclose(row["gain_over_mean"], row["score_test"] - row["mean_pool_test"])


def test_witness_rate_block_spans_the_temperature_grid():
    block = witness_rate_block(_frame())
    assert {"pr_auc_test_beta_min", "pr_auc_test_beta_max", "beta_star_test",
            "rel_gain_mean_to_max"} <= set(block.columns)


def test_kmatch_block_drops_operators_missing_an_arm():
    """rff_meanmap was never run at kmatch16 on v1; a partial row would emit a meaningless NaN delta."""
    data = _frame()
    data = data[~((data.pooling == "rff_meanmap") & (data.bag_arm == "kmatch16"))]
    block = kmatch_block(data)
    assert "rff_meanmap" not in set(block.pooling)
    assert not block[["first_seg_k16", "kmatched_k16"]].isna().any().any()
