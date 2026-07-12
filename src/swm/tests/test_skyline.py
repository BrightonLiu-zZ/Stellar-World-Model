"""Unit tests for the skyline scoring + bootstrap helpers (swm.eval.skyline).

The logistic scorer must separate a linearly-separable feature and collapse to the base rate under a
label shuffle (pipeline validity, no leakage); the paired bootstrap must return one aligned PR-AUC
column per method so any pairwise column difference is a genuine paired difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swm.eval.skyline import logistic_scores, paired_bootstrap_ap


def _toy_table(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Build a train/test table where feature f0 carries the label and f1 is pure noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    f0 = y + rng.normal(0, 0.3, n) # signal feature
    f1 = rng.normal(0, 1, n) # noise feature
    split = np.where(np.arange(n) < n // 2, "train", "test")
    df = pd.DataFrame({"tic_id": np.arange(n), "split": split, "task": y, "f0": f0, "f1": f1})
    return df, ["f0", "f1"]


def test_logistic_separates_signal_feature():
    df, cols = _toy_table()
    _, y_test, scores = logistic_scores(df, cols, "task")
    from sklearn.metrics import average_precision_score
    assert average_precision_score(y_test, scores) > 0.7 # signal feature is recoverable


def test_label_shuffle_collapses_to_base_rate():
    df, cols = _toy_table()
    shuffled = df.copy()
    shuffled["task"] = np.random.default_rng(1).permutation(shuffled["task"].to_numpy())
    _, y_test, scores = logistic_scores(shuffled, cols, "task")
    from sklearn.metrics import average_precision_score
    base = float(y_test.mean())
    assert average_precision_score(y_test, scores) < base + 0.15 # no signal left -> near base rate


def test_paired_bootstrap_is_aligned():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    good = y + rng.normal(0, 0.2, 300)
    bad = rng.normal(0, 1, 300)
    boot = paired_bootstrap_ap(y, {"good": good, "bad": bad}, n_boot=200, seed=0)
    assert set(boot.columns) == {"good", "bad"}
    assert (boot["good"] > boot["bad"]).mean() > 0.9 # the informative scorer wins on almost every resample
    assert float((boot["good"] - boot["bad"]).std()) > 0 # paired SE_diff is well defined
