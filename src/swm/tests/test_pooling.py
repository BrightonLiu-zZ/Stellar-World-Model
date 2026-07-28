"""Unit tests for the MIL pooling operators (swm.eval.pooling, plan 2026-07-25).

The properties that matter are the ones the sweep's interpretation rests on: ws_lse must actually
interpolate mean-to-max so beta* can be read as a witness-rate estimate, ws_topk must reduce to max
and mean at its endpoints, the order-aware operators must never look across a segment boundary, and
Noisy-AND must be rank-equivalent to ws_mean (which is why it is excluded from the sweep).
"""
from __future__ import annotations

import numpy as np

from swm.eval.pooling import (FeaturePooling, aggregate_scores, bag_size_features, noisy_and,
                              segment_offsets_from_counts, subsample_bags, trivial_offsets)


def _one_bag_scores() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """A single 6-window star holding one loud window: the localized-signal case the sweep is about."""
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1])
    counts = np.array([6])
    return scores, counts, [np.array([0, 6])]


def test_lse_interpolates_mean_to_max():
    scores, counts, offsets = _one_bag_scores()
    cold = aggregate_scores(scores, counts, offsets, "ws_lse", 1e-4)[0]
    hot = aggregate_scores(scores, counts, offsets, "ws_lse", 1e4)[0]
    assert abs(cold - scores.mean()) < 1e-3 # beta --> 0 recovers the mean
    assert abs(hot - scores.max()) < 1e-3 # beta --> inf recovers the max
    warm = aggregate_scores(scores, counts, offsets, "ws_lse", 10.0)[0]
    assert scores.mean() < warm < scores.max() # monotone in between


def test_lse_is_monotone_in_beta():
    scores, counts, offsets = _one_bag_scores()
    values = []
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        values.append(aggregate_scores(scores, counts, offsets, "ws_lse", beta)[0])
    assert np.all(np.diff(values) > 0)


def test_lse_overflow_safe_at_large_beta():
    scores, counts, offsets = _one_bag_scores()
    assert np.isfinite(aggregate_scores(scores, counts, offsets, "ws_lse", 1e5)[0])


def test_topk_endpoints_match_max_and_mean():
    scores, counts, offsets = _one_bag_scores()
    assert aggregate_scores(scores, counts, offsets, "ws_topk", 1)[0] == scores.max()
    assert abs(aggregate_scores(scores, counts, offsets, "ws_topk", 6)[0] - scores.mean()) < 1e-12


def test_localized_signal_ranks_above_diffuse_only_under_max_like_pooling():
    """The dilution failure in one assertion: mean ranks a mildly noisy star above a real transit."""
    transit = np.array([0.1, 0.1, 0.1, 0.9, 0.1, 0.1])
    noisy = np.array([0.30, 0.35, 0.30, 0.32, 0.30, 0.30])
    scores = np.concatenate([transit, noisy])
    counts = np.array([6, 6])
    offsets = [np.array([0, 6]), np.array([0, 6])]
    mean_pooled = aggregate_scores(scores, counts, offsets, "ws_mean")
    lse_pooled = aggregate_scores(scores, counts, offsets, "ws_lse", 10.0)
    assert mean_pooled[0] < mean_pooled[1] # mean gets it wrong
    assert lse_pooled[0] > lse_pooled[1] # a warm temperature gets it right


def test_linsoftmax_between_mean_and_max_and_parameter_free():
    scores, counts, offsets = _one_bag_scores()
    value = aggregate_scores(scores, counts, offsets, "ws_linsoftmax")[0]
    assert scores.mean() < value < scores.max()


def test_noisy_and_is_rank_equivalent_to_mean():
    """Documented reason Noisy-AND is not a sweep cell: identical star ranking, identical PR-AUC."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=60)
    counts = np.full(10, 6)
    offsets = []
    for _ in range(10):
        offsets.append(np.array([0, 6]))
    mean_pooled = aggregate_scores(scores, counts, offsets, "ws_mean")
    na = noisy_and(scores, counts)
    assert np.array_equal(np.argsort(mean_pooled), np.argsort(na))


def test_smoothing_never_crosses_a_segment_boundary():
    """A spike at the end of segment 0 must not leak into segment 1, which can be months later."""
    scores = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    counts = np.array([6])
    offsets = [np.array([0, 3, 6])]
    smoothed_star = aggregate_scores(scores, counts, offsets, "ws_smooth", 0.5)[0]
    assert smoothed_star > 0
    from swm.eval.pooling import _smooth_within_segments
    smoothed = _smooth_within_segments(scores, offsets[0], 0.5)
    assert np.allclose(smoothed[3:], 0.0) # segment 1 was all zeros and stays all zeros


def test_longest_run_never_crosses_a_segment_boundary():
    scores = np.array([0.0, 0.9, 0.9, 0.9, 0.9, 0.0])
    counts = np.array([6])
    split = aggregate_scores(scores, counts, [np.array([0, 3, 6])], "ws_ppv_lspv", 0.25)[0]
    joined = aggregate_scores(scores, counts, [np.array([0, 6])], "ws_ppv_lspv", 0.25)[0]
    assert split[2] < joined[2] # the run of 4 is split into 2 + 2 when a boundary sits inside it


def test_ppv_lspv_returns_five_named_statistics():
    scores, counts, offsets = _one_bag_scores()
    row = aggregate_scores(scores, counts, offsets, "ws_ppv_lspv", 0.5)
    assert row.shape == (1, 5)
    assert 0.0 <= row[0, 0] <= 1.0 # ppv is a fraction
    assert 0.0 <= row[0, 2] <= 1.0 # lspv is normalized by bag size


def test_feature_poolings_emit_expected_widths():
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(40):
        blocks.append(rng.normal(size=(16, 12)))
    widths = {"mean": 12, "max": 12, "quantile3": 36, "quantile5": 60, "moments": 36}
    for kind, width in widths.items():
        pooled = FeaturePooling(kind).fit(blocks).transform(blocks)
        assert pooled.shape == (40, width)
    pca = FeaturePooling("pca32_quantile5", 8).fit(blocks).transform(blocks)
    assert pca.shape == (40, 40)
    rff = FeaturePooling("rff_meanmap", 64).fit(blocks).transform(blocks)
    assert rff.shape == (40, 64)
    gmm = FeaturePooling("gmm_prototype", 4).fit(blocks).transform(blocks)
    assert gmm.shape == (40, 4 + 4 * 12 * 2)


def test_moments_skew_is_negative_for_one_sided_dips():
    """Eclipses and transits are one-sided negative excursions, which is why skew is in the menu."""
    block = np.zeros((20, 1))
    block[3, 0] = -5.0
    skew = FeaturePooling("moments").fit([block]).transform([block])[0, 2]
    assert skew < -1.0


def test_feature_pooling_state_is_fit_on_train_only():
    rng = np.random.default_rng(0)
    train = []
    for _ in range(30):
        train.append(rng.normal(size=(16, 12)))
    test = []
    for _ in range(7):
        test.append(rng.normal(loc=5.0, size=(16, 12)))
    op = FeaturePooling("pca32_quantile5", 8).fit(train)
    before = op.pca.mean_.copy()
    op.transform(test)
    assert np.array_equal(before, op.pca.mean_) # transforming test data must not refit anything


def test_segment_offsets_round_trip():
    offsets = segment_offsets_from_counts([np.array([16, 16, 20])])
    assert np.array_equal(offsets[0], np.array([0, 16, 32, 52]))


def test_subsample_bags_matches_k0_and_keeps_valid_offsets():
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(80, 4)), rng.normal(size=(10, 4))]
    offsets = segment_offsets_from_counts([np.array([16, 16, 16, 16, 16]), np.array([10])])
    small, small_offsets = subsample_bags(blocks, offsets, 16, seed=0)
    assert small[0].shape[0] == 16
    assert small[1].shape[0] == 10 # a bag already below K0 is returned whole
    for block, offset in zip(small, small_offsets):
        assert offset[0] == 0
        assert offset[-1] == block.shape[0]
        assert np.all(np.diff(offset) > 0)


def test_bag_size_control_tracks_window_count():
    blocks = [np.zeros((16, 4)), np.zeros((64, 4))]
    sizes = bag_size_features(blocks)
    assert sizes.shape == (2, 1)
    assert sizes[1, 0] > sizes[0, 0]


def test_trivial_offsets_cover_the_whole_bag():
    blocks = [np.zeros((16, 4)), np.zeros((20, 4))]
    for block, offset in zip(blocks, trivial_offsets(blocks)):
        assert np.array_equal(offset, np.array([0, block.shape[0]]))
