"""Unit tests for the Tier-3 learned pooling heads (swm.eval.mil_learned, plan 2026-07-25).

Padded bags are the whole reason these heads are tractable, so the tests that matter are the ones
that prove padding is inert: a masked window must not change the bag score, and neither must the
order of the real windows. Anything else silently makes bag size a feature, which is the exact
confound the sweep exists to control for.
"""
from __future__ import annotations

import numpy as np
import torch

from swm.eval.mil_learned import DSMIL, GatedABMIL, make_pseudo_bags, pad_batch


def _bag(n_win: int, dim: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_win, dim)).astype(np.float32)


def test_padding_does_not_change_the_bag_score():
    torch.manual_seed(0)
    dim = 16
    blocks = [_bag(5, dim, 0), _bag(12, dim, 1)]
    x, mask = pad_batch(blocks, "cpu")
    for model in [GatedABMIL(dim), DSMIL(dim)]:
        model.eval()
        with torch.no_grad():
            padded = model(x, mask)
            alone = []
            for block in blocks:
                xi, mi = pad_batch([block], "cpu")
                alone.append(model(xi, mi))
            solo = torch.cat(alone)
        assert torch.allclose(padded, solo, atol=1e-5)


def test_bag_score_is_permutation_invariant():
    torch.manual_seed(0)
    dim = 16
    block = _bag(9, dim, 2)
    shuffled = block[np.random.default_rng(3).permutation(9)]
    for model in [GatedABMIL(dim), DSMIL(dim)]:
        model.eval()
        with torch.no_grad():
            a = model(*pad_batch([block], "cpu"))
            b = model(*pad_batch([shuffled], "cpu"))
        assert torch.allclose(a, b, atol=1e-5)


def test_attention_never_lands_on_a_padded_window():
    torch.manual_seed(0)
    dim = 8
    x, mask = pad_batch([_bag(3, dim, 4), _bag(11, dim, 5)], "cpu")
    model = GatedABMIL(dim)
    model.eval()
    with torch.no_grad():
        a = model.attention(x, mask)
    assert torch.allclose(a[~mask], torch.zeros_like(a[~mask]))
    assert torch.allclose(a.sum(dim=1), torch.ones(2), atol=1e-5)


def test_pseudo_bags_multiply_bag_count_and_inherit_labels():
    blocks = [_bag(40, 4, 6), _bag(3, 4, 7)]
    y = np.array([1, 0])
    rng = np.random.default_rng(0)
    sub_blocks, sub_y = make_pseudo_bags(blocks, y, 4, rng)
    assert len(sub_blocks) == 5 # the 40-window bag splits into 4, the 3-window bag stays whole
    assert sub_y.tolist() == [1, 1, 1, 1, 0]
    total = 0
    for block in sub_blocks:
        total += block.shape[0]
    assert total == 43 # every window is dealt exactly once, none duplicated or dropped


def test_pseudo_bags_leave_small_bags_untouched():
    blocks = [_bag(6, 4, 8)]
    sub_blocks, sub_y = make_pseudo_bags(blocks, np.array([1]), 4, np.random.default_rng(0))
    assert len(sub_blocks) == 1
    assert sub_blocks[0].shape[0] == 6
