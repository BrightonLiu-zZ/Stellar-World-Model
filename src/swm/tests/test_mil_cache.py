"""Unit tests for the MIL bag-cache layer (swm.eval.mil_cache, plan 2026-07-25).

Two things must hold for the sweep to be trustworthy across pools: a cache written before this module
existed (new_task_extract's, which carries no segment bookkeeping) must still load with sane offsets,
and each pool's arms must resolve to their own files so the v1 and new-task populations can never be
silently scored against one another's mu.
"""
from __future__ import annotations

import numpy as np

from swm.eval.mil_cache import cache_path, load_cache


def _write_cache(path, with_segments: bool, counts_per_split: int = 4, n_win: int = 6, dim: int = 5):
    """Write a minimal cache in the readout_sweep layout, optionally with the segment bookkeeping."""
    arrays = {}
    for split in ["train", "val", "test"]:
        counts = np.full(counts_per_split, n_win, dtype=np.int64)
        arrays[f"{split}_mu"] = np.arange(counts.sum() * dim, dtype=np.float16).reshape(-1, dim)
        arrays[f"{split}_counts"] = counts
        arrays[f"{split}_tics"] = np.arange(counts_per_split, dtype=np.int64)
        if with_segments:
            arrays[f"{split}_seg_counts"] = np.tile(np.array([2, 4], dtype=np.int64), counts_per_split)
            arrays[f"{split}_n_segs"] = np.full(counts_per_split, 2, dtype=np.int64)
    np.savez(path, **arrays)


def test_cache_without_segment_counts_falls_back_to_one_segment_per_bag(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_cache(path, with_segments=False)
    cache = load_cache(path)
    for split in ["train", "val", "test"]:
        tics, blocks, offsets = cache[split]
        assert len(tics) == len(blocks) == len(offsets) == 4
        for block, offset in zip(blocks, offsets):
            assert block.shape == (6, 5)
            assert np.array_equal(offset, np.array([0, 6])) # whole bag is one contiguous segment


def test_cache_with_segment_counts_recovers_the_boundaries(tmp_path):
    path = tmp_path / "modern.npz"
    _write_cache(path, with_segments=True)
    _, blocks, offsets = load_cache(path)["train"]
    for block, offset in zip(blocks, offsets):
        assert np.array_equal(offset, np.array([0, 2, 6])) # two segments of 2 and 4 windows
        assert offset[-1] == block.shape[0]


def test_cache_is_returned_as_float32_regardless_of_stored_precision(tmp_path):
    path = tmp_path / "half.npz"
    _write_cache(path, with_segments=True)
    _, blocks, _ = load_cache(path)["train"]
    assert blocks[0].dtype == np.float32 # stored fp16 to halve disk, widened for sklearn


def test_pool_switch_resolves_to_separate_files():
    """Each (pool, scope) must own its files, so two star populations can never be scored on one cache."""
    v1 = cache_path("exp05_comb_fbwd_c1p0", 2, "all", pool="v1")
    assert v1.name == "exp05_comb_fbwd_c1p0_s2_all.npz"
    assert v1.parent.name == "bag_cache"

    reused = cache_path("exp05_comb_fbwd_c1p0", 2, "first", pool="new_task")
    assert reused.name == "comb_fbwd_c1p0_s2.npz" # new_task_extract drops the prefix and the scope
    assert reused.parent.name == "mu_cache" # first-segment arms reuse the caches that already exist
    assert cache_path("untrained", 0, "first", pool="new_task").name == "untrained.npz"

    built = cache_path("exp05_comb_fbwd_c1p0", 2, "all", pool="new_task")
    assert built.parent.name == "bag_cache_new_task" # all-segment has no new_task_extract counterpart
    assert built.parent != v1.parent
    assert built != reused
