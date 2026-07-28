"""Unit tests for pack.py load_and_filter across the three window regimes (split / passthrough / merge).

The merge path (window > native, exp06 w2048) is the new one: adjacent stored windows concatenate in
fixed chunks, a chunk survives only if EVERY member passes the absmax guard, and a ragged tail is
discarded - a merged window must never span a dropped stored window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from swm.data.pack import load_and_filter

NATIVE = 8 # stand-in for the corpus's stored 1024


def write_npz(tmp_path: Path, windows: np.ndarray) -> str:
    """Save a (N, NATIVE, 1) stack the way build_sequences_bulk stores segments."""
    path = tmp_path / "TIC1_s1_seg0.npz"
    np.savez(path, windows=windows.astype(np.float32))
    return str(path)


def ramp_windows(n: int) -> np.ndarray:
    """Window r filled with the constant r, so any reordering or mispairing is visible in the values."""
    block = np.zeros((n, NATIVE, 1), dtype=np.float32)
    for r in range(n):
        block[r] = float(r)
    return block


def test_merge_concatenates_adjacent_pairs_in_time_order(tmp_path: Path):
    path = write_npz(tmp_path, ramp_windows(6))
    flat, dropped_w, dropped_s = load_and_filter(path, window=2 * NATIVE, seq_len=2, max_absmax=20.0)
    assert dropped_w == 0 and dropped_s == 0
    assert flat.shape == (3, 2 * NATIVE) # (0,1), (2,3), (4,5)
    for c, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)]):
        assert (flat[c, :NATIVE] == a).all() and (flat[c, NATIVE:] == b).all()


def test_merge_drops_whole_chunk_when_one_member_fails_absmax(tmp_path: Path):
    windows = ramp_windows(6)
    windows[2, 0, 0] = 99.0 # stored window 2 fails the guard -> chunk (2,3) must die entirely
    path = write_npz(tmp_path, windows)
    flat, dropped_w, dropped_s = load_and_filter(path, window=2 * NATIVE, seq_len=2, max_absmax=20.0)
    assert dropped_w == 1 # absmax counter stays at STORED granularity: only window 2, not its partner
    assert flat.shape == (2, 2 * NATIVE)
    assert (flat[0, NATIVE:] == 1).all() and (flat[1, :NATIVE] == 4).all() # survivors are (0,1) and (4,5)


def test_merge_discards_ragged_tail(tmp_path: Path):
    path = write_npz(tmp_path, ramp_windows(5)) # chunks (0,1), (2,3); window 4 is an unpaired tail
    flat, _, _ = load_and_filter(path, window=2 * NATIVE, seq_len=2, max_absmax=20.0)
    assert flat.shape == (2, 2 * NATIVE)
    assert (flat != 4).all()


def test_merge_segment_dropped_below_seq_len(tmp_path: Path):
    path = write_npz(tmp_path, ramp_windows(3)) # one merged window < seq_len=2
    flat, _, dropped_s = load_and_filter(path, window=2 * NATIVE, seq_len=2, max_absmax=20.0)
    assert flat is None and dropped_s == 1


def test_split_and_passthrough_regimes_unchanged(tmp_path: Path):
    windows = ramp_windows(4)
    windows[1, 0, 0] = 99.0
    path = write_npz(tmp_path, windows)
    # passthrough (window == native): failing window dropped alone
    flat, dropped_w, _ = load_and_filter(path, window=NATIVE, seq_len=2, max_absmax=20.0)
    assert dropped_w == 1 and flat.shape == (3, NATIVE)
    # split (window == native/2): guard applied BEFORE splitting, survivors split 2x
    flat, dropped_w, _ = load_and_filter(path, window=NATIVE // 2, seq_len=2, max_absmax=20.0)
    assert dropped_w == 1 and flat.shape == (6, NATIVE // 2)
