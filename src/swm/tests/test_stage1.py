"""Unit tests for the Stage 1/2 building blocks: stop-grad, free-bits, dataset boundaries, shapes, split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from swm.data.dataset import SeqWindowDataset
from swm.models import WorldModel
from swm.train.losses import kl_free_bits

repo_root = Path(__file__).resolve().parents[3]


def make_small_model() -> WorldModel:
    return WorldModel(
        in_ch=1, enc_channels=[8, 16, 32, 64], kernel_size=5, z_dim=8, window=64, gru_hidden=16, gru_layers=1
    )


def test_dynamics_target_is_stop_gradient():
    model = make_small_model()
    x = torch.randn(3, 4, 64, 1) # (B, S, window, 1)
    out = model(x)
    assert out["target_next"].requires_grad is False
    assert torch.equal(out["target_next"], out["mu_seq"][:, 1:, :].detach())


def test_free_bits_floor():
    z_dim = 8
    free_bits = 0.5
    mu = torch.zeros(2, 4, z_dim) # KL of a zero-mean unit-variance posterior is 0
    logvar = torch.zeros(2, 4, z_dim)
    kl_loss, kl_total, kl_dim = kl_free_bits(mu, logvar, free_bits)
    assert float(kl_total) == pytest.approx(0.0, abs=1e-6) # true KL is zero
    assert float(kl_loss) == pytest.approx(free_bits * z_dim, abs=1e-6) # floored to free_bits per dim


def test_encoder_decoder_shape_roundtrip():
    model = make_small_model()
    x = torch.randn(3, 4, 64, 1)
    out = model(x)
    assert out["recon"].shape == x.shape
    assert out["mu_seq"].shape == (3, 4, 8)
    assert out["pred_next"].shape == (3, 3, 8)


def test_dataset_no_cross_boundary_no_pad(tmp_path: Path):
    window = 8
    seq_len = 4
    n_win = 6
    # one segment whose row r is filled with the constant r, so a returned sequence reveals its rows
    block = np.zeros((n_win, window), dtype=np.float32)
    for r in range(n_win):
        block[r, :] = float(r)
    (tmp_path / "train_windows.dat").write_bytes(block.tobytes())
    index = pd.DataFrame([{"seg_id": "seg0", "tic_id": 1, "sector": 1, "seg_idx": 0, "row_start": 0, "n_win": n_win}])
    index.to_parquet(tmp_path / "train_index.parquet", index=False)

    dataset = SeqWindowDataset(tmp_path, "train", seq_len, window, randomize=True)
    for _ in range(50):
        x = dataset[0] # (seq_len, window, 1)
        assert x.shape == (seq_len, window, 1)
        rows = x[:, 0, 0].numpy() # the per-window constant = its row index
        assert rows[0] >= 0 and rows[-1] <= n_win - 1 # stays inside the segment
        steps = np.diff(rows)
        assert np.all(steps == 1.0) # consecutive, never crosses a gap or wraps


def test_frozen_split_is_star_disjoint():
    split_path = repo_root / "processed" / "subset" / "split.parquet"
    if not split_path.exists():
        pytest.skip("split.parquet not built yet")
    split = pd.read_parquet(split_path)
    counts = split.groupby("tic_id")["split"].nunique()
    assert int(counts.max()) == 1 # every TIC lives in exactly one fold
