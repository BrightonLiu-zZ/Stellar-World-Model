"""Unit tests for exp03 dual-checkpoint tracking: best_recon_aux.pt lifecycle, defaults, resume metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from swm.train.loop import train

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="train() targets the GPU")


def write_packed_split(packed_dir: Path, split: str, n_seg: int, n_win: int, window: int) -> None:
    """Write one synthetic packed split: random windows plus the row-range index the Dataset expects."""
    rng = np.random.default_rng(0)
    rows = []
    blocks = []
    row_start = 0
    for seg in range(n_seg):
        blocks.append(rng.normal(size=(n_win, window)).astype(np.float32))
        rows.append({"seg_id": f"s{seg}", "tic_id": seg, "sector": 1, "seg_idx": 0, "row_start": row_start, "n_win": n_win})
        row_start += n_win
    (packed_dir / f"{split}_windows.dat").write_bytes(np.concatenate(blocks, axis=0).tobytes())
    pd.DataFrame(rows).to_parquet(packed_dir / f"{split}_index.parquet", index=False)


def make_cfg(tmp_path: Path, track: bool, max_epochs: int = 4):
    """Build the minimal resolved config train() needs, on a tiny model and the synthetic packed data."""
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir(exist_ok=True)
    if not (packed_dir / "train_index.parquet").exists():
        write_packed_split(packed_dir, "train", n_seg=4, n_win=4, window=32)
        write_packed_split(packed_dir, "val", n_seg=2, n_win=4, window=32)
    return OmegaConf.create({
        "seed": 0,
        "variant_name": "B",
        "exp_name": "test_dual_ckpt",
        "paths": {"packed_dir": str(packed_dir), "models_dir": str(tmp_path / "models")},
        "model": {"in_ch": 1, "enc_channels": [8, 16, 32, 64], "kernel_size": 5, "z_dim": 8,
                  "gru_hidden": 16, "gru_layers": 1},
        "data": {"window": 32, "seq_len": 2, "batch_size": 4, "num_workers": 0},
        "train": {
            "lambda_dyn": 1.0, "beta_target": 1.0, "beta_warmup_epochs": 1, "free_bits": 0.1,
            "lr": 3e-4, "max_epochs": max_epochs, "patience": 50, "grad_clip": 1.0, "amp": False,
            "accum_steps": 1, "active_unit_kl_threshold": 0.01, "resume": False,
            "track_recon_aux_best": track,
            "recon_aux": {"type": "none", "weight": 0.0, "psd_normalize": False, "psd_eps": 1e-8,
                          "hf_weight": 1.0, "mask_frac": 0.0, "mask_span": 8},
            "wandb": {"mode": "disabled", "project": "test", "entity": None},
        },
    })


def test_dual_checkpoint_saved_when_tracking(tmp_path: Path):
    cfg = make_cfg(tmp_path, track=True)
    train(cfg)
    run_dir = tmp_path / "models" / "B_seed0"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "best_recon_aux.pt").exists()
    ckpt = torch.load(run_dir / "best_recon_aux.pt", map_location="cpu", weights_only=False)
    assert ckpt["best_select"] is not None and np.isfinite(ckpt["best_select"])
    # the KL-free best must equal recon + w*aux + lambda*dyn at its epoch, so it is bounded by the monitor
    assert float(ckpt["best_select"]) < float(ckpt["best_val"])


def test_default_off_reproduces_single_checkpoint(tmp_path: Path):
    cfg = make_cfg(tmp_path, track=False)
    train(cfg)
    run_dir = tmp_path / "models" / "B_seed0"
    assert (run_dir / "best.pt").exists()
    assert not (run_dir / "best_recon_aux.pt").exists() # flag off --> exp00-02 behavior, no second best


def test_resume_restores_best_select(tmp_path: Path):
    cfg = make_cfg(tmp_path, track=True, max_epochs=3)
    train(cfg)
    run_dir = tmp_path / "models" / "B_seed0"
    before = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    cfg2 = make_cfg(tmp_path, track=True, max_epochs=5)
    cfg2.train.resume = True
    train(cfg2)
    after = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert int(after["epoch"]) == 4 # resumed at 3, ran to max_epochs-1
    assert float(after["best_select"]) <= float(before["best_select"]) # best carries over, never worsens
