"""Stage 2 step 1: extract frozen-encoder latents for the linear probe.

Loads a trained variant's encoder, encodes every packed window with the deterministic posterior
mean mu (no sampling), mean-pools mu over all windows of a segment, then mean-pools a star's
segment vectors into one vector per star. Per-segment vectors are cached too (for a deferred
per-segment probing view); the per-star vectors are what the probe consumes.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.eval.extract variant=B seed=0
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from swm.train.loop import build_model

log = logging.getLogger(__name__)


@torch.no_grad()
def encode_split(model, dat_path: Path, total_rows: int, window: int, batch: int, device: str) -> np.ndarray:
    """
    Encode every window of one split to its posterior mean in row order.
    Reads the flat memmap in chunks and returns mu for all rows, so segment means are then a
    simple slice-and-average over the index row ranges. Deterministic (mu only, no reparameterize).
    """
    windows = np.memmap(dat_path, dtype=np.float32, mode="r", shape=(total_rows, window))
    chunks = []
    for i in range(0, total_rows, batch):
        block = np.array(windows[i : i + batch], dtype=np.float32) # (b, window)
        x = torch.from_numpy(block).unsqueeze(-1).to(device) # (b, window, 1)
        mu = model.encode_mu(x) # (b, z)
        chunks.append(mu.float().cpu().numpy())
    return np.concatenate(chunks, axis=0) # (total_rows, z)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda"
    z_dim = int(cfg.model.z_dim)
    mu_cols = []
    for j in range(z_dim):
        mu_cols.append(f"mu{j}")

    run_name = f"{cfg.variant_name}_seed{cfg.seed}"
    ckpt_dir = Path(cfg.paths.models_dir) / run_name
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "last.pt"
    assert ckpt_path.exists(), f"no checkpoint for {run_name}; train it first"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) # ckpt holds cfg dict + RNG state
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"extracting with {ckpt_path}")

    packed = Path(cfg.paths.packed_dir)
    out_dir = ckpt_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_frames = []
    star_frames = []
    for split in ["train", "val", "test"]:
        index_path = packed / f"{split}_index.parquet"
        if not index_path.exists():
            continue
        index = pd.read_parquet(index_path).reset_index(drop=True)
        if len(index) == 0:
            continue
        total_rows = int(index["n_win"].sum())
        mu_all = encode_split(model, packed / f"{split}_windows.dat", total_rows, cfg.data.window, 4096, device)

        seg_mu = np.zeros((len(index), z_dim), dtype=np.float32)
        for k in range(len(index)):
            start = int(index["row_start"].iloc[k])
            n_win = int(index["n_win"].iloc[k])
            seg_mu[k] = mu_all[start : start + n_win].mean(axis=0) # pool over all windows of the segment

        seg_meta = index[["seg_id", "tic_id", "sector", "seg_idx"]].copy()
        seg_meta["split"] = split
        mu_frame = pd.DataFrame(seg_mu, columns=mu_cols)
        seg_df = pd.concat([seg_meta.reset_index(drop=True), mu_frame], axis=1) # one concat avoids per-column insert
        seg_frames.append(seg_df)

        star_df = seg_df.groupby("tic_id")[mu_cols].mean().reset_index() # pool over a star's segments
        star_df["split"] = split
        star_frames.append(star_df)
        log.info(f"[{split}] {len(index)} segments --> {len(star_df)} stars")

    pd.concat(seg_frames, ignore_index=True).to_parquet(out_dir / "segment_mu.parquet", index=False)
    pd.concat(star_frames, ignore_index=True).to_parquet(out_dir / "star_mu.parquet", index=False)
    log.info(f"wrote segment_mu + star_mu to {out_dir}")


if __name__ == "__main__":
    main()
