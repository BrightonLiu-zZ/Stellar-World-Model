"""Task 2 (plan 2026-07-22, D2) — extract first-segment window mu over the new-task pool.

The new-task pool lives outside the packed subset, so its windows are replayed straight from the
sequences npz (native 1024 -> absmax guard <= 20 -> subdivide to 256, exactly the exp01 pack recipe),
then encoded to posterior-mean mu by each requested arm. One .npz cache per arm, in the same layout
readout_sweep.cached_mu emits, so the scorecard reads it with no GPU pass. This is an encode pass, not
training: no gradients, no W&B, safe as a backgrounded Claude Code job (CPU) or fast on a GPU terminal.

Arms: the exp03 leader `fb0_b0p1_comb` best_recon_aux at 3 seeds + one capacity-matched untrained.

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.new_task_extract --limit 40           # smoke over the first 40 pool TICs
    PYTHONUNBUFFERED=1 python -m swm.eval.new_task_extract    # full pool, all arms
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.eval.readout_sweep import build_model_from_ckpt, encode_blocks
from swm.eval.skyline import _make_untrained
from swm.models import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("new_task_extract")

repo_root = Path(__file__).resolve().parents[3]
_NPZ_RE = re.compile(r"^TIC(\d+)_s(\d+)_seg(\d+)_run\d+\.npz$")

LEADER_DIR = repo_root / "experiments" / "exp03_fb0_b0p1_comb" / "models"
SEEDS = (0, 1, 2)
MAX_ABSMAX = 20.0  # exp01 pack_manifest guard: drop native windows whose |flux| exceeds this


def index_pool_npz(seq_dir: Path, pool_tics: set[int]) -> dict[int, tuple[int, int, Path]]:
    """One scandir pass --> {tic: (sector, seg_idx, path)} for each pool TIC's FIRST segment (min sector,seg)."""
    best: dict[int, tuple[int, int, Path]] = {}
    with os.scandir(seq_dir) as it:
        for entry in it:
            m = _NPZ_RE.match(entry.name)
            if not m:
                continue
            tic = int(m.group(1))
            if tic not in pool_tics:
                continue
            key = (int(m.group(2)), int(m.group(3)))
            if tic not in best or key < (best[tic][0], best[tic][1]):
                best[tic] = (key[0], key[1], Path(entry.path))
    return best


def replay_first_segment(npz_path: Path, window: int) -> np.ndarray:
    """Load one segment npz, apply the absmax guard at native granularity, subdivide to (n_win, window)."""
    with np.load(npz_path) as data:
        windows = data["windows"]  # (N, native, 1) float32
    native = windows.shape[1]
    assert native % window == 0, f"window {window} does not divide native {native}"
    absmax = np.abs(windows).max(axis=(1, 2))  # (N,) guard at stored granularity, exactly as pack.py
    keep = absmax <= MAX_ABSMAX
    block = windows[keep].reshape(-1, window)  # (M*k, window)
    return block.astype(np.float32)


def load_pool_blocks(seq_dir: Path, pool: pd.DataFrame, split: str, window: int,
                     npz_index: dict[int, tuple[int, int, Path]]) -> tuple[list[int], list[np.ndarray]]:
    """First-segment window block per pool star in one split, ascending tic order (drops stars with no kept windows)."""
    want = pool.loc[pool["split"] == split, "tic_id"].astype(int).sort_values().tolist()
    tics = []
    blocks = []
    for tic in tqdm(want, desc=f"read npz[{split}]", total=len(want)):
        entry = npz_index.get(tic)
        assert entry is not None, f"pool TIC {tic} has no npz under {seq_dir}"
        block = replay_first_segment(entry[2], window)
        if block.shape[0] == 0:
            continue
        tics.append(tic)
        blocks.append(block)
    return tics, blocks


def extract_arm(model: WorldModel, blocks_by_split: dict[str, tuple[list[int], list[np.ndarray]]],
                device: str, cache_path: Path, arm: str) -> None:
    """Encode the pre-loaded first-segment blocks with one arm and save the concat mu + per-star counts."""
    arrays = {}
    for split in ["train", "val", "test"]:
        tics, blocks = blocks_by_split[split]
        mu_blocks = []
        for i in tqdm(range(len(blocks)), desc=f"{arm}[{split}]", total=len(blocks)):
            mu_blocks.append(encode_blocks(model, [blocks[i]], device)[0])
        counts = []
        for mu_block in mu_blocks:
            counts.append(mu_block.shape[0])
        arrays[f"{split}_mu"] = np.concatenate(mu_blocks, axis=0)
        arrays[f"{split}_counts"] = np.array(counts, dtype=np.int64)
        arrays[f"{split}_tics"] = np.array(tics, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    log.info(f"wrote {cache_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract first-segment mu over the new-task pool.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N pool TICs per split (smoke).")
    ap.add_argument("--arms", nargs="+", default=["seed0", "seed1", "seed2", "untrained"],
                    help="Subset of {seed0,seed1,seed2,untrained} to extract.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pool", default=None, help="Default: processed/subset/new_task_pool.parquet")
    ap.add_argument("--out-dir", default=None, help="Default: experiments/new_task/mu_cache")
    args = ap.parse_args()

    seq_dir = repo_root / "processed" / "sequences"
    pool_path = Path(args.pool) if args.pool else repo_root / "processed" / "subset" / "new_task_pool.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "experiments" / "new_task" / "mu_cache"
    pool = pd.read_parquet(pool_path)
    if args.limit:
        keep = []
        for split in ["train", "val", "test"]:
            keep.append(pool[pool["split"] == split].sort_values("tic_id").head(args.limit))
        pool = pd.concat(keep, ignore_index=True)
    log.info(f"pool: {len(pool)} TICs, device={args.device}")

    npz_index = index_pool_npz(seq_dir, set(pool["tic_id"].astype(int)))
    log.info(f"indexed first-segment npz for {len(npz_index)} pool TICs")

    base = torch.load(LEADER_DIR / "B_seed0" / "best_recon_aux.pt", map_location="cpu", weights_only=False)
    window = int(base["cfg"]["data"]["window"])  # all leader arms + untrained share the exp01 window (256)
    blocks_by_split = {}
    for split in ["train", "val", "test"]:
        blocks_by_split[split] = load_pool_blocks(seq_dir, pool, split, window, npz_index)  # read npz once, reuse

    for arm in args.arms:
        cache_path = out_dir / f"{arm}.npz"
        if arm == "untrained":
            mc = base["cfg"]["model"]
            model = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                                    window, int(mc["gru_hidden"]), int(mc["gru_layers"]), args.device)
        else:
            seed = int(arm.replace("seed", ""))
            assert seed in SEEDS, f"unknown seed arm {arm}"
            ck = torch.load(LEADER_DIR / f"B_seed{seed}" / "best_recon_aux.pt", map_location="cpu",
                            weights_only=False)
            model, _ = build_model_from_ckpt(ck, args.device)
        extract_arm(model, blocks_by_split, args.device, cache_path, arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
