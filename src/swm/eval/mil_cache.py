"""
Bag caches for the MIL pooling sweep (plan 2026-07-25): encode each star's windows once per
(encoder arm, bag scope) and store them with the segment structure intact.

Two differences from swm.eval.readout_sweep.cached_mu, which this does not replace:

  bag scope   `first` keeps only each star's first packed segment (16-20 windows, the protocol of
              every table through exp05). `all` keeps every packed window of the star (median 32,
              mean 62, max 816). Both are cached so the sweep can report their delta.
  val split   readout_sweep caches train+test only. The val split has never been used, which makes
              it a clean selection set for the pooling hyperparameters; caches here hold all three.

Windows are encoded in large batches (measured 93k win/s at batch 2048 versus 8k win/s at the
per-star batch of 16) and mu is stored float16, which halves the ~5 GB all-segment footprint and is
far below the precision a linear probe on O(0.1) values can resolve.

Run (swm env, from repo root, PYTHONPATH=src):
    python -m swm.eval.mil_cache --cells exp05_comb_fbwd_c1p0 exp05_comb_off --seeds 0 1 2 3 --scope first
    PYTHONUNBUFFERED=1 python -m swm.eval.mil_cache --cells exp05_comb_fbwd_c1p0 --seeds 0 --scope all
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.eval.readout_sweep import build_model_from_ckpt
from swm.eval.skyline import _make_untrained

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

repo_root = Path(__file__).resolve().parents[3]
splits = ("train", "val", "test")


def load_bag_blocks(packed_dir: Path, split: str, window: int, scope: str,
                    ) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Read one split's packed windows and group them into per-star bags under the requested scope.
    Returns (tics, flux blocks, per-star segment window-counts) in ascending tic order, the same
    star ordering every other eval module uses so scores stay aligned across methods.
    """
    assert scope in ("first", "all"), f"unknown bag scope {scope!r}"
    index = pd.read_parquet(packed_dir / f"{split}_index.parquet")
    total = int(index["n_win"].sum())
    windows = np.memmap(packed_dir / f"{split}_windows.dat", dtype=np.float32, mode="r", shape=(total, window))
    ordered = index.sort_values(["tic_id", "sector", "seg_idx"])
    if scope == "first":
        ordered = ordered.drop_duplicates("tic_id")
    tics = []
    blocks = []
    seg_counts = []
    for tic, group in tqdm(ordered.groupby("tic_id", sort=True), desc=f"bags[{split}/{scope}]"):
        parts = []
        counts = []
        for row in group.itertuples(index=False):
            parts.append(np.array(windows[row.row_start : row.row_start + row.n_win], dtype=np.float32))
            counts.append(int(row.n_win))
        tics.append(int(tic))
        blocks.append(np.concatenate(parts, axis=0))
        seg_counts.append(np.array(counts, dtype=np.int64))
    return tics, blocks, seg_counts


def load_new_task_bags(seq_dir: Path, pool: pd.DataFrame, split: str, window: int, scope: str,
                       ) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Build all-segment bags for the new-task pool by replaying the raw sequence npz files.
    The pool lives OUTSIDE the packed subset, so there is no memmap to slice: each star's segments are
    read from processed/sequences, guarded at the stored granularity and subdivided exactly as
    pack.py does, which is the same recipe new_task_extract uses for its first-segment caches.
    Segments are ordered by (sector, seg_idx, run_idx) so the per-star window order is chronological.
    """
    from swm.eval.new_task_extract import MAX_ABSMAX, _NPZ_RE
    want = set(pool.loc[pool["split"] == split, "tic_id"].astype(int).tolist())
    by_tic: dict[int, list[tuple[int, int, str, Path]]] = {}
    with os.scandir(seq_dir) as it:
        for entry in it:
            match = _NPZ_RE.match(entry.name)
            if match is None:
                continue
            tic = int(match.group(1))
            if tic not in want:
                continue
            by_tic.setdefault(tic, []).append((int(match.group(2)), int(match.group(3)), entry.name, Path(entry.path)))
    tics = []
    blocks = []
    seg_counts = []
    for tic in tqdm(sorted(by_tic), desc=f"replay[{split}/{scope}]", total=len(by_tic)):
        entries = sorted(by_tic[tic])
        if scope == "first":
            entries = entries[:1]
        parts = []
        counts = []
        for _, _, _, path in entries:
            with np.load(path) as data:
                windows = data["windows"] # (N, native, 1)
            native = windows.shape[1]
            assert native % window == 0, f"window {window} does not divide native {native}"
            absmax = np.abs(windows).max(axis=(1, 2)) # guard at STORED granularity, exactly as pack.py
            survivors = windows[absmax <= MAX_ABSMAX]
            if survivors.shape[0] == 0:
                continue
            block = survivors.reshape(-1, window).astype(np.float32)
            parts.append(block)
            counts.append(block.shape[0])
        if len(parts) == 0:
            continue
        tics.append(tic)
        blocks.append(np.concatenate(parts, axis=0))
        seg_counts.append(np.array(counts, dtype=np.int64))
    return tics, blocks, seg_counts


def build_new_task_caches(arms: list[tuple[Path, "torch.nn.Module"]], pool: pd.DataFrame, seq_dir: Path,
                          window: int, scope: str, device: str) -> None:
    """
    Build every requested arm's new-task bag cache in ONE pass over the sequence npz files.
    Replaying ~90k npz is pure I/O and does not depend on the encoder, so the split loop is outermost
    and all arms are encoded from the same in-memory blocks; doing it per arm would repeat the read
    nine times for no reason. Blocks are freed before the next split to bound peak memory.
    """
    arrays: dict[Path, dict] = {}
    for out_path, _ in arms:
        arrays[out_path] = {}
    for split in splits:
        tics, blocks, seg_counts = load_new_task_bags(seq_dir, pool, split, window, scope)
        counts_meta = {
            f"{split}_tics": np.array(tics, dtype=np.int64),
            f"{split}_seg_counts": np.concatenate(seg_counts),
            f"{split}_n_segs": np.array([len(s) for s in seg_counts], dtype=np.int64),
        }
        for out_path, model in arms:
            mu_blocks = encode_batched(model, blocks, device)
            counts = []
            for mu_block in mu_blocks:
                counts.append(mu_block.shape[0])
            arrays[out_path][f"{split}_mu"] = np.concatenate(mu_blocks, axis=0)
            arrays[out_path][f"{split}_counts"] = np.array(counts, dtype=np.int64)
            arrays[out_path].update(counts_meta)
        del blocks
    for out_path, _ in arms:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **arrays[out_path])
        log.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6} MB)")


@torch.no_grad()
def encode_batched(model, blocks: list[np.ndarray], device: str, batch: int = 2048) -> list[np.ndarray]:
    """
    Encode every window of every bag to posterior-mean mu, batching across star boundaries.
    The encoder is position-independent, so flattening the bags and re-splitting afterwards is exact
    and roughly 11x faster than the per-star loop the older caches were built with.
    """
    counts = []
    for block in blocks:
        counts.append(block.shape[0])
    flat = np.concatenate(blocks, axis=0)
    mus = []
    for start in tqdm(range(0, flat.shape[0], batch), desc="encode", total=(flat.shape[0] + batch - 1) // batch):
        chunk = torch.from_numpy(flat[start : start + batch]).unsqueeze(-1).to(device) # (b, window, 1)
        mu, _ = model.encoder(chunk)
        mus.append(mu.float().cpu().numpy().astype(np.float16))
    out = np.concatenate(mus, axis=0)
    blocks_mu = []
    cursor = 0
    for count in counts:
        blocks_mu.append(out[cursor : cursor + count])
        cursor += count
    return blocks_mu


def build_cache(model, packed_dir: Path, window: int, scope: str, out_path: Path, device: str) -> None:
    """Encode all three splits for one arm and write a single npz holding mu, star counts, tics, segment counts."""
    arrays = {}
    for split in splits:
        tics, blocks, seg_counts = load_bag_blocks(packed_dir, split, window, scope)
        mu_blocks = encode_batched(model, blocks, device)
        counts = []
        for mu_block in mu_blocks:
            counts.append(mu_block.shape[0])
        arrays[f"{split}_mu"] = np.concatenate(mu_blocks, axis=0)
        arrays[f"{split}_counts"] = np.array(counts, dtype=np.int64)
        arrays[f"{split}_tics"] = np.array(tics, dtype=np.int64)
        arrays[f"{split}_seg_counts"] = np.concatenate(seg_counts)
        arrays[f"{split}_n_segs"] = np.array([len(s) for s in seg_counts], dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    log.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6} MB)")


def load_cache(path: Path) -> dict[str, tuple[list[int], list[np.ndarray], list[np.ndarray]]]:
    """
    Read a bag cache back into {split: (tics, mu blocks, per-star segment offsets)} for the sweep.
    Caches written by swm.eval.new_task_extract predate this module and carry no segment bookkeeping;
    they are first-segment-only, so each bag is one contiguous segment and the offsets are trivial.
    That fallback is what lets the new-task pool reuse its existing mu caches with no GPU pass.
    """
    assert path.exists(), f"no bag cache at {path}; build it with swm.eval.mil_cache"
    payload = np.load(path, allow_pickle=False)
    has_segments = f"{splits[0]}_seg_counts" in payload.files
    result = {}
    for split in splits:
        flat = payload[f"{split}_mu"].astype(np.float32)
        counts = payload[f"{split}_counts"]
        tics = payload[f"{split}_tics"].tolist()
        blocks = []
        offsets = []
        row = 0
        seg = 0
        if has_segments:
            seg_counts = payload[f"{split}_seg_counts"]
            n_segs = payload[f"{split}_n_segs"]
        for i, count in enumerate(counts):
            blocks.append(flat[row : row + int(count)])
            if has_segments:
                per_star = seg_counts[seg : seg + int(n_segs[i])]
                starts = np.zeros(len(per_star) + 1, dtype=np.int64)
                starts[1:] = np.cumsum(per_star)
                seg += int(n_segs[i])
            else:
                starts = np.array([0, int(count)], dtype=np.int64)
            offsets.append(starts)
            row += int(count)
        result[split] = (tics, blocks, offsets)
    return result


def cache_path(cell: str, seed: int, scope: str, root: Path | None = None, pool: str = "v1") -> Path:
    """
    Canonical location of one arm's bag cache; `untrained` is geometry-shared and lives at the root.
    The new-task pool reuses the caches new_task_extract already wrote for the exp05 arms, whose file
    names drop the `exp05_` prefix and carry no bag-scope suffix (they are first-segment only).
    """
    if pool == "new_task":
        if scope == "all": # built here by replaying the sequence npz; new_task_extract has no all-segment path
            base = root if root is not None else (repo_root / "experiments" / "mil_pooling" / "bag_cache_new_task")
            return base / f"{cell}_s{seed}_all.npz"
        base = root if root is not None else (repo_root / "experiments" / "new_task_exp05" / "mu_cache")
        if cell == "untrained":
            return base / "untrained.npz"
        stem = cell[len("exp05_"):] if cell.startswith("exp05_") else cell
        return base / f"{stem}_s{seed}.npz"
    base = root if root is not None else (repo_root / "experiments" / "mil_pooling" / "bag_cache")
    return base / f"{cell}_s{seed}_{scope}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="build bag caches for the MIL pooling sweep")
    parser.add_argument("--cells", nargs="+", default=["exp05_comb_fbwd_c1p0", "exp05_comb_off"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--scope", default="first", choices=["first", "all"])
    parser.add_argument("--ckpt", default="best_recon_aux")
    parser.add_argument("--variant", default="B")
    parser.add_argument("--untrained", action="store_true", default=True,
                        help="also build the capacity-matched untrained arm (geometry-shared, seed 0)")
    parser.add_argument("--packed", default=None, help="packed dir override (default: the first cell's own)")
    parser.add_argument("--pool", default="v1", choices=["v1", "new_task"],
                        help="new_task replays the sequence npz instead of slicing the packed memmap")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device {device}")
    first_run = repo_root / "experiments" / args.cells[0] / "models" / f"{args.variant}_seed{args.seeds[0]}"
    ckpt0 = torch.load(first_run / f"{args.ckpt}.pt", map_location=device, weights_only=False)
    cfg0 = ckpt0["cfg"]
    window = int(cfg0["data"]["window"])
    packed_dir = Path(args.packed) if args.packed else (repo_root / "experiments" / args.cells[0] / "packed")
    seq_dir = repo_root / "processed" / "sequences"
    pool_frame = None
    if args.pool == "new_task":
        pool_frame = pd.read_parquet(repo_root / "processed" / "subset" / "new_task_pool.parquet")

    pending = [] # (out_path, model) for every arm still missing a cache
    for cell in args.cells:
        for seed in args.seeds:
            out = cache_path(cell, seed, args.scope, pool=args.pool)
            if out.exists():
                log.info(f"{out.name}: exists, skipped")
                continue
            ckpt_path = repo_root / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            if not ckpt_path.exists():
                log.warning(f"{cell} seed{seed}: no {args.ckpt}.pt; skipped")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model, _ = build_model_from_ckpt(ckpt, device)
            pending.append((out, model))

    if args.untrained:
        out = cache_path("untrained", 0, args.scope, pool=args.pool)
        if out.exists():
            log.info(f"{out.name}: exists, skipped")
        else:
            model = _make_untrained(
                list(cfg0["model"]["enc_channels"]), int(cfg0["model"]["kernel_size"]),
                int(cfg0["model"]["z_dim"]), window,
                int(cfg0["model"]["gru_hidden"]), int(cfg0["model"]["gru_layers"]), device,
            )
            pending.append((out, model))

    if len(pending) == 0:
        log.info("every requested cache already exists")
        return 0
    if args.pool == "new_task":
        build_new_task_caches(pending, pool_frame, seq_dir, window, args.scope, device)
    else:
        for out, model in pending: # packed memmap reads are cheap, so per-arm is fine here
            build_cache(model, packed_dir, window, args.scope, out, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
