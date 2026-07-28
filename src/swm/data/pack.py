"""Stage 1 prep: pack the subset's NaN-free windows into one flat float32 memmap per split.

Reading hundreds of thousands of tiny .npz files per epoch is slow locally and catastrophic on
Lustre later, so each split is streamed into a single raw float32 file ({split}_windows.dat,
shape [total_windows, window]) with a parquet index mapping every segment to its row range. A
pack-time guard drops any window whose max|flux| exceeds data.max_absmax (defense in depth on top
of the notebook source cleanup); a segment with fewer than seq_len survivors is skipped entirely.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.data.pack
    python -m swm.data.pack data.limit=50        # smoke: at most 50 segments per split
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from swm.data.sequence_files import scan_sequence_files

log = logging.getLogger(__name__)


def load_and_filter(path: str, window: int, seq_len: int, max_absmax: float) -> tuple[np.ndarray | None, int, int]:
    """
    Read one segment's .npz, drop windows exceeding the flux-absmax guard, and flatten survivors.
    Returns (survivor_block or None, dropped_window_count, dropped_segment_flag); the block is None
    when fewer than seq_len windows survive. This is the I/O-bound step run across threads.

    When cfg.data.window is smaller than the STORED window length (Path A window-shrink ablation, e.g.
    256 vs the native 1024), each stored window is split into native//window contiguous sub-windows.
    The absmax guard is applied at the STORED granularity BEFORE splitting, so segment/window survival
    stays identical to the un-subdivided baseline (dropping a whole original window, never a partial one,
    keeps the no-cross-gap invariant that consecutive packed rows are time-adjacent). Per-segment MAD
    normalization (upstream in build_sequences_bulk) makes this split identical to a full re-window.

    When cfg.data.window is LARGER than the stored length (exp06 window-merge, e.g. 2048 vs 1024),
    adjacent stored windows are concatenated in fixed chunks of window//native (offsets 0..k-1, k..2k-1,
    ...). Stored windows within a segment are consecutive stride=native slices, so a chunk is genuinely
    contiguous in time. A chunk is kept only if EVERY stored member passes the absmax guard - a merged
    window must never span a dropped stored window (same no-cross-gap invariant as above). A tail of
    fewer than k stored windows is discarded. dropped_windows still counts absmax-dropped STORED windows
    only (comparable across geometries); partner/tail losses are not absmax drops.
    """
    data = np.load(path)
    windows = data["windows"] # (N, native, 1); native = stored window length (1024 for the v1 corpus)
    native = windows.shape[1]
    absmax = np.abs(windows).max(axis=(1, 2)) # (N,) per-window peak |flux|, at the stored granularity
    keep = absmax <= max_absmax
    dropped_windows = int((~keep).sum()) # counts dropped STORED windows (comparable to the 1024 baseline)
    if window < native:
        assert native % window == 0, f"cfg.data.window={window} must evenly divide stored window {native}"
        k = native // window
        survivors = windows[keep] # (M, native, 1)
        survivors = survivors.reshape(survivors.shape[0] * k, window, 1) # (M*k, window, 1), time order preserved
    elif window > native:
        assert window % native == 0, f"cfg.data.window={window} must be a multiple of stored window {native}"
        k = window // native
        n_chunks = windows.shape[0] // k # fixed-offset chunks; ragged tail discarded
        if n_chunks == 0:
            return None, dropped_windows, 1
        chunk_keep = keep[: n_chunks * k].reshape(n_chunks, k).all(axis=1) # (n_chunks,) all members must pass
        chunks = windows[: n_chunks * k].reshape(n_chunks, window, 1) # (n_chunks, k*native, 1), time-contiguous
        survivors = chunks[chunk_keep]
    else:
        survivors = windows[keep] # (M, native, 1)
    if survivors.shape[0] < seq_len:
        return None, dropped_windows, 1
    flat = np.ascontiguousarray(survivors.reshape(survivors.shape[0], window), dtype=np.float32)
    return flat, dropped_windows, 0


def pack_split(
    split_files: pd.DataFrame, dat_path: Path, window: int, seq_len: int, max_absmax: float, workers: int
) -> tuple[pd.DataFrame, int, int]:
    """
    Stream one split's surviving windows into a raw float32 file and build its row index.
    Threads prefetch+filter the .npz files (I/O bound) while a single consumer writes them in input
    order, so the row ranges recorded in the index stay deterministic. Returns the index frame plus
    dropped-window and dropped-segment counts for the manifest.
    """
    records = list(split_files.itertuples(index=False))
    index_rows = []
    row_cursor = 0
    dropped_windows = 0
    dropped_segments = 0
    with open(dat_path, "wb") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda r: load_and_filter(r.path, window, seq_len, max_absmax), records)
            for record, (flat, dropped_w, dropped_s) in zip(records, results): # map preserves input order
                dropped_windows += dropped_w
                dropped_segments += dropped_s
                if flat is None:
                    continue
                fout.write(flat.tobytes())
                n_win = flat.shape[0]
                index_rows.append(
                    {
                        "seg_id": f"TIC{record.tic_id}_s{record.sector}_seg{record.seg_idx}",
                        "tic_id": int(record.tic_id),
                        "sector": int(record.sector),
                        "seg_idx": int(record.seg_idx),
                        "row_start": row_cursor,
                        "n_win": n_win,
                    }
                )
                row_cursor += n_win
    index = pd.DataFrame(index_rows)
    return index, dropped_windows, dropped_segments


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    split_path = Path(cfg.paths.subset_dir) / "split.parquet"
    assert split_path.exists(), f"run swm.data.subset first; missing {split_path}"
    split = pd.read_parquet(split_path)
    tic_to_split = dict(zip(split["tic_id"].tolist(), split["split"].tolist()))

    files = scan_sequence_files(cfg.paths.sequences_dir)
    files = files[files["tic_id"].isin(tic_to_split.keys())].copy()
    files["split"] = files["tic_id"].map(tic_to_split)
    files = files.sort_values(["split", "tic_id", "sector", "seg_idx"]).reset_index(drop=True)

    out_dir = Path(cfg.paths.packed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = int(cfg.data.window)
    seq_len = int(cfg.data.seq_len)
    max_absmax = float(cfg.data.max_absmax)

    manifest: dict[str, object] = {
        "window": window,
        "seq_len": seq_len,
        "max_absmax": max_absmax,
        "dtype": "float32",
        "splits": {},
    }
    for split_name in ["train", "val", "test"]:
        split_files = files[files["split"] == split_name]
        if cfg.data.limit is not None:
            split_files = split_files.head(int(cfg.data.limit)) # smoke cap
        dat_path = out_dir / f"{split_name}_windows.dat"
        index, dropped_w, dropped_s = pack_split(
            split_files, dat_path, window, seq_len, max_absmax, int(cfg.data.pack_workers)
        )
        index.to_parquet(out_dir / f"{split_name}_index.parquet", index=False)
        total_windows = int(index["n_win"].sum()) if len(index) else 0
        manifest["splits"][split_name] = {
            "segments_in": int(len(split_files)),
            "segments_kept": int(len(index)),
            "segments_dropped": int(dropped_s),
            "windows_dropped_absmax": int(dropped_w),
            "total_windows": total_windows,
            "stars": int(index["tic_id"].nunique()) if len(index) else 0,
        }
        log.info(f"[{split_name}] {manifest['splits'][split_name]}")

    with open(out_dir / "pack_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"wrote memmaps + indices + manifest to {out_dir}")


if __name__ == "__main__":
    main()
