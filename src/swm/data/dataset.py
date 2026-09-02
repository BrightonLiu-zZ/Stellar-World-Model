from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeqWindowDataset(Dataset):
    """
    DataLoader requires __len__ + __getitem__; one class shared across the train and val splits.
    Each item is a length-seq_len sequence of consecutive windows drawn from a single segment,
    read out of the packed flat float32 memmap by the segment's row range. Training randomizes the
    start index within the segment each epoch (the model sees different sub-sequences over time);
    validation takes a fixed start so the monitoring signal is deterministic. Sequences never cross
    a segment boundary (each segment is one contiguous row block) and are never padded.

    features_path (exp10) attaches the star's standardized 25-feature vector to every item, joined on
    the segment's tic_id, so the loop can hand it to the conditioned decoder (E1) or the decorrelation
    penalty (E2). Items then come back as (x, feats) instead of x. The default None returns bare
    tensors, which is the exp00-09 path unchanged.
    """

    def __init__(self, packed_dir: str | Path, split: str, seq_len: int, window: int, randomize: bool,
                 features_path: str | Path | None = None) -> None:
        packed = Path(packed_dir)
        index_path = packed / f"{split}_index.parquet"
        dat_path = packed / f"{split}_windows.dat"
        assert index_path.exists(), f"missing {index_path}; run swm.data.pack"
        assert dat_path.exists(), f"missing {dat_path}; run swm.data.pack"
        self.index = pd.read_parquet(index_path).reset_index(drop=True)
        self.dat_path = dat_path
        self.total_rows = int(self.index["n_win"].sum())
        self.seq_len = seq_len
        self.window = window
        self.randomize = randomize
        self._windows: np.memmap | None = None # opened lazily, once per DataLoader worker
        self.features: np.ndarray | None = None # (n_segments, n_feat), one row per index row
        self.n_missing_features = 0
        if features_path is not None:
            self.features, self.n_missing_features = self._join_features(Path(features_path))

    def _join_features(self, features_path: Path) -> tuple[np.ndarray, int]:
        """
        Materialize a per-SEGMENT feature matrix by looking each segment's star up in the exp10 table.
        Done once at construction (the table is ~13k rows) so __getitem__ stays a memmap slice plus an
        array index. A star absent from the table would silently train against zeros, so the count is
        kept and the caller logs it; the builder already guarantees every subset TIC has a row, and this
        is the second line of defence rather than the first.
        """
        assert features_path.exists(), f"missing {features_path}; run experiments/exp10_build_features.py"
        table = pd.read_parquet(features_path)
        assert table["tic_id"].is_unique, f"{features_path} has duplicate tic_id rows"
        feature_cols = []
        for column in table.columns:
            if column not in ("tic_id", "split", "feats_missing"):
                feature_cols.append(column)
        values = table[feature_cols].to_numpy(dtype=np.float32)
        row_of_tic = {}
        for position, tic in enumerate(table["tic_id"].astype(int).tolist()):
            row_of_tic[tic] = position
        out = np.zeros((len(self.index), len(feature_cols)), dtype=np.float32)
        n_missing = 0
        for i, tic in enumerate(self.index["tic_id"].astype(int).tolist()):
            position = row_of_tic.get(tic)
            if position is None:
                n_missing += 1 # standardized zero vector = the train mean; contributes nothing
                continue
            out[i] = values[position]
        return out, n_missing

    def _mm(self) -> np.memmap:
        """
        Open the memmap on first access inside the current process.
        Lazy opening keeps the Dataset picklable to Windows spawn workers (only the path and
        shape cross the process boundary) and gives each worker its own file handle.
        """
        if self._windows is None:
            self._windows = np.memmap(self.dat_path, dtype=np.float32, mode="r", shape=(self.total_rows, self.window))
        return self._windows

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        row = self.index.iloc[i]
        start = int(row["row_start"])
        n_win = int(row["n_win"])
        max_offset = n_win - self.seq_len # >= 0, guaranteed by the packer
        if self.randomize and max_offset > 0:
            offset = random.randint(0, max_offset)
        else:
            offset = 0
        block = self._mm()[start + offset : start + offset + self.seq_len] # (seq_len, window)
        x = torch.from_numpy(np.array(block, dtype=np.float32)).unsqueeze(-1) # (seq_len, window, 1); copy -> writable
        if self.features is None:
            return x
        return x, torch.from_numpy(self.features[i].copy()) # (n_feat,) this segment's star
