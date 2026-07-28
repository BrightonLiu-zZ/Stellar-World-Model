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
    """

    def __init__(self, packed_dir: str | Path, split: str, seq_len: int, window: int, randomize: bool) -> None:
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

    def __getitem__(self, i: int) -> torch.Tensor:
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
        return x
