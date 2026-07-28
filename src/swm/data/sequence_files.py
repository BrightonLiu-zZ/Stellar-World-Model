from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

_FNAME_RE = re.compile(r"TIC(\d+)_s(\d+)_seg(\d+)_run(\d+)\.npz")


def scan_sequence_files(sequences_dir: str | Path) -> pd.DataFrame:
    """
    Enumerate every per-segment .npz in the sequences directory.
    Returns one row per file (path, tic_id, sector, seg_idx, run_idx); this is the corpus
    index that both the subset selector and the packer build on. Files whose names do not
    match the canonical TIC<id>_s<sector>_seg<idx>_run<idx>.npz pattern are skipped.
    """
    seq_dir = Path(sequences_dir)
    assert seq_dir.is_dir(), f"sequences dir not found: {seq_dir}"
    rows = []
    with os.scandir(seq_dir) as it:
        for entry in it:
            if not entry.name.endswith(".npz"):
                continue
            m = _FNAME_RE.match(entry.name)
            if m is None:
                continue
            rows.append((entry.path, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    df = pd.DataFrame(rows, columns=["path", "tic_id", "sector", "seg_idx", "run_idx"])
    assert len(df) > 0, f"no sequence .npz files found in {seq_dir}"
    return df
