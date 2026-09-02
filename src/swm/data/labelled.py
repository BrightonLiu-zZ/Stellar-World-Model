"""Labelled star populations for the C1/C2 supervised baselines (roadmap D20 / Y13b).

The supervised arms must score on IDENTICAL splits and populations to the F1 fusion scorecard -- that
is the whole fairness protocol, and a re-derivation of the splits inside a training module would turn
it from a property of the code into a claim someone has to re-verify. So nothing here derives a
population: every star set, every split and every keep mask is taken from the SAME functions the probe
scorers call.

    v1 population    experiments/exp01_window256_seq16/packed, first segment per star, via
                     skyline.load_first_segment_blocks -- the bag scope of every table since exp01
                     (16-20 windows, 16.2 mean). Carries pulsating/eb/rotation/transit plus the two
                     ADR-0010 probes that ride the subset, rotation_period and ijspeert.
    pool population  processed/subset/new_task_pool.parquet replayed from processed/sequences/*.npz
                     via new_task_extract.{index_pool_npz, replay_first_segment} -- the exp01 pack
                     recipe (native 1024 -> absmax guard -> subdivide to 256). Carries osc_giant,
                     solar_like_osc, flare, numax_hon, rgb_vs_heb.

The two populations are disjoint star sets, which is the reason swm.train could not host this work:
its dataset reads one packed dir, and the pool has none.

WHY val IS USED HERE AND NOWHERE ELSE. Both populations carry a `val` split that no probe has ever
touched (v1 2,021 stars; pool 3,429). The supervised arms need a selection signal that is not the test
set, so they use it. This leaks nothing into any published probe number.

CACHE. Blocks are cached per (population, split) under the C1/C2 output tree, NOT beside the mu
caches: those key on `{arm}.npz` with no checkpoint in the key and short-circuit on exists(), and this
project has already been bitten by that once. A raw-flux cache has no arm in it at all, so it is
namespaced by population instead and the manifest's measured star counts are asserted against it.

Run (nothing to run directly; imported by swm.train.supervised and by the tests).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from swm.eval.new_task_extract import index_pool_npz, replay_first_segment
from swm.eval.new_task_scorecard import label_frame
from swm.eval.skyline import load_first_segment_blocks

log = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parents[3]
SPLITS = ("train", "val", "test")
V1_PACKED = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
POOL_PARQUET = repo_root / "processed" / "subset" / "new_task_pool.parquet"
SEQ_DIR = repo_root / "processed" / "sequences"
IJSPEERT_CSV = repo_root / "labels" / "external" / "ijspeert2024_bright.csv"
CANON_CSV = repo_root / "labels" / "variability_labels_star.csv"
# v1 binary columns as the probe reads them; missing tic --> 0, matching y_for()'s reindex-and-fill.
V1_COLUMNS = ("pulsating", "eb", "rotation", "transit")


class StarBags:
    """One split of one population: the first-segment windows of each star, kept grouped by star.

    Held as a flat (n_windows, window) array plus per-star counts rather than a list of blocks, so a
    batch of stars is a contiguous slice-and-gather instead of a Python loop, and so the whole
    population (~400 MB across both) sits in RAM for the whole run. `star_index` is what the trainer
    scatters window logits back onto to build the star-level bag mean.
    """

    def __init__(self, tics: np.ndarray, windows: np.ndarray, counts: np.ndarray) -> None:
        assert len(tics) == len(counts), "one count per star"
        assert int(counts.sum()) == len(windows), "counts must tile the flat window array"
        self.tics = tics
        self.windows = windows
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.tics)

    def gather(self, star_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Windows of the selected stars, plus the 0..len(star_rows)-1 bag id of each window.

        Returned as (windows, star_index) so the caller can forward every window in one pass and then
        reduce by bag. The bag ids are LOCAL to this batch, not global star rows.
        """
        pieces = []
        bag_ids = []
        for local, row in enumerate(star_rows):
            start = self.offsets[row]
            stop = self.offsets[row + 1]
            pieces.append(self.windows[start:stop])
            bag_ids.append(np.full(stop - start, local, dtype=np.int64))
        return np.concatenate(pieces, axis=0), np.concatenate(bag_ids, axis=0)

    def subset(self, keep: np.ndarray) -> StarBags:
        """Drop the stars a task's keep mask excludes, before any training touches the population."""
        rows = np.flatnonzero(keep)
        pieces = []
        for row in rows:
            pieces.append(self.windows[self.offsets[row]:self.offsets[row + 1]])
        return StarBags(self.tics[rows], np.concatenate(pieces, axis=0), self.counts[rows])


def _build_v1(split: str, window: int) -> StarBags:
    """v1 subset blocks straight from the packed memmap, in the probe's ascending-tic order."""
    tics, blocks = load_first_segment_blocks(V1_PACKED, split, window)
    counts = np.array([len(block) for block in blocks], dtype=np.int64)
    return StarBags(np.array(tics, dtype=np.int64), np.concatenate(blocks, axis=0), counts)


def _build_pool(split: str, window: int) -> StarBags:
    """Pool blocks replayed from the sequences npz, dropping stars whose guard leaves no windows.

    The drop is not a policy choice here: new_task_extract.load_pool_blocks drops them too, so keeping
    them would put stars in the supervised population that the probe never scored.
    """
    pool = pd.read_parquet(POOL_PARQUET)
    want = pool.loc[pool["split"] == split, "tic_id"].astype(int).sort_values().tolist()
    npz_index = index_pool_npz(SEQ_DIR, set(want))
    tics = []
    blocks = []
    counts = []
    for tic in tqdm(want, desc=f"replay pool npz[{split}]", total=len(want)):
        entry = npz_index.get(tic)
        assert entry is not None, f"pool TIC {tic} has no npz under {SEQ_DIR}"
        block = replay_first_segment(entry[2], window)
        if block.shape[0] == 0:
            continue
        tics.append(tic)
        blocks.append(block)
        counts.append(len(block))
    return StarBags(np.array(tics, dtype=np.int64), np.concatenate(blocks, axis=0),
                    np.array(counts, dtype=np.int64))


def load_bags(population: str, split: str, window: int, cache_dir: Path) -> StarBags:
    """First-segment star bags for one (population, split), backed by an npz cache.

    The pool branch is a scandir over processed/sequences plus one npz read per star, which is minutes
    the first time and seconds afterwards; the v1 branch is a memmap slice either way. Both are cached
    so the 66-run queue pays the cost once.
    """
    assert split in SPLITS, f"unknown split {split!r}"
    cache_path = Path(cache_dir) / f"{population}_{split}_w{window}.npz"
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        return StarBags(payload["tics"], payload["windows"], payload["counts"])
    if population == "v1":
        bags = _build_v1(split, window)
    elif population == "pool":
        bags = _build_pool(split, window)
    else:
        raise ValueError(f"unknown population {population!r}; expected 'v1' or 'pool'")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, tics=bags.tics, windows=bags.windows, counts=bags.counts)
    log.info(f"cached {population}/{split}: {len(bags)} stars, {len(bags.windows)} windows --> {cache_path}")
    return bags


def star_label_frame() -> pd.DataFrame:
    """One tic-indexed frame carrying every column the 11 tasks read.

    Built on top of the probe's own `label_frame()` (which already merges the new-task catalog with the
    canonical flare flag) so the menu columns cannot drift from what F1 scored, then extended with the
    four v1 variability columns, the TARS period, and an ijspeert indicator. `flare` is deliberately
    taken from label_frame's merge rather than re-read, for the same reason.
    """
    frame = label_frame()
    canon = pd.read_csv(CANON_CSV)
    canon["tic_id"] = canon["tic_id"].astype(int)
    for column in V1_COLUMNS:
        canon[column] = pd.to_numeric(canon[column], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    canon = canon.set_index("tic_id")
    for column in [*V1_COLUMNS, "rotation_period"]:
        frame[column] = canon[column].reindex(frame.index)
    ijspeert = set(pd.read_csv(IJSPEERT_CSV)["TIC"].astype(int))
    frame["ijspeert"] = frame.index.isin(ijspeert).astype(int)
    return frame


def task_targets(task: dict, tics: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Target vector and keep mask for one task on one split's stars, in the probe's own terms.

    Returns (y, keep) both aligned to `tics`; the caller subsets the bags with `keep` before training.
    The three keep kinds are exactly the three the F1 scorers apply:
      all                 every star in the split (detection flags reindex to 0 for absent tics)
      labelled            the catalog-only population -- score_regression_task / score_contrastive
      tars_rotator_le5d   score_rotation_period_from_mu's mask: rotation == 1, period present, <= 5 d
    """
    column = task["label"]["column"]
    keep_kind = task["keep"]
    assert column in labels.columns, f"task {task['name']}: no column {column!r} in the label frame"
    series = labels[column].reindex(tics)

    if keep_kind == "all":
        keep = np.ones(len(tics), dtype=bool)
        y = series.fillna(0).to_numpy(dtype=float)
    elif keep_kind == "labelled":
        keep = series.notna().to_numpy()
        y = series.to_numpy(dtype=float)
    elif keep_kind == "tars_rotator_le5d":
        rotation = labels["rotation"].reindex(tics).fillna(0)
        keep = (series.notna() & (series <= 5) & (rotation == 1)).to_numpy()
        y = series.to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown keep kind {keep_kind!r}")

    if task.get("target_transform") == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log10(y)
    y = np.where(keep, y, 0.0)
    return y, keep
