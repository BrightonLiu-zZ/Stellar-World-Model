"""exp10 prerequisite 1: the per-star engineered-feature table that E1/E2 read during training.

exp10's two content-creation cells need each star's 25 engineered features INSIDE the training loop:
`exp10_cond_dec` concatenates them to z before decoding, `exp10_decorr` penalises their correlation
with mu. Both need one standardized vector per TIC in the v1 packed subset, available at dataset
construction time -- so the F1 scorecard's feature source is materialised once, here, into
experiments/exp10_features/subset_features25.parquet.

Provenance (manifest D-E10.10): train and test rows come from `cached_subset_features`, the SAME
parquet cache the F1 fusion scorecard reads, so a feature the model trains against and a feature the
scorecard fuses with are the same number. The cache covers only train+test (the scorecard never scores
val), so val is computed here through the identical code path -- `load_first_segment_blocks` +
`extract_features` on the concatenated first segment.

Standardization constants (per-feature mean/sd) come from the TRAIN split ONLY and are stored as
parquet metadata; val/test are standardized with those same constants. A subset star with no packed
first segment gets the standardized zero vector (i.e. the train mean, contributing nothing) and is
COUNTED -- the count prints here and is recorded in the exp10 README.

Run (swm env, from repo root, PYTHONPATH=src; CPU only, no GPU, no checkpoints):
    PYTHONUNBUFFERED=1 python experiments/exp10_build_features.py
    python experiments/exp10_build_features.py --force        # ignore an existing output file
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from swm.eval.features import FEATURE_NAMES, extract_features  # noqa: E402
from swm.eval.new_task_ceiling import cached_subset_features  # noqa: E402
from swm.eval.skyline import load_first_segment_blocks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp10_features")

window = 256 # exp01 geometry; the pack every exp10 cell trains on
packed_dir = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
split_path = repo_root / "processed" / "subset" / "split.parquet"
out_path = repo_root / "experiments" / "exp10_features" / "subset_features25.parquet"


def raw_feature_frame() -> pd.DataFrame:
    """
    Assemble the UNstandardized (tic_id, split, 25 features) table over all three packed splits.
    Train and test are lifted straight out of the F1 scorecard's cache so the two uses of "the 25
    features" cannot drift apart; val has no cache entry (the scorecard never scores it) and is
    recomputed through the same extractor on the same first-segment blocks.
    """
    cached = cached_subset_features(packed_dir) # {split: (tics, 1-row feature blocks)}; F1's own source
    frames = []
    for split in ["train", "test"]:
        tics, blocks = cached[split]
        values = np.concatenate(blocks, axis=0)
        frame = pd.DataFrame(values, columns=FEATURE_NAMES)
        frame.insert(0, "split", split)
        frame.insert(0, "tic_id", np.asarray(tics, dtype=np.int64))
        frames.append(frame)

    tics, blocks = load_first_segment_blocks(packed_dir, "val", window)
    values = np.zeros((len(tics), len(FEATURE_NAMES)), dtype=np.float64)
    for i in tqdm(range(len(blocks)), desc="feats[val]", total=len(blocks)):
        values[i] = extract_features(blocks[i].reshape(-1)) # first-segment windows concatenated, as F1 does
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    frame.insert(0, "split", "val")
    frame.insert(0, "tic_id", np.asarray(tics, dtype=np.int64))
    frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    assert out["tic_id"].is_unique, "a TIC appears in more than one split of the packed subset"
    assert np.isfinite(out[FEATURE_NAMES].to_numpy()).all(), "non-finite engineered feature (extractor contract)"
    return out


def standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """
    Centre and scale every split with TRAIN-split constants, the no-leakage rule the probes already use.
    A feature with zero train variance would divide by zero, so its sd is replaced by 1.0 and the column
    ends up identically zero -- visible in the metadata rather than silently NaN.
    """
    train_rows = frame[frame["split"] == "train"]
    assert len(train_rows) > 0, "no train rows; cannot derive standardization constants"
    means = train_rows[FEATURE_NAMES].to_numpy(dtype=np.float64).mean(axis=0)
    sds = train_rows[FEATURE_NAMES].to_numpy(dtype=np.float64).std(axis=0, ddof=0)
    degenerate = []
    for i, name in enumerate(FEATURE_NAMES):
        if sds[i] <= 0.0:
            sds[i] = 1.0
            degenerate.append(name)
    if degenerate:
        log.warning(f"zero train variance, sd forced to 1.0: {degenerate}")
    out = frame.copy()
    out[FEATURE_NAMES] = (frame[FEATURE_NAMES].to_numpy(dtype=np.float64) - means) / sds
    constants = {"mean": means.tolist(), "sd": sds.tolist(), "degenerate": degenerate}
    return out, constants


def add_missing_stars(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Give every subset TIC a row, filling stars absent from the pack with the standardized zero vector.
    A star can be in the split table but not the pack (its segments lost every window to the absmax
    guard, or it held fewer than seq_len survivors). Zero is the train mean in standardized space, so
    such a star contributes no conditioning signal and no decorrelation pressure; D-E10.10 requires the
    count to be visible, hence the `feats_missing` flag and the coverage table returned alongside.
    """
    split = pd.read_parquet(split_path)
    split["tic_id"] = split["tic_id"].astype(np.int64)
    have = set(frame["tic_id"].tolist())
    missing_rows = []
    for row in split.itertuples(index=False):
        if int(row.tic_id) in have:
            continue
        record = {"tic_id": int(row.tic_id), "split": row.split, "feats_missing": True}
        for name in FEATURE_NAMES:
            record[name] = 0.0
        missing_rows.append(record)

    frame = frame.copy()
    frame["feats_missing"] = False
    if missing_rows:
        frame = pd.concat([frame, pd.DataFrame(missing_rows)], ignore_index=True)
    frame = frame[["tic_id", "split", "feats_missing", *FEATURE_NAMES]]
    frame = frame.sort_values(["split", "tic_id"]).reset_index(drop=True)

    coverage = frame.groupby("split").agg(stars=("tic_id", "size"), missing=("feats_missing", "sum"))
    coverage["covered"] = coverage["stars"] - coverage["missing"]
    return frame, coverage.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the exp10 per-star standardized 25-feature table.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output parquet.")
    args = ap.parse_args()

    if out_path.exists() and not args.force:
        log.error(f"{out_path} exists; pass --force to rebuild (the training cells read this file)")
        return 1

    raw = raw_feature_frame()
    standardized, constants = standardize(raw)
    frame, coverage = add_missing_stars(standardized)

    log.info(f"coverage by split:\n{coverage.to_string(index=False)}")
    n_missing = int(frame["feats_missing"].sum())
    log.info(f"{len(frame)} subset stars, {n_missing} with no packed first segment (standardized zero vector)")

    payload = {
        "feature_names": FEATURE_NAMES,
        "standardization": "train-split mean/sd of processed/subset, applied to every split",
        "mean": constants["mean"],
        "sd": constants["sd"],
        "degenerate_features": constants["degenerate"],
        "source": "swm.eval.new_task_ceiling.cached_subset_features (train/test) + the same extractor on val",
        "packed_dir": packed_dir.relative_to(repo_root).as_posix(),
        "window": window,
        "n_missing": n_missing,
        "coverage": coverage.to_dict(orient="records"),
        "manifest": "experiments/configs/exp10_fusion_spine.yaml (D-E10.10)",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"exp10_features"] = json.dumps(payload).encode("utf-8") # constants travel with the data
    table = table.replace_schema_metadata(metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    log.info(f"wrote {out_path} ({len(frame)} rows x {len(FEATURE_NAMES)} features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
