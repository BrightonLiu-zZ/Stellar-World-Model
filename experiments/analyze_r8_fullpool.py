"""R8 (roadmap 2026-08-11) - rescore the four v1 tasks at survey prevalence.

Every published v1 absolute is measured on a population of all positives plus 10,000 of the 102,008
quiet stars, so eb sits at 9.1% of the test split against a corpus rate of 1.07%. The claims matrix
discloses this ("gaps valid, absolutes not"); this script converts the disclosure into a measured row.
Only absolutes move: the probe is refit on nothing, the train split is untouched, and every paired
delta in the paper is prevalence-invariant by construction.

Two populations are reported because "survey prevalence" has two defensible readings and they answer
different objections:

  quiet    Negatives drawn from catalogue-clean stars only, the existing protocol with the base rate
           corrected. Isolates the prevalence effect, but a real survey's negatives contain rotators
           and pulsators, so this pool is unrepresentative in the easy direction.
  survey   Negatives = every corpus star that is not a positive for THIS task, including the 14,927
           rotation positives and the other classes. This is what the paper sentence claims. The
           difference survey - quiet is the confusable-negatives cost, reported separately rather
           than left confounded with prevalence inside one drop.

Each population is scored twice: `full` uses every available negative (deterministic, and for the
survey pool it lands BELOW the corpus rate because only 15% of positives sit in test while 100% of
the negatives do), and `matched` subsamples negatives to hit the corpus rate exactly, repeated over
`--draws` draws so the row carries a negative-draw spread alongside the seed spread (F17: both sides).

Fault isolation is deliberate here and overrides the usual assert-instead-of-except style: the run is
launched unattended, so a shard that fails to read or an arm whose checkpoint is missing is logged and
skipped rather than allowed to kill the queue. Every stage is resumable - shards and arms are
skip-if-present, and the score stage skips arms already in the output CSV.

Run (repo root, swm env, PYTHONPATH=src):
    PYTHONUNBUFFERED=1 python experiments/analyze_r8_fullpool.py 2>&1 | tee experiments/r8_fullpool/run.log
    python experiments/analyze_r8_fullpool.py --stages pool check --limit-added 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from swm.data.sequence_files import scan_sequence_files
from swm.eval.features import extract_features
from swm.eval.new_task_extract import _NPZ_RE, replay_first_segment
from swm.eval.readout_sweep import build_model_from_ckpt
from swm.eval.skyline import _make_untrained, load_first_segment_blocks, logistic_scores

log = logging.getLogger("r8")
root = Path(__file__).resolve().parents[1]

out_dir = root / "experiments" / "r8_fullpool"
mu_cache_dir = out_dir / "mu_cache"
seq_dir = root / "processed" / "sequences"
packed = root / "experiments" / "exp01_window256_seq16" / "packed"
exp07_cache = root / "experiments" / "exp07_forensics" / "mu_cache"
subset_features = root / "experiments" / "exp06_features_cache.parquet"

tasks = ("transit", "eb", "pulsating", "rotation")
label_cols = ["transit", "eb", "pulsating", "rotation", "flare_ever"] # `quiet` = zero in all five
amplitude_cols = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"] # the mean_resid basis, periodicity-free

window = 256
shard_size = 5000 # ~205 MB of flux per shard in RAM, and a crash costs at most one shard
fbwd_cell = "exp07_hann0p3_fbwd"
off_cell = "exp07_hann0p3_off"
all_seeds = [0, 1, 2, 3, 4, 5]

# Projection thresholds from the R8a go/no-go (grilled 2026-08-11). Wall-clock hours for the whole
# extraction, arms included; the first band that fits wins.
arm_set_bands = [("full", 3.0), ("fbwd6", 8.0), ("fbwd3", 16.0)]


def arm_table(arm_set: str) -> pd.DataFrame:
    """Expand a sizing decision into the concrete (arm, cell, seed) jobs the extract stage runs."""
    rows = []
    if arm_set == "fbwd3":
        seeds = [0, 1, 2]
    else:
        seeds = all_seeds
    for seed in seeds:
        rows.append({"arm": f"{fbwd_cell}_s{seed}", "cell": fbwd_cell, "seed": seed, "kind": "trained"})
    if arm_set == "full":
        for seed in all_seeds:
            rows.append({"arm": f"{off_cell}_s{seed}", "cell": off_cell, "seed": seed, "kind": "trained"})
        for seed in all_seeds:
            rows.append({"arm": f"untrained_i{seed}", "cell": "untrained", "seed": seed, "kind": "untrained"})
    return pd.DataFrame(rows)


def ckpt_path_of(cell: str, seed: int) -> Path:
    """Locate one run's evaluation checkpoint; `best_recon_aux` is the selection pinned by test_dual_checkpoint."""
    return root / "experiments" / cell / "models" / f"B_seed{seed}" / "best_recon_aux.pt"


def load_arm_model(cell: str, seed: int, device: str) -> torch.nn.Module:
    """Rebuild one arm's encoder, either from its checkpoint or as a random init at the given seed."""
    reference = torch.load(ckpt_path_of(fbwd_cell, 0), map_location="cpu", weights_only=False)
    if cell == "untrained":
        mc = reference["cfg"]["model"]
        return _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                               window, int(mc["gru_hidden"]), int(mc["gru_layers"]), device, seed=seed)
    ckpt = torch.load(ckpt_path_of(cell, seed), map_location="cpu", weights_only=False)
    model, _ = build_model_from_ckpt(ckpt, device)
    return model


# ----------------------------------------------------------------------------------------------------
# stage: pool - who is a candidate negative, and who is fenced off
# ----------------------------------------------------------------------------------------------------
def build_pool() -> pd.DataFrame:
    """
    Assign every corpus star with windows to its role in the rescore.
    Subset train stars fitted the probe and subset val stars define the selection fold, so both are
    fenced off; subset test stars keep their existing mu from the exp07 cache and are not re-encoded.
    Everything else is an `added` candidate, and which of those count as negatives is decided per task
    at score time (quiet pool vs survey pool).
    """
    files = scan_sequence_files(str(seq_dir))
    present = set(files["tic_id"].unique().tolist())
    labels = pd.read_csv(root / "labels" / "variability_labels_star.csv")
    for col in label_cols:
        labels[col] = labels[col].fillna(0).astype(int)
    pool = labels[labels["tic_id"].isin(present)][["tic_id", *label_cols]].copy()
    pool["is_quiet"] = pool[label_cols].sum(axis=1) == 0

    subset = pd.read_parquet(root / "processed" / "subset" / "subset_tics.parquet")
    split_of = dict(zip(subset["tic_id"].tolist(), subset["split"].tolist()))
    roles = []
    for tic in pool["tic_id"].tolist():
        split = split_of.get(tic)
        if split is None:
            roles.append("added")
        elif split == "test":
            roles.append("existing_test")
        else:
            roles.append(f"excluded_{split}")
    pool["role"] = roles
    return pool.sort_values("tic_id").reset_index(drop=True)


def index_npz(want: set[int], index_path: Path) -> pd.DataFrame:
    """
    One scandir over the ~400k-file sequences dir --> the first segment (min sector, seg) of each wanted star.
    Cached to parquet because the scan is the single largest fixed cost of the extraction and a resumed
    run must not pay it again. The cache path is explicit so the check stage's scan, which wants a
    different star set, cannot overwrite the extraction's index.
    """
    if index_path.exists():
        return pd.read_parquet(index_path)
    best = {}
    started = time.time()
    with os.scandir(seq_dir) as it:
        for entry in it:
            match = _NPZ_RE.match(entry.name)
            if match is None:
                continue
            tic = int(match.group(1))
            if tic not in want:
                continue
            key = (int(match.group(2)), int(match.group(3)))
            if tic not in best or key < (best[tic][0], best[tic][1]):
                best[tic] = (key[0], key[1], entry.path)
    rows = []
    for tic in sorted(best):
        sector, seg, path = best[tic]
        rows.append({"tic_id": tic, "sector": sector, "seg_idx": seg, "path": path})
    frame = pd.DataFrame(rows)
    frame.to_parquet(index_path, index=False)
    log.info(f"scandir indexed {len(frame)} stars in {time.time() - started} s --> {index_path}")
    return frame


# ----------------------------------------------------------------------------------------------------
# stage: check - is a replayed star the same star as a packed star?
# ----------------------------------------------------------------------------------------------------
def run_check(device: str) -> None:
    """
    Compare the npz-replay path against the packed path on stars that exist in both.
    The added negatives are replayed from sequences npz while the existing test stars come from the
    packed memmap, so a systematic difference between the two readers would show up as a fake
    prevalence effect. This measures it on subset test stars before anything is trusted.
    """
    tics, blocks = load_first_segment_blocks(packed, "test", window)
    index = index_npz(set(tics[:400]), out_dir / "check_npz_index.parquet") # own cache: not the extraction index
    by_tic = dict(zip(index["tic_id"].tolist(), index["path"].tolist()))
    model = load_arm_model(fbwd_cell, 0, device)
    feature_cache = pd.read_parquet(subset_features).drop_duplicates("tic_id").set_index("tic_id")

    rows = []
    checked = 0
    for tic, packed_block in tqdm(list(zip(tics, blocks)), desc="check replay vs packed", total=len(tics)):
        if checked >= 200: # 200 stars is already far tighter than any effect this could hide
            break
        path = by_tic.get(tic)
        if path is None:
            continue
        replayed = replay_first_segment(Path(path), window) # (n_win, window)
        if replayed.shape != packed_block.shape:
            rows.append({"tic_id": tic, "shape_match": False, "max_abs_flux_diff": np.nan,
                         "max_abs_mu_diff": np.nan, "max_abs_feature_diff": np.nan})
            checked += 1
            continue
        flux_diff = float(np.abs(replayed - packed_block).max())
        with torch.no_grad():
            mu_a = model.encoder(torch.from_numpy(replayed).unsqueeze(-1).to(device))[0].float().cpu().numpy()
            mu_b = model.encoder(torch.from_numpy(packed_block).unsqueeze(-1).to(device))[0].float().cpu().numpy()
        feature_diff = np.nan
        if tic in feature_cache.index:
            fresh = extract_features(replayed.reshape(-1))[-4:] # the four amplitude cols, in FEATURE_NAMES order
            stored = feature_cache.loc[tic, amplitude_cols].to_numpy(dtype=float)
            feature_diff = float(np.abs(fresh - stored).max())
        rows.append({"tic_id": tic, "shape_match": True, "max_abs_flux_diff": flux_diff,
                     "max_abs_mu_diff": float(np.abs(mu_a.mean(axis=0) - mu_b.mean(axis=0)).max()),
                     "max_abs_feature_diff": feature_diff})
        checked += 1
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "replay_vs_packed_check.csv", index=False)
    agree = bool(frame["shape_match"].all()) and float(frame["max_abs_flux_diff"].max()) < 1e-5
    log.info(f"replay-vs-packed on {len(frame)} stars: shapes all match={bool(frame['shape_match'].all())}, "
             f"max flux diff={frame['max_abs_flux_diff'].max()}, max mu diff={frame['max_abs_mu_diff'].max()}, "
             f"max amplitude-feature diff={frame['max_abs_feature_diff'].max()}")
    if not agree:
        log.warning("REPLAY PATH DISAGREES WITH THE PACKED PATH - the added negatives are not read the "
                    "same way as the existing test stars; treat every R8 absolute as suspect")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------
# stage: extract - per-star pooled mu and amplitude features for the added stars
# ----------------------------------------------------------------------------------------------------
def shard_bounds(n: int) -> list[tuple[int, int]]:
    """Split the added-star list into fixed-size contiguous shards, the unit of resume."""
    bounds = []
    start = 0
    while start < n:
        bounds.append((start, min(start + shard_size, n)))
        start += shard_size
    return bounds


def read_shard(index: pd.DataFrame, lo: int, hi: int) -> tuple[list[int], list[np.ndarray]]:
    """Replay one shard's first-segment windows; stars whose windows are all cut by the absmax guard drop out."""
    tics = []
    blocks = []
    rows = index.iloc[lo:hi]
    for row in rows.itertuples(index=False):
        try:
            block = replay_first_segment(Path(row.path), window) # (n_win, window)
        except Exception:
            log.warning(f"unreadable npz for TIC {row.tic_id}, skipped\n{traceback.format_exc()}")
            continue
        if block.shape[0] == 0:
            continue
        tics.append(int(row.tic_id))
        blocks.append(block)
    return tics, blocks


@torch.no_grad() # no gradients at eval time, halves the memory of a full-shard encode
def pool_shard_mu(model: torch.nn.Module, blocks: list[np.ndarray], device: str,
                  batch: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode one shard's windows in flat batches and reduce each star's bag to (mean, std, n_windows).
    Pooling at extraction is what keeps the cache at ~94 MB per arm instead of ~1.9 GB: nothing in the
    R8 deliverable reads per-window mu, since MIL is decided out of the frozen tables (roadmap 9).
    """
    counts = []
    for block in blocks:
        counts.append(block.shape[0])
    flat = np.concatenate(blocks, axis=0) # (total_windows, window)
    chunks = []
    for start in range(0, flat.shape[0], batch):
        x = torch.from_numpy(flat[start : start + batch]).unsqueeze(-1).to(device) # (b, window, 1)
        mu, _ = model.encoder(x) # (b, z)
        chunks.append(mu.float().cpu().numpy())
    mu_all = np.concatenate(chunks, axis=0)
    means = []
    stds = []
    start = 0
    for n in counts:
        block_mu = mu_all[start : start + n] # (n_win, z)
        means.append(block_mu.mean(axis=0))
        stds.append(block_mu.std(axis=0))
        start += n
    return (np.stack(means).astype(np.float32), np.stack(stds).astype(np.float32),
            np.asarray(counts, dtype=np.int64))


def shard_features(tics: list[int], blocks: list[np.ndarray], path: Path) -> None:
    """Engineered amplitude features for one shard, computed during the replay so the npz I/O is paid once."""
    rows = []
    for tic, block in zip(tics, blocks):
        vector = extract_features(block.reshape(-1)) # first-segment concatenated flux, skyline protocol
        row = {"tic_id": tic}
        for name, value in zip(amplitude_cols, vector[-4:]):
            row[name] = float(value)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def run_extract(index: pd.DataFrame, arms: pd.DataFrame, device: str) -> None:
    """
    Fill the shard caches for every arm, skipping work that already exists.
    Models are held for the whole pass because the replay, not the encode, is the expensive half: a
    shard's flux is read once and pushed through every arm before it leaves RAM.
    """
    models = {}
    for row in arms.itertuples(index=False):
        try:
            models[row.arm] = load_arm_model(row.cell, row.seed, device)
        except Exception:
            log.error(f"arm {row.arm} failed to load, dropped from this run\n{traceback.format_exc()}")
    log.info(f"loaded {len(models)} of {len(arms)} arms")

    for row in arms.itertuples(index=False):
        if row.arm not in models:
            continue
        try:
            cache_subset_mu(row.arm, row.cell, row.seed, row.kind, device)
        except Exception:
            log.error(f"arm {row.arm} subset-side mu failed, it will be skipped at score time"
                      f"\n{traceback.format_exc()}")

    manifest_rows = []
    bounds = shard_bounds(len(index))
    for shard_id, (lo, hi) in enumerate(tqdm(bounds, desc="extract shards", total=len(bounds))):
        feature_path = out_dir / "features" / f"shard_{shard_id}.parquet"
        pending = []
        for arm in models:
            if not (mu_cache_dir / arm / f"shard_{shard_id}.npz").exists():
                pending.append(arm)
        if not pending and feature_path.exists():
            continue
        tics, blocks = read_shard(index, lo, hi)
        if not tics:
            log.warning(f"shard {shard_id} produced no usable stars")
            continue
        if not feature_path.exists():
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            shard_features(tics, blocks, feature_path)
        for arm in pending:
            try:
                mean, std, counts = pool_shard_mu(models[arm], blocks, device)
            except Exception:
                log.error(f"shard {shard_id} arm {arm} failed, continuing\n{traceback.format_exc()}")
                manifest_rows.append({"shard": shard_id, "arm": arm, "n_stars": 0, "status": "error"})
                continue
            arm_dir = mu_cache_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            np.savez(arm_dir / f"shard_{shard_id}.npz", tics=np.asarray(tics, dtype=np.int64),
                     mean=mean, std=std, n_win=counts)
            manifest_rows.append({"shard": shard_id, "arm": arm, "n_stars": len(tics), "status": "ok"})
        if device == "cuda":
            torch.cuda.empty_cache()
    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        path = out_dir / "shard_manifest.csv"
        if path.exists():
            manifest = pd.concat([pd.read_csv(path), manifest], ignore_index=True)
        manifest.to_csv(path, index=False)


def subset_mu_path(arm: str, cell: str, seed: int, kind: str) -> Path:
    """
    Where an arm's SUBSET-side mu lives: the exp07 cache for arms it already holds, else an R8-local cache.
    The exp07 cache carries exactly one untrained init (seed -1 = init 0), so untrained inits 1-5 have no
    train/test mu there. Pairing those added rows with init-0 train rows would mix two different random
    encoders inside one probe, which is why they get their own subset pass instead.
    """
    if kind == "untrained" and seed == 0:
        return exp07_cache / "untrained_w256_seed-1.npz"
    if kind == "trained":
        return exp07_cache / f"{cell}_seed{seed}.npz"
    return out_dir / "subset_mu" / f"{arm}.npz"


def cache_subset_mu(arm: str, cell: str, seed: int, kind: str, device: str) -> None:
    """Encode the packed subset train/test stars for an arm the exp07 cache does not cover, same layout."""
    path = subset_mu_path(arm, cell, seed, kind)
    if path.exists():
        return
    model = load_arm_model(cell, seed, device)
    payload = {}
    for split in ["train", "test"]:
        tics, blocks = load_first_segment_blocks(packed, split, window)
        mean, std, _ = pool_shard_mu(model, blocks, device)
        payload[f"tics_{split}"] = np.asarray(tics, dtype=np.int64)
        payload[f"mean_{split}"] = mean
        payload[f"std_{split}"] = std
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    log.info(f"wrote subset-side mu for {arm} --> {path}")


def load_added_arm(arm: str) -> dict[str, np.ndarray] | None:
    """Concatenate one arm's shard caches back into a single star-ordered block, or None if it has none."""
    arm_dir = mu_cache_dir / arm
    if not arm_dir.exists():
        return None
    parts = sorted(arm_dir.glob("shard_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not parts:
        return None
    tics, means, stds = [], [], []
    for part in parts:
        with np.load(part) as data:
            tics.append(data["tics"])
            means.append(data["mean"])
            stds.append(data["std"])
    return {"tics": np.concatenate(tics), "mean": np.concatenate(means, axis=0),
            "std": np.concatenate(stds, axis=0)}


# ----------------------------------------------------------------------------------------------------
# stage: pilot - measure the rate, then choose how many arms the night can afford
# ----------------------------------------------------------------------------------------------------
def run_pilot(index: pd.DataFrame, device: str, pilot_stars: int) -> dict:
    """
    Time one shard end to end and project the full extraction, then pick the arm set from the bands.
    The scandir is timed separately by `index_added_npz` because it is a fixed cost paid once, not a
    per-star rate, and folding it into the rate would overstate every projection.
    """
    decision_path = out_dir / "sizing.json"
    if decision_path.exists():
        with open(decision_path) as handle:
            return json.load(handle)
    sample = index.iloc[:pilot_stars]
    started = time.time()
    tics, blocks = read_shard(sample, 0, len(sample))
    replay_s = time.time() - started
    started = time.time()
    shard_features(tics, blocks, out_dir / "pilot_features.parquet")
    feature_s = time.time() - started
    model = load_arm_model(fbwd_cell, 0, device)
    started = time.time()
    pool_shard_mu(model, blocks, device)
    encode_s = time.time() - started
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    n_added = len(index)
    scale = n_added / max(len(tics), 1)
    per_arm_h = encode_s * scale / 3600.0
    fixed_h = (replay_s + feature_s) * scale / 3600.0
    decision = {"pilot_stars": len(tics), "replay_s": replay_s, "feature_s": feature_s,
                "encode_s": encode_s, "n_added": n_added, "fixed_hours": fixed_h,
                "hours_per_arm": per_arm_h, "arm_set": "drop"}
    for name, budget in arm_set_bands:
        projected = fixed_h + per_arm_h * len(arm_table(name))
        decision[f"projected_hours_{name}"] = projected
        if decision["arm_set"] == "drop" and projected <= budget:
            decision["arm_set"] = name
    with open(decision_path, "w") as handle:
        json.dump(decision, handle, indent=2)
    write_sizing_note(decision)
    return decision


def write_sizing_note(decision: dict) -> None:
    """Persist the R8a deliverable: the measured rate, the projection, and the go/no-go it produced."""
    lines = ["# R8a sizing pilot", "",
             f"Pilot stars: {decision['pilot_stars']}", f"Added stars to extract: {decision['n_added']}",
             f"Replay: {decision['replay_s']} s | features: {decision['feature_s']} s | "
             f"encode (1 arm): {decision['encode_s']} s", "",
             f"Fixed cost (replay + features, all arms share it): {decision['fixed_hours']} h",
             f"Marginal cost per arm: {decision['hours_per_arm']} h", ""]
    for name, budget in arm_set_bands:
        lines.append(f"- {name} ({len(arm_table(name))} arms): {decision[f'projected_hours_{name}']} h "
                     f"vs budget {budget} h")
    lines += ["", f"Decision: **{decision['arm_set']}**", ""]
    if decision["arm_set"] == "drop":
        lines.append("Every band overran its budget, so R8b/R8c are dropped and R8 stays a disclosed "
                     "stretch in the claims matrix. This note is the evidence.")
    with open(out_dir / "sizing_note.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------------
# stage: score - the probe is frozen, only the negative population moves
# ----------------------------------------------------------------------------------------------------
def build_probe_table(train_x: np.ndarray, test_x: np.ndarray, train_tics: np.ndarray,
                      test_tics: np.ndarray, amp: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the star-level table the published probe consumes, train rows then test rows.
    Built as a DataFrame rather than raw arrays so this shares the exact estimator that produced
    exp07_aux_gap_6seed.csv, down to the float32 the residualization sees: a re-implementation with
    float64 amplitudes drifts the mean_resid arm by ~6e-4 and costs the exact reproduction gate.
    """
    parts = []
    for split, x, tics in [("train", train_x, train_tics), ("test", test_x, test_tics)]:
        frame = pd.DataFrame(x, columns=[f"f{j}" for j in range(x.shape[1])])
        frame.insert(0, "tic_id", tics)
        frame.insert(1, "split", split)
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    merged = table.merge(amp[["tic_id", *amplitude_cols]], on="tic_id", how="left")
    merged = merged.merge(pool[["tic_id", *tasks, "is_quiet"]], on="tic_id", how="left")
    assert len(merged) == len(table), "a star appears twice in the amplitude or label join"
    assert not merged[amplitude_cols].isna().any().any(), "a star has no amplitude features"
    return merged


def residualize(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove what the amplitude basis linearly predicts from each feature dim, fitted on train only."""
    out = table.copy()
    train = table[table["split"] == "train"]
    fitter = LinearRegression().fit(train[amplitude_cols].to_numpy(), train[cols].to_numpy())
    out[cols] = table[cols].to_numpy() - fitter.predict(table[amplitude_cols].to_numpy())
    return out


def corpus_prevalence(pool: pd.DataFrame) -> dict[str, float]:
    """Base rate of each task over every corpus star with windows: the rate the `matched` row targets."""
    out = {}
    for task in tasks:
        out[task] = float((pool[task] == 1).sum()) / float(len(pool))
    return out


def score_arm(arm: str, cell: str, seed: int, kind: str, pool: pd.DataFrame, added: dict,
              added_amp: pd.DataFrame, subset_amp: pd.DataFrame, prevalence: dict[str, float],
              draws: int) -> list[dict]:
    """
    Score one arm's four tasks across every population and pooling.
    Train rows and existing test rows come from the exp07 mu cache so this reproduces the published
    estimator exactly; the added rows come from R8's own cache and only ever enter the test side.
    """
    cached = np.load(subset_mu_path(arm, cell, seed, kind))
    train_tics = cached["tics_train"]
    test_tics = np.concatenate([cached["tics_test"], added["tics"]])
    amp = pd.concat([subset_amp, added_amp], ignore_index=True).drop_duplicates("tic_id")

    train_mean = cached["mean_train"]
    test_mean = np.concatenate([cached["mean_test"], added["mean"]], axis=0)
    mean_table = build_probe_table(train_mean, test_mean, train_tics, test_tics, amp, pool)
    mean_cols = []
    for column in mean_table.columns:
        if column.startswith("f"):
            mean_cols.append(column)
    std_table = build_probe_table(
        np.concatenate([train_mean, cached["std_train"]], axis=1),
        np.concatenate([test_mean, np.concatenate([cached["std_test"], added["std"]], axis=0)], axis=1),
        train_tics, test_tics, amp, pool)
    std_cols = []
    for column in std_table.columns:
        if column.startswith("f"):
            std_cols.append(column)
    poolings = {"mean": (mean_table, mean_cols),
                "mean_resid": (residualize(mean_table, mean_cols), mean_cols),
                "mean_std": (std_table, std_cols)}

    n_existing = len(cached["tics_test"])
    is_added = np.concatenate([np.zeros(n_existing, dtype=bool), np.ones(len(added["tics"]), dtype=bool)])
    is_quiet = mean_table.loc[mean_table["split"] == "test", "is_quiet"].to_numpy()

    rows = []
    for pooling, (table, use_cols) in poolings.items():
        for task in tasks:
            _, test_y, scores = logistic_scores(table, use_cols, task)
            base = {"arm": arm, "cell": cell, "seed": seed, "kind": kind, "pooling": pooling, "task": task}
            keep_quiet = (~is_added) | is_quiet
            keep_survey = (~is_added) | (test_y == 0)
            for population, keep in [("subset_test", ~is_added), ("quiet_full", keep_quiet),
                                     ("survey_full", keep_survey)]:
                rows.append(population_row(base, population, test_y[keep], scores[keep]))
            for population, keep in [("quiet_matched", keep_quiet), ("survey_matched", keep_survey)]:
                rows.append(matched_row(base, population, test_y, scores, keep, is_added,
                                        prevalence[task], draws))
    return rows


def population_row(base: dict, population: str, y: np.ndarray, scores: np.ndarray) -> dict:
    """One deterministic population's PR-AUC, carrying the counts that make its prevalence explicit."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return {**base, "population": population, "n_pos": n_pos, "n_neg": n_neg,
            "prevalence": n_pos / float(len(y)), "pr_auc": float(average_precision_score(y, scores)),
            "pr_auc_draw_sd": np.nan, "n_draws": 0}


def matched_row(base: dict, population: str, y: np.ndarray, scores: np.ndarray, keep: np.ndarray,
                is_added: np.ndarray, target_prevalence: float, draws: int) -> dict:
    """
    PR-AUC with the negative pool subsampled to the corpus base rate, averaged over repeated draws.
    Positives cannot expand (85% of them sit permanently in the probe's train split), so matching the
    survey rate means thinning negatives, and the draw is a second noise source that gets its own
    spread rather than being hidden inside the seed spread.
    Usually the existing test negatives are all kept and the shortfall is drawn from the added pool.
    `rotation` inverts this: quiet excludes rotators, so its positives reach the subset only through the
    other strata and the test split already sits BELOW the 12.5% corpus rate. There the only way to
    match is to thin the existing negatives, so that branch draws from them instead of returning NaN.
    """
    positives = keep & (y == 1)
    fixed_negatives = keep & (y == 0) & (~is_added) # the existing test negatives, kept whole when possible
    draw_pool = np.where(keep & (y == 0) & is_added)[0]
    n_pos = int(positives.sum())
    n_neg_target = int(round(n_pos * (1.0 - target_prevalence) / target_prevalence))
    n_draw = n_neg_target - int(fixed_negatives.sum())
    if n_draw < 0:
        base_index = np.where(positives)[0]
        draw_pool = np.where(fixed_negatives)[0]
        n_draw = n_neg_target
    else:
        base_index = np.where(positives | fixed_negatives)[0]
    if n_pos == 0 or n_draw > len(draw_pool):
        return {**base, "population": population, "n_pos": n_pos, "n_neg": np.nan,
                "prevalence": np.nan, "pr_auc": np.nan, "pr_auc_draw_sd": np.nan, "n_draws": 0}
    values = []
    for draw in range(draws):
        rng = np.random.default_rng(draw)
        chosen = rng.choice(draw_pool, size=n_draw, replace=False)
        index = np.concatenate([base_index, chosen])
        values.append(float(average_precision_score(y[index], scores[index])))
    return {**base, "population": population, "n_pos": n_pos, "n_neg": n_neg_target,
            "prevalence": n_pos / float(n_pos + n_neg_target), "pr_auc": float(np.mean(values)),
            "pr_auc_draw_sd": float(np.std(values, ddof=1)), "n_draws": draws}


def run_score(pool: pd.DataFrame, arms: pd.DataFrame, draws: int) -> None:
    """Score every extracted arm, appending per arm so an interrupted run resumes where it stopped."""
    scores_path = out_dir / "fullpool_scores.csv"
    done = set()
    if scores_path.exists():
        done = set(pd.read_csv(scores_path)["arm"].unique().tolist())
    subset_amp = pd.read_parquet(subset_features)[["tic_id", *amplitude_cols]].drop_duplicates("tic_id")
    feature_parts = []
    for path in sorted((out_dir / "features").glob("shard_*.parquet")):
        feature_parts.append(pd.read_parquet(path))
    added_amp = pd.concat(feature_parts, ignore_index=True).drop_duplicates("tic_id")
    prevalence = corpus_prevalence(pool)
    log.info(f"corpus prevalence: {prevalence}")

    for row in tqdm(list(arms.itertuples(index=False)), desc="score arms", total=len(arms)):
        if row.arm in done:
            continue
        added = load_added_arm(row.arm)
        if added is None:
            log.warning(f"arm {row.arm} has no extracted shards, skipped")
            continue
        if not subset_mu_path(row.arm, row.cell, row.seed, row.kind).exists():
            log.warning(f"arm {row.arm} has no subset-side mu, skipped")
            continue
        try:
            rows = score_arm(row.arm, row.cell, row.seed, row.kind, pool, added, added_amp,
                             subset_amp, prevalence, draws)
        except Exception:
            log.error(f"arm {row.arm} failed to score, continuing\n{traceback.format_exc()}")
            continue
        frame = pd.DataFrame(rows)
        if scores_path.exists():
            frame = pd.concat([pd.read_csv(scores_path), frame], ignore_index=True)
        frame.to_csv(scores_path, index=False)
    summarize(scores_path)


def summarize(scores_path: Path) -> None:
    """Collapse the per-seed rows to the reportable table: mean, across-seed spread, across-draw spread."""
    if not scores_path.exists():
        return
    frame = pd.read_csv(scores_path)
    grouped = frame.groupby(["cell", "kind", "pooling", "task", "population"], dropna=False)
    summary = grouped.agg(n_seeds=("seed", "nunique"), pr_auc=("pr_auc", "mean"),
                          pr_auc_seed_sd=("pr_auc", "std"), pr_auc_draw_sd=("pr_auc_draw_sd", "mean"),
                          n_pos=("n_pos", "mean"), n_neg=("n_neg", "mean"),
                          prevalence=("prevalence", "mean")).reset_index()
    summary.to_csv(out_dir / "fullpool_summary.csv", index=False)
    log.info(f"wrote {out_dir / 'fullpool_summary.csv'} ({len(summary)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description="R8: rescore the v1 tasks at survey prevalence")
    ap.add_argument("--stages", nargs="+", default=["pool", "check", "pilot", "extract", "score"],
                    choices=["pool", "check", "pilot", "extract", "score"])
    ap.add_argument("--arm-set", default=None, choices=["full", "fbwd6", "fbwd3"],
                    help="Override the pilot's sizing decision.")
    ap.add_argument("--pilot-stars", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=20, help="Negative subsamples per matched row.")
    ap.add_argument("--limit-added", type=int, default=0, help="Smoke test: truncate the added-star list.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-file", default="", help="Default: <out_dir>/run.log. Pass 'none' to disable.")
    args = ap.parse_args()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log to stdout, not the logging default of stderr, and own the log file rather than relying on a
    # shell pipeline. Windows PowerShell turns a native command's stderr into NativeCommandError records
    # under `2>&1 |`, which under ErrorActionPreference=Stop kills the run on its very first log line.
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file != "none":
        log_path = Path(args.log_file) if args.log_file else out_dir / "run.log"
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers, force=True)
    log.info(f"device {args.device} | torch.cuda.is_available()={torch.cuda.is_available()} | "
             f"stages {args.stages}")

    pool_path = out_dir / "r8_pool.parquet"
    if "pool" in args.stages or not pool_path.exists():
        log.info("stage pool: scanning the sequences dir (~15 s, no output until it finishes)")
        pool = build_pool()
        pool.to_parquet(pool_path, index=False)
        log.info(f"pool roles:\n{pool['role'].value_counts()}")
        log.info(f"quiet among added: {int(pool.loc[pool['role'] == 'added', 'is_quiet'].sum())}")
    else:
        pool = pd.read_parquet(pool_path)

    if "check" in args.stages:
        try:
            run_check(args.device)
        except Exception:
            log.error(f"check stage failed, continuing to extraction\n{traceback.format_exc()}")

    added_tics = set(pool.loc[pool["role"] == "added", "tic_id"].tolist())
    index = index_npz(added_tics, out_dir / "added_npz_index.parquet")
    if args.limit_added:
        index = index.iloc[: args.limit_added]
    log.info(f"added stars with a first segment: {len(index)}")

    arm_set = args.arm_set
    if "pilot" in args.stages and arm_set is None:
        decision = run_pilot(index, args.device, args.pilot_stars)
        arm_set = decision["arm_set"]
        log.info(f"sizing decision: {arm_set} (fixed {decision['fixed_hours']} h, "
                 f"per-arm {decision['hours_per_arm']} h)")
    if arm_set is None:
        arm_set = "fbwd6"
    if arm_set == "drop":
        log.warning("projection exceeded every band: R8b/R8c dropped, see sizing_note.md")
        return 0
    arms = arm_table(arm_set)

    if "extract" in args.stages:
        run_extract(index, arms, args.device)
    if "score" in args.stages:
        run_score(pool, arms, args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
