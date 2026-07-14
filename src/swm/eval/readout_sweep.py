"""exp03 eval-time sweep: frozen readout x pooling fan over trained checkpoints (plan 2026-07-13).

For every (experiment, checkpoint kind) this module encodes the first-segment windows once, caches the
window-level mu to disk, then scores every requested (readout x pooling x task) cell on the identical
first-segment protocol the skyline/gap tables use. The win condition per cell is trained PR-AUC minus the
untrained (random-init, seed-0) encoder's PR-AUC under the SAME readout and pooling; the untrained arm is
geometry-shared, so it is encoded and scored once and joined onto every experiment's rows.

Readouts (all frozen-encoder; the encoder is never fine-tuned):
  logistic  - the v1 linear probe (standardize on train, balanced class weights)
  gbm       - HistGradientBoostingClassifier, the nonlinear-but-frozen reference (ADR-0008, proposed)
  mlp       - one-hidden-layer sklearn MLP on standardized mu, positives oversampled to balance
Poolings over a star's first-segment window mu rows:
  mean / max / quantile - feature pooling before the readout (quantile = q10|q50|q90 concat, 3z dims)
  window_score          - MIL-style score pooling: fit the readout on window-level rows (star label
                          broadcast), star score = max over its windows' scores (grill 2026-07-13)

Nonlinear-readout numbers are DIAGNOSTIC until ADR-0008 is signed; the linear probe remains the v1 headline.
Point estimates only - the paired-bootstrap CI machinery stays in swm.eval.skyline for the winners.
Rows append to experiments/<exp>/results/readout_sweep.csv with run_id + git_sha (append-only, auditable).

Run (from repo root, swm env, PYTHONPATH=src), e.g. the quick scan then a full fan on one combo:
    python -m swm.eval.readout_sweep --exp-glob "exp03_*" --ckpts best_recon_aux best --readouts logistic gbm --poolings mean
    python -m swm.eval.readout_sweep --exp-glob "exp03_fb0p02_b0p3_lpsd" --ckpts best_recon_aux best last
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from swm.eval.skyline import _git_sha, _make_untrained, load_first_segment_blocks
from swm.models import WorldModel

log = logging.getLogger(__name__)

repo_root = Path(__file__).resolve().parents[3]
tasks_default = ("pulsating", "eb", "rotation", "transit") # transit is report-only (weak-signal data-side)
readouts_default = ("logistic", "gbm", "mlp")
poolings_default = ("mean", "max", "quantile", "window_score")


# ----------------------------------------------------------------------------------------------------
# mu caching: encode each star's first-segment windows once per checkpoint, reuse for every cell
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_blocks(model: WorldModel, blocks: list[np.ndarray], device: str) -> list[np.ndarray]:
    """Encode each star's first-segment window block to posterior-mean mu, keeping the per-star grouping."""
    mu_blocks = []
    for block in blocks:
        x = torch.from_numpy(block).unsqueeze(-1).to(device) # (n_win, window, 1)
        mu, _ = model.encoder(x) # (n_win, z)
        mu_blocks.append(mu.float().cpu().numpy())
    return mu_blocks


def cached_mu(cache_path: Path, model: WorldModel | None, packed_dir: Path, window: int, device: str,
              desc: str) -> dict[str, tuple[list[int], list[np.ndarray]]]:
    """
    Return {split: (tics, per-star window-mu blocks)} for train and test, backed by an .npz cache.
    The cache stores the concatenated mu plus per-star row counts so the expensive encoder pass runs
    once per checkpoint; a later invocation (resume, extra cells) loads it instead of touching the GPU.
    """
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        result = {}
        for split in ["train", "test"]:
            flat = payload[f"{split}_mu"]
            counts = payload[f"{split}_counts"]
            tics = payload[f"{split}_tics"].tolist()
            blocks = []
            start = 0
            for count in counts:
                blocks.append(flat[start : start + int(count)])
                start += int(count)
            result[split] = (tics, blocks)
        return result
    assert model is not None, f"no cache at {cache_path} and no model to build it"
    result = {}
    arrays = {}
    for split in ["train", "test"]:
        tics, blocks = load_first_segment_blocks(packed_dir, split, window)
        mu_blocks = []
        for i in tqdm(range(len(blocks)), desc=f"{desc}[{split}]", total=len(blocks)):
            mu_blocks.append(encode_blocks(model, [blocks[i]], device)[0])
        result[split] = (tics, mu_blocks)
        counts = []
        for mu_block in mu_blocks:
            counts.append(mu_block.shape[0])
        arrays[f"{split}_mu"] = np.concatenate(mu_blocks, axis=0)
        arrays[f"{split}_counts"] = np.array(counts, dtype=np.int64)
        arrays[f"{split}_tics"] = np.array(tics, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    return result


# ----------------------------------------------------------------------------------------------------
# poolings + readouts
# ----------------------------------------------------------------------------------------------------
def pool_stars(mu_blocks: list[np.ndarray], pooling: str) -> np.ndarray:
    """Reduce each star's (n_win, z) window-mu block to one feature vector under the given pooling."""
    feats = []
    for block in mu_blocks:
        if pooling == "mean":
            feats.append(block.mean(axis=0))
        elif pooling == "max":
            feats.append(block.max(axis=0))
        elif pooling == "quantile":
            q = np.quantile(block, [0.1, 0.5, 0.9], axis=0) # (3, z)
            feats.append(q.reshape(-1)) # q10|q50|q90 concat, 3z dims
        else:
            raise ValueError(f"unknown feature pooling {pooling}")
    return np.stack(feats, axis=0)


def fit_readout_scores(readout: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
    Fit one frozen readout on train features and return P(positive) for the test rows.
    Imbalance handling per readout: class_weight for logistic/gbm; the sklearn MLP takes neither
    class_weight nor sample_weight, so the positive rows are oversampled to parity instead.
    """
    if readout == "logistic":
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train) # learn mean/std on train only (no leakage)
        x_te = scaler.transform(x_test)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(x_tr, y_train)
        return clf.predict_proba(x_te)[:, 1]
    if readout == "gbm":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0) # gradient-boosted trees
        clf.fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1]
    if readout == "mlp":
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train)
        x_te = scaler.transform(x_test)
        pos = np.flatnonzero(y_train == 1)
        neg = np.flatnonzero(y_train == 0)
        rng = np.random.default_rng(0)
        boost = rng.choice(pos, size=max(0, len(neg) - len(pos)), replace=True) # oversample positives to parity
        order = np.concatenate([np.arange(len(y_train)), boost])
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=0)
        clf.fit(x_tr[order], y_train[order])
        return clf.predict_proba(x_te)[:, 1]
    raise ValueError(f"unknown readout {readout}")


def window_score_scores(readout: str, train_blocks: list[np.ndarray], y_train: np.ndarray,
                        test_blocks: list[np.ndarray]) -> np.ndarray:
    """
    MIL-style score pooling: fit the readout on window-level mu with each star's label broadcast to its
    windows, then score every test window and take the max per star. Catches localized signal (eclipses,
    transits) that feature-pooling over a long segment dilutes.
    """
    x_train_rows = np.concatenate(train_blocks, axis=0)
    y_rows = []
    for i in range(len(train_blocks)):
        y_rows.append(np.full(train_blocks[i].shape[0], y_train[i], dtype=np.int64))
    y_train_rows = np.concatenate(y_rows)
    x_test_rows = np.concatenate(test_blocks, axis=0)
    row_scores = fit_readout_scores(readout, x_train_rows, y_train_rows, x_test_rows)
    star_scores = np.zeros(len(test_blocks), dtype=np.float64)
    start = 0
    for i in range(len(test_blocks)):
        n = test_blocks[i].shape[0]
        star_scores[i] = row_scores[start : start + n].max() # max over the star's windows
        start += n
    return star_scores


def score_cells(mu: dict, subset: pd.DataFrame, tasks: tuple[str, ...], readouts: tuple[str, ...],
                poolings: tuple[str, ...], label: str) -> pd.DataFrame:
    """
    Score every (pooling x readout x task) cell for one encoder arm and return long-form rows.
    Star-level label vectors come from the subset frame in the aligned ascending-tic block order.
    """
    train_tics, train_blocks = mu["train"]
    test_tics, test_blocks = mu["test"]
    label_of = {}
    for task in tasks:
        label_of[task] = dict(zip(subset["tic_id"].tolist(), subset[task].tolist()))
    rows = []
    cells = []
    for pooling in poolings:
        for readout in readouts:
            cells.append((pooling, readout))
    pooled = {}
    for pooling in poolings:
        if pooling != "window_score":
            pooled[pooling] = (pool_stars(train_blocks, pooling), pool_stars(test_blocks, pooling))
    for pooling, readout in tqdm(cells, desc=f"cells[{label}]", total=len(cells)):
        for task in tasks:
            y_train = np.array([label_of[task][t] for t in train_tics], dtype=np.int64)
            y_test = np.array([label_of[task][t] for t in test_tics], dtype=np.int64)
            if y_train.sum() == 0 or y_test.sum() == 0:
                log.warning(f"{label} {pooling}/{readout}/{task}: a split lacks positives; skipped")
                continue
            if pooling == "window_score":
                scores = window_score_scores(readout, train_blocks, y_train, test_blocks)
            else:
                x_train, x_test = pooled[pooling]
                scores = fit_readout_scores(readout, x_train, y_train, x_test)
            rows.append({
                "pooling": pooling, "readout": readout, "task": task,
                "pr_auc": float(average_precision_score(y_test, scores)),
                "base_rate": float(y_test.mean()), "n_test_pos": int(y_test.sum()), "n_test": int(len(y_test)),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------------------------------
def build_model_from_ckpt(ckpt: dict, device: str) -> tuple[WorldModel, dict]:
    """Instantiate the world model recorded in a checkpoint's cfg dict and load its weights."""
    cfg = ckpt["cfg"]
    mc = cfg["model"]
    model = WorldModel(
        in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
        z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="frozen readout x pooling sweep over experiment checkpoints")
    parser.add_argument("--exp-glob", required=True, help="glob under experiments/ selecting experiment folders")
    parser.add_argument("--ckpts", nargs="+", default=["best_recon_aux", "best"], help="checkpoint stems to score")
    parser.add_argument("--readouts", nargs="+", default=list(readouts_default))
    parser.add_argument("--poolings", nargs="+", default=list(poolings_default))
    parser.add_argument("--tasks", nargs="+", default=list(tasks_default))
    parser.add_argument("--variant", default="B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--untrained-cache", default="experiments/exp03_eval_cache",
                        help="directory holding the shared untrained-arm mu cache + scored cells")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available; encoding targets the GPU"
    readouts = tuple(args.readouts)
    poolings = tuple(args.poolings)
    tasks = tuple(args.tasks)
    subset = pd.read_parquet(repo_root / "processed" / "subset" / "subset_tics.parquet")

    exp_dirs = []
    for exp_dir in sorted((repo_root / "experiments").glob(args.exp_glob)):
        if (exp_dir / "models").exists():
            exp_dirs.append(exp_dir)
    assert len(exp_dirs) > 0, f"no experiment folders with models/ match {args.exp_glob}"
    log.info(f"{len(exp_dirs)} experiments x ckpts {list(args.ckpts)} x {len(readouts)} readouts x {len(poolings)} poolings")

    # Untrained arm: geometry is shared across the sweep (window/seq_len/z locked), so encode + score once.
    first_ckpt = torch.load(
        next((exp_dirs[0] / "models" / f"{args.variant}_seed{args.seed}").glob("*.pt")),
        map_location=device, weights_only=False,
    )
    cfg0 = first_ckpt["cfg"]
    window = int(cfg0["data"]["window"])
    packed_dir = exp_dirs[0] / "packed"
    untrained_dir = repo_root / args.untrained_cache
    untrained = _make_untrained(
        list(cfg0["model"]["enc_channels"]), int(cfg0["model"]["kernel_size"]), int(cfg0["model"]["z_dim"]),
        window, int(cfg0["model"]["gru_hidden"]), int(cfg0["model"]["gru_layers"]), device,
    )
    mu_untrained = cached_mu(untrained_dir / f"untrained_mu_w{window}.npz", untrained, packed_dir, window, device, "untrained")
    untrained_cells_path = untrained_dir / f"untrained_cells_w{window}.csv"
    if untrained_cells_path.exists():
        untrained_cells = pd.read_csv(untrained_cells_path)
    else:
        untrained_cells = pd.DataFrame()
    needed = score_cells(mu_untrained, subset, tasks, readouts, poolings, "untrained")
    if len(untrained_cells) > 0: # keep previously scored cells, add only the new ones
        key_cols = ["pooling", "readout", "task"]
        merged = needed.merge(untrained_cells[key_cols].assign(_seen=1), on=key_cols, how="left")
        needed = needed[merged["_seen"].isna().to_numpy()]
        untrained_cells = pd.concat([untrained_cells, needed], ignore_index=True)
    else:
        untrained_cells = needed
    untrained_cells.to_csv(untrained_cells_path, index=False)
    untrained_lookup = untrained_cells.set_index(["pooling", "readout", "task"])["pr_auc"]

    for exp_dir in tqdm(exp_dirs, desc="experiments", total=len(exp_dirs)):
        run_dir = exp_dir / "models" / f"{args.variant}_seed{args.seed}"
        out_rows = []
        for stem in args.ckpts:
            ckpt_path = run_dir / f"{stem}.pt"
            if not ckpt_path.exists():
                log.warning(f"{exp_dir.name}: no {stem}.pt; skipped (cell dropped, not silent)")
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model, cfg = build_model_from_ckpt(ckpt, device)
            cache_path = run_dir / "extracted" / f"first_segment_window_mu_{stem}.npz"
            mu_trained = cached_mu(cache_path, model, exp_dir / "packed", window, device, f"{exp_dir.name}:{stem}")
            cells = score_cells(mu_trained, subset, tasks, readouts, poolings, f"{exp_dir.name}:{stem}")
            cells["exp_name"] = exp_dir.name
            cells["ckpt"] = stem
            cells["ckpt_epoch"] = int(ckpt["epoch"])
            out_rows.append(cells)
        if len(out_rows) == 0:
            continue
        result = pd.concat(out_rows, ignore_index=True)
        keys = list(zip(result["pooling"], result["readout"], result["task"]))
        result["pr_auc_untrained"] = untrained_lookup.reindex(keys).to_numpy()
        result["gap"] = result["pr_auc"] - result["pr_auc_untrained"]
        result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        result["git_sha"] = _git_sha()
        results_path = exp_dir / "results" / "readout_sweep.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            previous = pd.read_csv(results_path)
            result = pd.concat([previous, result], ignore_index=True) # append-only audit trail
        result.to_csv(results_path, index=False)
        log.info(f"{exp_dir.name}: wrote {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
