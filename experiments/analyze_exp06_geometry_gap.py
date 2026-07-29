"""exp06 eval: the geometry/coverage gap table and the H2 mu-ACF pre-gate, per the exp06 manifest.

For every exp06 cell x seed this script encodes the frozen v1 subset's first-segment windows at the
cell's own geometry and scores the pre-registered probe arms, all inside the v1 linear lock:

  mean            the protocol of record through exp05 (mean-pooled window mu -> logistic)
  mean_resid      THE GATE ARM: amplitude-residualized mean-pooled mu (regress every mu dim on the four
                  amplitude scalars, fitted on train only) -> logistic. The amplitude axis cannot carry
                  the cross-geometry comparison (manifest LOCKED DESIGN; forensics section 2.2).
  mean_std        mean (+) std concatenated over the bag (MIL verdict pooling), raw mu
  kmatch4         mean pooling on bags subsampled to K=4 windows (3 draws, PR-AUC averaged) - equalizes
                  bag size across geometries (16/8/4) so within-task coverage separates from bag size.
                  Skipped at w2048 (first-segment bags hold as few as 2 windows, K=4 unreachable).

Every arm is also scored on the per-geometry capacity-matched UNTRAINED encoder (fixed seed-0 init,
seed column -1) - mandatory because decoder/encoder parameter count grows with window (2.3M -> 13.3M),
so absolute gains across geometries can be pure capacity (ablation skill rule 6).

The H2 pre-gate rides in the same encoder pass: per-star mu-trajectory autocorrelation over the first
segment's consecutive windows (test split), aggregated by lag and by class. "periodic" here is
pulsating|eb|rotation = 1 in the subset labels and "quiet" is no v1 label - coarser than the temporal
forensics script's catalogued-period classes, but the pre-gate only asks whether lag-1 ACF rises with
window length (threshold 0.3, manifest).

The amplitude scalars come from experiments/exp06_features_cache.parquet (star-keyed; computed from the
same underlying flux, so they are geometry-independent and reusable as-is).

Writes experiments/exp06_geometry_gap.csv and experiments/exp06_geometry_acf.csv.
Run (repo root, swm env, PYTHONPATH=src):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp06_geometry_gap.py
    python experiments/analyze_exp06_geometry_gap.py --cells exp06_w256_off --seeds 0 --limit-stars 300
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from swm.eval.skyline import _make_untrained, load_first_segment_blocks, logistic_scores
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

TASKS = ("pulsating", "eb", "rotation", "transit")
AMPLITUDE_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"] # forensics 2.2 basis, periodicity-free

DEFAULT_CELLS = ["exp06_w256_off", "exp06_w256_fbwd", "exp06_w512_off", "exp06_w512_fbwd",
                 "exp06_w1024_off", "exp06_w1024_fbwd", "exp06_w2048_off"]
KMATCH_K = 4
KMATCH_DRAWS = 3
MAX_ACF_LAG = 15


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild one run's world model from its checkpoint (strict load; window read from the stored cfg)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    mc = cfg["model"]
    model = WorldModel(
        in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
        z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
        gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
        dyn_mode=str(mc.get("dyn_mode", "fwd")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_blocks_batched(model: WorldModel, blocks: list[np.ndarray], device: str,
                          batch: int = 4096) -> list[np.ndarray]:
    """
    Encode every star's window block in large concatenated batches, then split back per star.
    One GPU call per ~4096 windows instead of one per star makes a full split pass a few seconds.
    """
    counts = [b.shape[0] for b in blocks]
    flat = np.concatenate(blocks, axis=0) # (total_windows, window)
    chunks = []
    for i in range(0, flat.shape[0], batch):
        x = torch.from_numpy(flat[i : i + batch]).unsqueeze(-1).to(device) # (b, window, 1)
        mu, _ = model.encoder(x) # (b, z)
        chunks.append(mu.float().cpu().numpy())
    mu_all = np.concatenate(chunks, axis=0)
    out = []
    start = 0
    for n in counts:
        out.append(mu_all[start : start + n])
        start += n
    return out


def pool_table(mu_blocks: dict[str, list[np.ndarray]], tics: dict[str, list[int]],
               labels: pd.DataFrame, pooling: str, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """One star-level feature table (train+test) for a pooling operator, labels attached."""
    parts = []
    for split in ["train", "test"]:
        rows = []
        for block in mu_blocks[split]:
            if pooling == "mean":
                rows.append(block.mean(axis=0))
            elif pooling == "mean_std":
                rows.append(np.concatenate([block.mean(axis=0), block.std(axis=0)]))
            elif pooling == "kmatch4":
                pick = rng.choice(block.shape[0], KMATCH_K, replace=False)
                rows.append(block[pick].mean(axis=0))
            else:
                raise ValueError(f"unknown pooling {pooling!r}")
        frame = pd.DataFrame(np.stack(rows), columns=[f"f{j}" for j in range(len(rows[0]))])
        frame.insert(0, "tic_id", tics[split])
        frame.insert(1, "split", split)
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    merged = table.merge(labels[["tic_id", *TASKS, *AMPLITUDE_COLS]], on="tic_id", how="inner")
    assert len(merged) == len(table), "a packed star is missing from the feature cache / labels"
    return merged


def residualize(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove what the amplitude basis linearly predicts from each feature dim (fit on train only)."""
    out = table.copy()
    train = table[table["split"] == "train"]
    fitter = LinearRegression().fit(train[AMPLITUDE_COLS].to_numpy(), train[cols].to_numpy())
    out[cols] = table[cols].to_numpy() - fitter.predict(table[AMPLITUDE_COLS].to_numpy())
    return out


def probe_rows(table: pd.DataFrame, meta: dict) -> list[dict]:
    """Protocol-matched logistic probe on every v1 task; one output row per task."""
    cols = [c for c in table.columns if c.startswith("f")]
    rows = []
    for task in TASKS:
        _, y, scores = logistic_scores(table, cols, task)
        rows.append({**meta, "task": task, "pr_auc": float(average_precision_score(y, scores)),
                     "n_test": int(len(y)), "n_pos": int(y.sum())})
    return rows


def trajectory_acf(mu: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation of one star's mu trajectory averaged over dims (same estimator as the temporal
    forensics script: demean each dim over time; lags beyond the trajectory return NaN)."""
    centred = mu - mu.mean(axis=0, keepdims=True) # (T, z)
    denom = (centred ** 2).sum(axis=0) # (z,)
    out = np.full(max_lag, np.nan)
    for lag in range(1, max_lag + 1):
        if lag >= centred.shape[0]:
            break
        num = (centred[:-lag] * centred[lag:]).sum(axis=0) # (z,)
        usable = denom > 1e-12
        if usable.any():
            out[lag - 1] = float(np.mean(num[usable] / denom[usable]))
    return out


def acf_rows(mu_blocks: list[np.ndarray], tics: list[int], labels: pd.DataFrame, meta: dict) -> list[dict]:
    """Per-class mean ACF by lag over the test stars' first-segment mu trajectories."""
    lab = labels.set_index("tic_id")
    periodic = (lab[["pulsating", "eb", "rotation"]].sum(axis=1) > 0)
    quiet = (lab[list(TASKS)].sum(axis=1) == 0)
    per_star = {"periodic": [], "quiet": []}
    for tic, block in zip(tics, mu_blocks):
        # T=2 trajectories make the lag-1 estimator deterministically -0.5 (a demeaned 2-point series is
        # antisymmetric), so require >= 3 windows; at w2048 this restricts to longer-segment stars.
        if block.shape[0] < 3 or tic not in lab.index:
            continue
        cls = "periodic" if periodic.get(tic, False) else ("quiet" if quiet.get(tic, False) else None)
        if cls is None:
            continue
        per_star[cls].append(trajectory_acf(block, MAX_ACF_LAG))
    rows = []
    for cls, stack in per_star.items():
        if not stack: # a class can be empty when the T>=3 guard bites (w2048 bags are mostly 2 windows)
            continue
        arr = np.stack(stack) # (n_stars, max_lag)
        for lag in range(MAX_ACF_LAG):
            vals = arr[:, lag]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            rows.append({**meta, "cls": cls, "lag": lag + 1, "acf": float(vals.mean()),
                         "n_stars": int(len(vals))})
    return rows


def score_arm(mu_blocks, tics, labels, meta: dict, want_kmatch: bool) -> tuple[list[dict], pd.DataFrame]:
    """All pooling/residual arms for one encoder; returns probe rows + the mean table (for reuse)."""
    rows = []
    mean_table = pool_table(mu_blocks, tics, labels, "mean")
    rows += probe_rows(mean_table, {**meta, "pooling": "mean"})
    cols = [c for c in mean_table.columns if c.startswith("f")]
    rows += probe_rows(residualize(mean_table, cols), {**meta, "pooling": "mean_resid"})
    rows += probe_rows(pool_table(mu_blocks, tics, labels, "mean_std"), {**meta, "pooling": "mean_std"})
    if want_kmatch:
        per_task: dict[str, list[float]] = {t: [] for t in TASKS}
        n_info: dict[str, tuple[int, int]] = {}
        for draw in range(KMATCH_DRAWS): # PR-AUC averaged over draws, per task (MIL kmatch protocol)
            rng = np.random.default_rng(draw) # fixed draw seeds: measurement RNG, not a model seed
            table = pool_table(mu_blocks, tics, labels, "kmatch4", rng=rng)
            for r in probe_rows(table, {**meta, "pooling": "kmatch4"}):
                per_task[r["task"]].append(r["pr_auc"])
                n_info[r["task"]] = (r["n_test"], r["n_pos"])
        for task, aps in per_task.items():
            rows.append({**meta, "pooling": "kmatch4", "task": task, "pr_auc": float(np.mean(aps)),
                         "n_test": n_info[task][0], "n_pos": n_info[task][1]})
    return rows, mean_table


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 geometry gap table + H2 mu-ACF pre-gate")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--feature-cache", default="experiments/exp06_features_cache.parquet")
    ap.add_argument("--limit-stars", type=int, default=0, help="smoke test: truncate each split")
    ap.add_argument("--out", default="experiments/exp06_geometry_gap.csv")
    ap.add_argument("--out-acf", default="experiments/exp06_geometry_acf.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = pd.read_parquet(ROOT / args.feature_cache)[["tic_id", *TASKS, *AMPLITUDE_COLS]]
    labels = labels.drop_duplicates("tic_id")

    gap_rows: list[dict] = []
    acf_out: list[dict] = []
    block_cache: dict[int, tuple] = {} # window -> (tics, blocks); one geometry cached at a time
    untrained_done: set[int] = set()

    # geometry-major order so the raw window blocks load once per geometry
    cells = sorted(args.cells, key=lambda c: int(torch.load(
        ROOT / "experiments" / c / "models" / f"{args.variant}_seed{args.seeds[0]}" / f"{args.ckpt}.pt",
        map_location="cpu", weights_only=False)["cfg"]["data"]["window"]))

    for cell in tqdm(cells, desc="cells", total=len(cells)):
        for seed in args.seeds:
            ckpt_path = ROOT / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            assert ckpt_path.exists(), f"missing {ckpt_path}"
            model, cfg = load_model(ckpt_path, device)
            window = int(cfg["data"]["window"])

            if window not in block_cache:
                block_cache.clear() # cells are geometry-major, so earlier geometries never come back
                packed = ROOT / "experiments" / cell / "packed"
                tics, blocks = {}, {}
                for split in ["train", "test"]:
                    t, b = load_first_segment_blocks(packed, split, window)
                    if args.limit_stars:
                        t, b = t[: args.limit_stars], b[: args.limit_stars]
                    tics[split], blocks[split] = t, b
                block_cache[window] = (tics, blocks)
            tics, blocks = block_cache[window]
            want_kmatch = window != 2048 # w2048 bags can hold only 2 windows; K=4 unreachable

            mu = {s: encode_blocks_batched(model, blocks[s], device) for s in ["train", "test"]}
            meta = {"cell": cell, "window": window, "seed": seed, "arm": "trained"}
            rows, _ = score_arm(mu, tics, labels, meta, want_kmatch)
            gap_rows += rows
            acf_out += acf_rows(mu["test"], tics["test"], labels, meta)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            if window not in untrained_done: # one capacity-matched untrained arm per geometry (seed -1)
                mc = cfg["model"]
                untrained = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]),
                                            int(mc["z_dim"]), window, int(mc["gru_hidden"]),
                                            int(mc["gru_layers"]), device)
                mu_u = {s: encode_blocks_batched(untrained, blocks[s], device) for s in ["train", "test"]}
                meta_u = {"cell": f"untrained_w{window}", "window": window, "seed": -1, "arm": "untrained"}
                rows, _ = score_arm(mu_u, tics, labels, meta_u, want_kmatch)
                gap_rows += rows
                acf_out += acf_rows(mu_u["test"], tics["test"], labels, meta_u)
                untrained_done.add(window)
                del untrained
                if device == "cuda":
                    torch.cuda.empty_cache()
            log.info(f"scored {cell} seed {seed} (window {window})")
            # rewrite after every run so a late crash never loses the finished work
            pd.DataFrame(gap_rows).to_csv(ROOT / args.out, index=False)
            pd.DataFrame(acf_out).to_csv(ROOT / args.out_acf, index=False)

    log.info(f"wrote {args.out} ({len(gap_rows)} rows) and {args.out_acf} ({len(acf_out)} rows)")


if __name__ == "__main__":
    main()
