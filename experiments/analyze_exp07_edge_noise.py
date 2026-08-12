"""exp07 follow-up: is the residual window-edge error of the tapered recipe noise-driven?

The exp07 verdict reports one number for the edge defect per run: `edge_max`, the ratio of the worst
endpoint MSE to the interior level, averaged over a fixed window sample (`comb0p3_fbwd` 31.4 +/- 4.8,
`hann0p3_fbwd` 1.16 +/- 0.03). The live light-curve panels added to the exp07 diagnostics notebook
(section I, 2026-08-04) show something that number cannot express: on some individual stars the tapered
recipe still draws a visible spike at every 256-cadence boundary, and those stars look like the noisy
ones. A mean ratio of 1.16 is consistent with both "a small residual spread evenly over all stars" and
"no residual on most stars plus a real one on a noisy minority", and the two readings imply different
things for exp08 -- the second would mean the taper's job is not finished for the low-SNR half of the
corpus.

This measures the per-STAR version of the defect and correlates it with that star's own noise level.
For every window: squared error at the two endpoints against the interior band [8, 248), plus a
point-to-point scatter estimate std(diff(x))/sqrt(2) (in MAD units, since the pipeline MAD-normalises
each segment) and the window's structure amplitude std(x). Per star those are reduced by median over
all of its test windows -- all segments, not just the first, because this is a reconstruction property
and carries none of the label-side bag-size confound (F2/F3).

Reported per run: Spearman rho between star noise and star edge excess, the same against the
amplitude-to-noise ratio, and a noise-decile table. Runs are both extension cells x both arms x seeds
0-5, so the winner-vs-baseline contrast is paired by seed and the arm is never pooled (F21).

Writes experiments/exp07_edge_noise_stars.csv (one row per run x star) and
experiments/exp07_edge_noise_summary.csv (one row per run). Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp07_edge_noise.py
    python experiments/analyze_exp07_edge_noise.py --cells exp07_hann0p3_fbwd --seeds 0 --limit 200
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from swm.models import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp07_edge_noise")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"  # every exp07 cell junctions to this
WINDOW = 256
INTERIOR = (8, 248)          # interior reference band, matching analyze_exp06_edge.py / C1
DEFAULT_CELLS = ["exp07_hann0p3_fbwd", "exp07_comb0p3_fbwd", "exp07_hann0p3_off", "exp07_comb0p3_off"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5]
BATCH = 512
N_DECILES = 10


def load_model(cell: str, seed: int, device: str) -> WorldModel:
    """Rebuild one run's world model from its selected checkpoint (eval mode)."""
    path = ROOT / "experiments" / cell / "models" / f"B_seed{seed}" / "best_recon_aux.pt"
    assert path.exists(), f"missing checkpoint {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    mc = cfg["model"]
    model = WorldModel(in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
                       z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
                       gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
                       dyn_mode=str(mc.get("dyn_mode", "fwd"))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_test_split(limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows, tic_id per window) for the packed test split, all segments of every star."""
    index = pd.read_parquet(PACKED / "test_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(PACKED / "test_windows.dat", dtype=np.float32, mode="r", shape=(total, WINDOW))
    if limit is not None:
        keep_tics = sorted(index["tic_id"].unique())[:limit]
        index = index[index["tic_id"].isin(keep_tics)]
    rows, tics = [], []
    for row_start, n_win, tic in zip(index["row_start"], index["n_win"], index["tic_id"]):
        rows.append(np.arange(int(row_start), int(row_start) + int(n_win)))
        tics.append(np.full(int(n_win), int(tic), dtype=np.int64))
    rows = np.concatenate(rows)
    return np.array(memmap[rows], dtype=np.float32), np.concatenate(tics)  # (n_win, 256), (n_win,)


def window_stats(x: np.ndarray) -> pd.DataFrame:
    """Per-window noise and amplitude, both in the MAD units the pipeline normalises each segment to.

    `noise` is the point-to-point scatter std(diff)/sqrt(2), which is dominated by photon noise rather
    than by the astrophysical signal at 2-min cadence; `amp` is the window's total scatter, so amp/noise
    is a crude per-window SNR of whatever structure the star has.
    """
    noise = np.std(np.diff(x, axis=1), axis=1) / np.sqrt(2.0)  # (n_win,)
    amp = np.std(x, axis=1)                                    # (n_win,)
    return pd.DataFrame({"noise": noise, "amp": amp})


@torch.no_grad()
def edge_excess(model: WorldModel, x: np.ndarray, device: str, desc: str) -> pd.DataFrame:
    """Per-window endpoint error against the interior level, for one run.

    Returns the raw squared errors so both readings are available downstream: the RATIO edge/interior
    (the form `edge_max` uses, which a quiet star can inflate by having a tiny interior error) and the
    absolute EXCESS edge - interior, which is in flux units and cannot be inflated that way.
    """
    lo, hi = INTERIOR
    edge, interior = [], []
    for start in tqdm(range(0, x.shape[0], BATCH), desc=desc, total=(x.shape[0] + BATCH - 1) // BATCH):
        chunk = torch.from_numpy(x[start:start + BATCH]).unsqueeze(-1).to(device)  # (b, 256, 1)
        mu, _ = model.encoder(chunk)
        recon = model.decoder(mu)[:, :, 0]                                          # (b, 256)
        err = (recon - chunk[:, :, 0]) ** 2                                         # (b, 256)
        edge.append(((err[:, 0] + err[:, -1]) / 2.0).cpu().numpy())
        interior.append(err[:, lo:hi].mean(dim=1).cpu().numpy())
    return pd.DataFrame({"edge_mse": np.concatenate(edge), "interior_mse": np.concatenate(interior)})


def per_star(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-reduce the per-window measurements to one row per star."""
    frame = frame.copy()
    frame["edge_ratio"] = frame["edge_mse"] / frame["interior_mse"]
    frame["edge_excess"] = frame["edge_mse"] - frame["interior_mse"]
    stars = frame.groupby("tic_id").median(numeric_only=True)
    stars["n_win"] = frame.groupby("tic_id").size()
    stars["snr"] = stars["amp"] / stars["noise"]
    return stars.reset_index()


def summarise(stars: pd.DataFrame, cell: str, seed: int) -> dict:
    """Spearman associations and the decile spread for one run."""
    row = {"cell": cell, "seed": seed, "n_stars": int(len(stars)),
           "edge_ratio_median": float(stars["edge_ratio"].median()),
           "edge_excess_median": float(stars["edge_excess"].median())}
    for target in ["edge_ratio", "edge_excess"]:
        for driver in ["noise", "snr"]:
            rho, p = spearmanr(stars[driver], stars[target])
            row[f"rho_{driver}_{target}"] = float(rho)
            row[f"p_{driver}_{target}"] = float(p)
    deciles = pd.qcut(stars["noise"], N_DECILES, labels=False, duplicates="drop")
    for name, target in [("ratio", "edge_ratio"), ("excess", "edge_excess")]:
        by_decile = stars.groupby(deciles)[target].median()
        row[f"{name}_noisiest_decile"] = float(by_decile.iloc[-1])
        row[f"{name}_quietest_decile"] = float(by_decile.iloc[0])
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-star window-edge error against per-star noise.")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--limit", type=int, default=None, help="First N test stars only (smoke).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", default=None, help="Default: experiments/exp07_edge_noise")
    args = ap.parse_args()

    prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "experiments" / "exp07_edge_noise"
    windows, tics = load_test_split(args.limit)
    log.info(f"test split: {windows.shape[0]} windows over {len(np.unique(tics))} stars, device={args.device}")
    stats = window_stats(windows)
    stats["tic_id"] = tics

    star_rows, summary_rows = [], []
    for cell in args.cells:
        for seed in args.seeds:
            model = load_model(cell, seed, args.device)
            measured = edge_excess(model, windows, args.device, desc=f"{cell}:s{seed}")
            stars = per_star(pd.concat([stats.reset_index(drop=True), measured], axis=1))
            stars.insert(0, "seed", seed)
            stars.insert(0, "cell", cell)
            star_rows.append(stars)
            summary_rows.append(summarise(stars, cell, seed))
            log.info(f"{cell} seed {seed}: median ratio {summary_rows[-1]['edge_ratio_median']:.3f}, "
                     f"rho(noise, ratio) {summary_rows[-1]['rho_noise_edge_ratio']:+.3f}")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    pd.concat(star_rows, ignore_index=True).to_csv(f"{prefix}_stars.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    log.info(f"wrote {prefix}_stars.csv and {prefix}_summary.csv")
    print(summary[["cell", "seed", "edge_ratio_median", "rho_noise_edge_ratio",
                   "rho_noise_edge_excess", "ratio_noisiest_decile", "ratio_quietest_decile"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
