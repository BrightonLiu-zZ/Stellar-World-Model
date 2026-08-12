"""Pre-exp08 CHK-2: the Hann taper relocated the reconstruction impulse, it did not remove it.

exp07 closed the window-EDGE defect (`hann0p3` edge_max 1.15x against `comb0p3`'s 31x) and F23 closed
it as a probe cost. Reducing the per-position error profile PER SEED rather than seed-averaged shows
the taper bought that with a new impulse in the window INTERIOR:

    max(mse, pos 112-144) / median(mse, pos 16-239)   comb0p3_fbwd 1.09   hann0p3_fbwd 13.5 +/- 6.0

which is the same order as the edge defect it replaced. The mechanism is immediate: a symmetric Hann
window is exactly zero at the endpoints and maximal at the centre, so the cheapest place for a decoder
to buy log-PSD power moves from p0 to the middle of the window. The spike's position wanders seed to
seed (125 -> 143), which is why a seed-AVERAGED profile smears it to 5x and a fixed 125-143 band both
misses it and cannot be compared against cells that have no bump -- the same ratio-of-means pathology
F23 records for the edge, one level down. This script therefore locates the spike per run by argmax
over the interior instead of assuming where it is.

Two estimators, because the obvious one has no power:

  cell-level   within-arm rho(centre severity, probe score) over the cells of one arm -- the B4/F21
               analogue the exp08 handoff asks for. Reported, but NOT the verdict: centre severity is
               bimodal (hann ~13, everything else ~1-2), so at cell granularity the statistic is a
               two-group difference whose grouping variable is `recipe`, a deliberate design axis. That
               is the F21 trap one level down.

  per-star     the verdict. Within one (cell, seed) -- recipe, arm and seed all held fixed, so there is
               no axis left to pool over -- correlate each star's own centre severity against that
               star's held-out probe score, over ~2000 test stars. This is the estimator that closed Q5
               for the edge (F23/C5), so the answer is directly comparable to it.

Writes experiments/exp07_centre_{stars,summary,cells}.csv. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp07_centre_artifact.py
    python experiments/analyze_exp07_centre_artifact.py --cells exp07_hann0p3_fbwd --seeds 0 --limit 200
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
log = logging.getLogger("exp07_centre")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
WINDOW = 256
INTERIOR = (16, 240)   # reference band, excludes both the edge impulse and the taper's roll-off shoulders
CENTRE = (112, 145)    # search band for the taper-induced spike; the spike's position varies by seed
EDGE_GUARD = 8         # positions nearer than this to an endpoint are edge, never centre
DEFAULT_CELLS = ["exp07_hann0p3_fbwd", "exp07_comb0p3_fbwd", "exp07_hann0p3_off", "exp07_comb0p3_off"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5]
BATCH = 512
STAR_SCORES = ROOT / "experiments" / "exp07_diag_star_scores.parquet"
# the 6-seed dump covers the four extension cells at every seed; the 4-seed dump covers all ten cells.
# Concatenating and de-duplicating gives the widest profile table without preferring one over the other.
PROFILE_DUMPS = [ROOT / "experiments" / "exp07_edge_profiles_6seed.parquet",
                 ROOT / "experiments" / "exp07_edge_profiles.parquet"]


def load_profiles() -> pd.DataFrame:
    """Per-position MSE profiles, 6-seed dump first so it wins any (cell, seed, pos) collision."""
    frames = [pd.read_parquet(p) for p in PROFILE_DUMPS if p.exists()]
    assert frames, f"no per-position profile dump found at {PROFILE_DUMPS}"
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["cell", "seed", "pos"], keep="first")


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
    return np.array(memmap[rows], dtype=np.float32), np.concatenate(tics)


def spike_position(cell: str, seed: int) -> int:
    """Locate this run's interior spike: argmax of the pooled per-position MSE inside the centre band.

    Read from the already-computed profile dump rather than re-derived, so the position used for the
    per-star reduction is the same one the cell-level table reports. Falls back to the band centre for
    runs the 4-seed profile dump does not cover.
    """
    prof = load_profiles()
    sel = prof[(prof["cell"] == cell) & (prof["seed"] == seed)]
    if sel.empty:
        return (CENTRE[0] + CENTRE[1]) // 2
    band = sel[(sel["pos"] >= CENTRE[0]) & (sel["pos"] < CENTRE[1])]
    return int(band.loc[band["mse"].idxmax(), "pos"])


@torch.no_grad()
def centre_excess(model: WorldModel, x: np.ndarray, pos: int, device: str, desc: str) -> pd.DataFrame:
    """Per-window error at the located interior spike, against the interior level and against the edges.

    Returns raw squared errors so both framings survive downstream: the RATIO centre/interior (which a
    quiet star inflates by having a tiny interior error) and the absolute EXCESS centre - interior,
    which is in flux units and cannot be inflated that way. F23's lesson, applied up front.
    """
    lo, hi = INTERIOR
    keep = np.r_[np.arange(lo, max(lo, pos - 2)), np.arange(min(hi, pos + 3), hi)]  # interior minus the spike
    centre, interior, edge = [], [], []
    for start in tqdm(range(0, x.shape[0], BATCH), desc=desc, total=(x.shape[0] + BATCH - 1) // BATCH):
        chunk = torch.from_numpy(x[start:start + BATCH]).unsqueeze(-1).to(device)  # (b, 256, 1)
        mu, _ = model.encoder(chunk)
        recon = model.decoder(mu)[:, :, 0]                                          # (b, 256)
        err = (recon - chunk[:, :, 0]) ** 2                                         # (b, 256)
        centre.append(err[:, pos].cpu().numpy())
        interior.append(err[:, keep].mean(dim=1).cpu().numpy())
        edge.append(((err[:, 0] + err[:, -1]) / 2.0).cpu().numpy())
    return pd.DataFrame({"centre_mse": np.concatenate(centre), "interior_mse": np.concatenate(interior),
                         "edge_mse": np.concatenate(edge)})


def per_star(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-reduce the per-window measurements to one row per star."""
    frame = frame.copy()
    frame["centre_ratio"] = frame["centre_mse"] / frame["interior_mse"]
    frame["centre_excess"] = frame["centre_mse"] - frame["interior_mse"]
    frame["edge_ratio"] = frame["edge_mse"] / frame["interior_mse"]
    stars = frame.groupby("tic_id").median(numeric_only=True)
    stars["n_win"] = frame.groupby("tic_id").size()
    return stars.reset_index()


def cell_table(cells: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    """Cell-level centre and edge severity from the profile dump, one row per (cell, seed).

    Reduced per seed, never over seed-averaged profiles: the spike position moves seed to seed, so the
    seed-mean profile understates it (13.5x -> 5.3x on hann0p3_fbwd).

    Defaults to EVERY run in the dump, not just the four extension cells, because the cell-level
    within-arm correlation needs all five recipes of an arm to have any spread at all.
    """
    prof = load_profiles()
    lo, hi = INTERIOR
    rows = []
    for (cell, seed), g in prof.groupby(["cell", "seed"]):
        if cells and cell not in cells:
            continue
        if seeds and int(seed) not in seeds:
            continue
        m = g.set_index("pos")["mse"]
        interior = float(m.loc[lo:hi - 1].median())
        band = m.loc[CENTRE[0]:CENTRE[1] - 1]
        rows.append({"cell": cell, "seed": int(seed), "interior_mse": interior,
                     "centre_pos": int(band.idxmax()),
                     "centre_ratio": float(band.max() / interior),
                     "edge_ratio": float(max(m.loc[0], m.loc[WINDOW - 1]) / interior),
                     "arm": "fbwd" if cell.endswith("fbwd") else "off",
                     "recipe": cell.replace("exp07_", "").rsplit("_", 1)[0]})
    return pd.DataFrame(rows)


def star_probe_join(stars: pd.DataFrame, cell: str, seed: int) -> list[dict]:
    """Within-run rho between a star's centre severity and its own held-out probe score.

    Recipe, arm and seed are all fixed inside one call, so there is no design axis left to pool over --
    which is the whole point (F21). Scored against `mean` pooling, the readout every exp07 table uses.
    """
    if not STAR_SCORES.exists():
        return []
    scores = pd.read_parquet(STAR_SCORES)
    sel = scores[(scores["cell"] == cell) & (scores["seed"] == seed) & (scores["pooling"] == "mean")]
    out = []
    for task, g in sel.groupby("task"):
        joined = g.merge(stars[["tic_id", "centre_ratio", "centre_excess", "edge_ratio"]], on="tic_id")
        if len(joined) < 100:
            continue
        row = {"cell": cell, "seed": seed, "task": task, "n_stars": int(len(joined))}
        for driver in ["centre_ratio", "centre_excess", "edge_ratio"]:
            # correlate severity against the probe's score for the star; restricted to true negatives is
            # not appropriate here because the question is whether the artifact perturbs the score at all
            rho, p = spearmanr(joined[driver], joined["score"])
            row[f"rho_{driver}"] = float(rho)
            row[f"p_{driver}"] = float(p)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Centre-of-window reconstruction artifact and its probe cost.")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--limit", type=int, default=None, help="First N test stars only (smoke).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", default=None, help="Default: experiments/exp07_centre")
    args = ap.parse_args()

    prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "experiments" / "exp07_centre"

    cells = cell_table()  # every run in the profile dump: the within-arm rho needs all five recipes
    cells.to_csv(f"{prefix}_cells.csv", index=False)
    log.info(f"cell-level severity written: {len(cells)} runs over {cells['cell'].nunique()} cells")

    windows, tics = load_test_split(args.limit)
    log.info(f"test split: {windows.shape[0]} windows over {len(np.unique(tics))} stars, device={args.device}")

    star_rows, summary_rows = [], []
    for cell in args.cells:
        for seed in args.seeds:
            pos = spike_position(cell, seed)
            assert EDGE_GUARD <= pos < WINDOW - EDGE_GUARD, f"spike at {pos} is an edge, not an interior, position"
            model = load_model(cell, seed, args.device)
            measured = centre_excess(model, windows, pos, args.device, desc=f"{cell}:s{seed}@{pos}")
            frame = pd.concat([pd.DataFrame({"tic_id": tics}), measured], axis=1)
            stars = per_star(frame)
            stars.insert(0, "centre_pos", pos)
            stars.insert(0, "seed", seed)
            stars.insert(0, "cell", cell)
            star_rows.append(stars)
            summary_rows.extend(star_probe_join(stars, cell, seed))
            log.info(f"{cell} s{seed} pos {pos}: per-star median centre ratio "
                     f"{stars['centre_ratio'].median():.3f}, excess {stars['centre_excess'].median():+.4f}")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    pd.concat(star_rows, ignore_index=True).to_csv(f"{prefix}_stars.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    log.info(f"wrote {prefix}_{{cells,stars,summary}}.csv")

    print("\ncell-level severity (per seed, never seed-averaged):")
    print(cells.groupby("cell")[["centre_ratio", "edge_ratio"]].agg(["mean", "std"]).round(3).to_string())
    if not summary.empty:
        print("\nwithin-run rho(centre severity, star probe score), mean over seeds:")
        print(summary.groupby(["cell", "task"])[["rho_centre_ratio", "rho_centre_excess"]]
              .agg(["mean", "std"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
