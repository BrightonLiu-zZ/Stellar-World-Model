"""exp09 G9-artifact: does closing the loss-hacking channel actually remove the within-window impulse?

WHAT IS MEASURED, AND WHY IT IS NOT THE exp07 ESTIMATOR.

exp07 measured the impulse at the window EDGE (positions 0 / 255) and pronounced `hann0p3` fixed at
1.15x. F27 then found the taper had RELOCATED the impulse to the window CENTRE (13.5 +/- 6.0), where
the Hann weight is maximal -- the fix was positional, not objective-level. `analyze_exp07_centre_
artifact.py` therefore searches a CENTRE band (112, 145), chosen because that is where a Hann taper
peaks.

Neither estimator is right for exp09. `aux_dpss` trains under a DPSS taper FAMILY whose weight profile
peaks nowhere in particular, and `aux_impulse_pen` penalises impulsiveness without reference to
position at all. Assuming any band would repeat exactly the mistake F27 exposed. So G9-artifact is
defined as the MAX OVER ALL 256 POSITIONS:

    max_ratio = max_pos( mse[pos] ) / median( mse[INTERIOR] )

against the same interior reference band exp07 uses, so the numbers stay comparable. The centre-band
and edge ratios are reported alongside, and the exp07 reference cells are RECOMPUTED here rather than
read from the profile dump, so the published 13.5x / 1.15x act as a protocol gate on this script
before any exp09 number is believed.

GATE (pre-registered, experiments/configs/exp09_loss_exploit_ladder.yaml):
    G9-artifact PASSES iff max_ratio <= 1.5 at full aux pressure over 6 seeds
    AND the user signs off visually on a fresh random sample of reconstructions. The number never
    passes alone -- F18 was retracted precisely because a summary statistic said "fixed" while the
    impulse had only moved.

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp09_artifact.py
    python experiments/analyze_exp09_artifact.py --limit 200          # smoke
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.models import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp09_artifact")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
WINDOW = 256
INTERIOR = (16, 240)  # exp07's reference band, kept identical so ratios stay comparable
CENTRE = (112, 145)   # exp07/F27 search band, REPORTED for comparison, never the gate
BATCH = 512
SEEDS = [0, 1, 2, 3, 4, 5]
# exp09 cells ship best_recon_only.pt (the aux-independent primary); exp07 reference cells predate it.
CELLS = {
    "exp09_aux_none": "best_recon_only",
    "exp09_aux_dpss": "best_recon_only",
    "exp09_aux_impulse_pen": "best_recon_only",
    "exp09_aux_clip": "best_recon_only",
    "exp09_aux_dpss_impulse": "best_recon_only",   # wave 2: dpss AND the kurtosis penalty together
    # wave 3: the same recipe at lower recon_aux.weight. best_recon_only is not optional here - the
    # sweep axis IS the aux weight, so best_recon_aux would compare cells at checkpoints chosen by four
    # different rules (decision A3).
    "exp09_dpss_impulse_w0p0125": "best_recon_only",   # wave 4: bisects 0 < w < 0.05, where wave 3
    "exp09_dpss_impulse_w0p025": "best_recon_only",    # put both the gate crossing and the probe cliff
    "exp09_dpss_impulse_w0p05": "best_recon_only",
    "exp09_dpss_impulse_w0p10": "best_recon_only",
    "exp09_dpss_impulse_w0p20": "best_recon_only",
    "exp07_hann0p3_fbwd": "best_recon_aux",   # protocol gate: must reproduce centre ~13.5, edge ~1.15
    "exp07_comb0p3_fbwd": "best_recon_aux",   # rectangular contrast: impulse at the EDGE (~31x)
}
PUBLISHED = {"exp07_hann0p3_fbwd": {"centre": 13.5, "centre_tol": 6.0, "edge": 1.15, "edge_tol": 0.35}}


def load_model(cell: str, seed: int, ckpt: str, device: str) -> WorldModel:
    """Rebuild one run's world model from its selected checkpoint (eval mode)."""
    path = ROOT / "experiments" / cell / "models" / f"B_seed{seed}" / f"{ckpt}.pt"
    assert path.exists(), f"missing checkpoint {path}"
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg, mc = blob["cfg"], blob["cfg"]["model"]
    model = WorldModel(in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
                       z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
                       gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
                       dyn_mode=str(mc.get("dyn_mode", "fwd"))).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_test_windows(limit: int | None) -> np.ndarray:
    """Every packed test-split window, in stored order."""
    index = pd.read_parquet(PACKED / "test_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(PACKED / "test_windows.dat", dtype=np.float32, mode="r", shape=(total, WINDOW))
    if limit is not None:
        index = index[index["tic_id"].isin(sorted(index["tic_id"].unique())[:limit])]
    rows = np.concatenate([np.arange(int(a), int(a) + int(n))
                           for a, n in zip(index["row_start"], index["n_win"])])
    return np.array(memmap[rows], dtype=np.float32)


@torch.no_grad()
def position_profile(model: WorldModel, x: np.ndarray, device: str, desc: str) -> np.ndarray:
    """Mean squared reconstruction error at each of the 256 within-window positions."""
    total = np.zeros(WINDOW, dtype=np.float64)
    n = 0
    for start in tqdm(range(0, x.shape[0], BATCH), desc=desc, total=(x.shape[0] + BATCH - 1) // BATCH):
        chunk = torch.from_numpy(x[start:start + BATCH]).unsqueeze(-1).to(device)  # (b, 256, 1)
        mu, _ = model.encoder(chunk)
        recon = model.decoder(mu)[:, :, 0]                                          # (b, 256)
        err = (recon - chunk[:, :, 0]) ** 2                                         # (b, 256)
        total += err.sum(dim=0).double().cpu().numpy()
        n += chunk.shape[0]
    return total / max(n, 1)  # (256,)


def ratios(profile: np.ndarray) -> dict:
    """Impulse severity against the interior reference level, three framings."""
    lo, hi = INTERIOR
    interior = float(np.median(profile[lo:hi]))
    return {
        "interior_mse": interior,
        "max_ratio": float(profile.max() / interior),          # THE GATE: max over ALL positions
        "max_pos": int(profile.argmax()),
        "centre_ratio": float(profile[CENTRE[0]:CENTRE[1]].max() / interior),  # exp07/F27 comparison
        "edge_ratio": float(max(profile[0], profile[-1]) / interior),          # exp07 edge comparison
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="exp09 G9-artifact: max-over-position impulse severity.")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--limit", type=int, default=None, help="First N test stars only (smoke).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", default=str(ROOT / "experiments" / "exp09_impulse"))
    args = ap.parse_args()

    windows = load_test_windows(args.limit)
    log.info(f"test windows: {windows.shape[0]} x {WINDOW}, device={args.device}")

    rows, profiles = [], []
    for cell in args.cells:
        ckpt = CELLS.get(cell, "best_recon_aux")
        for seed in args.seeds:
            model = load_model(cell, seed, ckpt, args.device)
            profile = position_profile(model, windows, args.device, desc=f"{cell}:s{seed}")
            rows.append({"cell": cell, "seed": seed, "ckpt": ckpt, **ratios(profile)})
            profiles.append(pd.DataFrame({"cell": cell, "seed": seed,
                                          "pos": np.arange(WINDOW), "mse": profile}))
            log.info(f"  {cell} s{seed}: max {rows[-1]['max_ratio']:.2f}x @pos {rows[-1]['max_pos']} | "
                     f"centre {rows[-1]['centre_ratio']:.2f}x | edge {rows[-1]['edge_ratio']:.2f}x")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    runs = pd.DataFrame(rows)
    runs.to_csv(f"{args.out_prefix}_runs.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_parquet(f"{args.out_prefix}_profile.parquet", index=False)

    summary = runs.groupby("cell").agg(
        max_ratio=("max_ratio", "mean"), max_sd=("max_ratio", "std"),
        centre_ratio=("centre_ratio", "mean"), edge_ratio=("edge_ratio", "mean"),
        max_pos_min=("max_pos", "min"), max_pos_max=("max_pos", "max"), n=("seed", "size"))
    summary["G9_artifact_pass"] = summary["max_ratio"] <= 1.5
    summary.to_csv(f"{args.out_prefix}_summary.csv")

    print("\n=== PROTOCOL GATE: recomputed exp07 reference vs published")
    for cell, pub in PUBLISHED.items():
        if cell not in summary.index:
            continue
        got = summary.loc[cell]
        ok_c = abs(got["centre_ratio"] - pub["centre"]) <= pub["centre_tol"]
        ok_e = abs(got["edge_ratio"] - pub["edge"]) <= pub["edge_tol"]
        print(f"  {cell}: centre {got['centre_ratio']:.2f} vs published {pub['centre']} -> {'OK' if ok_c else 'DRIFT'}"
              f" | edge {got['edge_ratio']:.2f} vs {pub['edge']} -> {'OK' if ok_e else 'DRIFT'}")

    print("\n=== G9-artifact: max-over-position impulse ratio (gate <= 1.5x, 6 seeds)")
    print(summary.round(3).to_string())
    print("\nNOTE: the number is necessary, NOT sufficient. G9-artifact also requires visual sign-off on a"
          "\nfresh random sample of reconstructions (src/notebooks/exp09_diagnostics.ipynb).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
