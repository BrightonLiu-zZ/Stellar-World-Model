"""exp09: are aux_dpss's patch-scale bumps a PURCHASE of spectral power, or honest residual error?

THE QUESTION. The visual sign-off found that `aux_dpss` has no single tall spike but ~8 low, wide
bumps, and an FFT of its per-position error profile puts leading power at period 16 - the decoder's
bottleneck-patch scale (4x ConvTranspose1d(k=4,s=2) upsamples a 16-position bottleneck by 16x).
G9-artifact reads 5.28x and calls that a fail, but the gate cannot distinguish:

    (i)  the decoder still buying log-PSD power, merely DISTRIBUTED instead of concentrated
    (ii) honest reconstruction residual that was previously MASKED by a dominant impulse

Those have opposite consequences. Under (i) DPSS only redistributed the exploit. Under (ii) DPSS
closed the channel and the 5.28x is measuring decoder granularity, i.e. the 1.5x bar is the wrong
instrument for this cell.

THE TEST is exp07 pre-check C1, generalised from the window edge to wherever the error actually
concentrates. C1 forced the EDGE residual to zero and found log-PSD ROSE +237..+348% while time-MSE
FELL 11-17% -- the signature of a purchase: the spike was not fidelity, it was spectral currency.

    ablate(positions):  recon[positions] := target[positions]      (residual forced to zero)
    then recompute the cell's OWN training objective terms.

    time-MSE MUST fall (the residual is smaller by construction) - that is a sanity check, not a result.
    log-PSD is the readout:
        rises sharply at bumps, and much more than at random positions -> PURCHASE      (i)
        flat / falls, or no better than random                        -> HONEST RESIDUAL (ii)

CONTROLS (both required; the C1 lesson is that a single-position measurement proves nothing):
    positive: exp07_hann0p3_fbwd -- its centre spike is a known purchase. If the method does not
              light up here, the method is broken and the aux_dpss null means nothing.
    negative: RANDOM interior positions, same count, same cell, several draws. This is C1's
              location-asymmetry control (it measured 27-38x between edge and interior).

Every cell is scored under ITS OWN training taper, because the question is whether it was buying
power in the term it was actually optimising.

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp09_bump_ablation.py
    python experiments/analyze_exp09_bump_ablation.py --limit 400 --seeds 0     # smoke
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from swm.models import WorldModel
from swm.train.losses import hf_time_loss, spectral_recon_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp09_bump")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
WINDOW = 256
BATCH = 512
# Ablation budget, in cadences. C1 ablated ~2 positions; ablating 40 of 256 (15.6% of the window)
# perturbs the spectrum so hard that the random control alone moves log-PSD by +63..+88%, which
# compresses every asymmetry toward 1 and destroys the test's dynamic range. The budget is therefore
# SWEPT, and the asymmetry is read where the control is still small.
BUDGETS = [2, 6, 20, 40]
HALF_WIDTH = 0         # one cadence per selected peak, so `budget` is literally the position count
N_RANDOM_DRAWS = 5     # random-control draws, averaged
SEEDS = [0, 1, 2, 3, 4, 5]
# cell -> (checkpoint, training taper) : each cell is judged under the term it optimised
CELLS = {
    "exp07_comb0p3_fbwd": ("best_recon_aux", "none"),
    "exp07_hann0p3_fbwd": ("best_recon_aux", "hann"),
    "exp09_aux_dpss": ("best_recon_only", "dpss"),
    "exp09_aux_impulse_pen": ("best_recon_only", "hann"),
    "exp09_aux_clip": ("best_recon_only", "hann"),
    "exp09_aux_none": ("best_recon_only", "hann"),
    # wave 2: the taper it optimised under is dpss, so that is the taper it is judged under - the
    # kurtosis penalty acts on the time-domain residual and does not change which taper is fair here.
    "exp09_aux_dpss_impulse": ("best_recon_only", "dpss"),
    # wave 3: same recipe, lower recon_aux.weight. Same taper, so the same term is recomputed - the
    # weight scales the objective but not which spectrum the purchase is measured in.
    "exp09_dpss_impulse_w0p0125": ("best_recon_only", "dpss"),   # wave 4
    "exp09_dpss_impulse_w0p025": ("best_recon_only", "dpss"),
    "exp09_dpss_impulse_w0p05": ("best_recon_only", "dpss"),
    "exp09_dpss_impulse_w0p10": ("best_recon_only", "dpss"),
    "exp09_dpss_impulse_w0p20": ("best_recon_only", "dpss"),
}


def load_model(cell: str, seed: int, ckpt: str, device: str) -> WorldModel:
    blob = torch.load(ROOT / "experiments" / cell / "models" / f"B_seed{seed}" / f"{ckpt}.pt",
                      map_location=device, weights_only=False)
    mc, cfg = blob["cfg"]["model"], blob["cfg"]
    model = WorldModel(in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
                       z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
                       gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
                       dyn_mode=str(mc.get("dyn_mode", "fwd"))).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def load_windows(limit: int | None) -> np.ndarray:
    index = pd.read_parquet(PACKED / "test_index.parquet")
    total = int(index["n_win"].sum())
    memmap = np.memmap(PACKED / "test_windows.dat", dtype=np.float32, mode="r", shape=(total, WINDOW))
    rows = np.concatenate([np.arange(int(a), int(a) + int(n))
                           for a, n in zip(index["row_start"], index["n_win"])])
    if limit:
        rows = rows[:limit]
    return np.array(memmap[rows], dtype=np.float32)


@torch.no_grad()
def reconstruct(model: WorldModel, x: np.ndarray, device: str, desc: str) -> np.ndarray:
    out = []
    for start in tqdm(range(0, x.shape[0], BATCH), desc=desc, leave=False,
                      total=(x.shape[0] + BATCH - 1) // BATCH):
        chunk = torch.from_numpy(x[start:start + BATCH]).unsqueeze(-1).to(device)   # (b, 256, 1)
        mu, _ = model.encoder(chunk)
        out.append(model.decoder(mu)[:, :, 0].cpu().numpy())                         # (b, 256)
    return np.concatenate(out)


def peak_positions(profile: np.ndarray, budget: int, half: int) -> np.ndarray:
    """Highest-error positions, taken as prominence-ranked local maxima and dilated to +/- half.

    Ranking by prominence rather than raw height is what lets one procedure serve both a cell with a
    single tall spike (hann: one peak absorbs the budget) and a cell with many low bumps (dpss).

    THE PROFILE IS PADDED FIRST. scipy.find_peaks cannot return an endpoint, because index 0 and
    index 255 have only one neighbour -- so an unpadded call silently skips the window-EDGE impulse,
    which is the entire artifact for the rectangular recipe. That bug made comb0p3 (the strongest
    known purchase in the project, C1's 27-38x) score BELOW its own random control. Padding with a
    below-minimum sentinel makes both endpoints eligible.
    """
    pad = np.concatenate([[profile.min() - 1.0], profile, [profile.min() - 1.0]])
    peaks, props = find_peaks(pad, prominence=0.0)
    peaks = peaks - 1                                     # undo the pad offset
    order = peaks[np.argsort(props["prominences"])[::-1]]
    chosen: list[int] = []
    for p in order:
        for q in range(max(0, p - half), min(WINDOW, p + half + 1)):
            if q not in chosen:
                chosen.append(q)
        if len(chosen) >= budget:
            break
    return np.array(sorted(chosen[:budget]))


def objective_terms(recon: np.ndarray, target: np.ndarray, window_fn: str, device: str) -> dict:
    """The cell's own loss terms on a (n, 256) pair, batched to bound memory."""
    tot = {"log_psd": 0.0, "hf_time": 0.0, "time_mse": 0.0}
    n = 0
    for start in range(0, recon.shape[0], BATCH):
        r = torch.from_numpy(recon[start:start + BATCH]).to(device)[None, :, :, None]  # (1, b, 256, 1)
        t = torch.from_numpy(target[start:start + BATCH]).to(device)[None, :, :, None]
        b = r.shape[1]
        tot["log_psd"] += float(spectral_recon_loss(r, t, window_fn=window_fn)) * b
        tot["hf_time"] += float(hf_time_loss(r, t)) * b
        tot["time_mse"] += float(torch.mean((r - t) ** 2)) * b
        n += b
    return {k: v / n for k, v in tot.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the residual structure a spectral purchase?")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--limit", type=int, default=8000, help="test windows to score")
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "exp09_bump_ablation.csv"))
    args = ap.parse_args()

    x = load_windows(args.limit)
    log.info(f"scoring {x.shape[0]} test windows on {args.device}")
    rng = np.random.default_rng(0)   # controls only; the ablated positions are measured, not sampled
    rows = []

    for cell in args.cells:
        ckpt, window_fn = CELLS[cell]
        for seed in args.seeds:
            model = load_model(cell, seed, ckpt, args.device)
            recon = reconstruct(model, x, args.device, f"{cell}:s{seed}")
            profile = ((recon - x) ** 2).mean(axis=0)                    # (256,) per-position MSE
            base = objective_terms(recon, x, window_fn, args.device)

            for budget in args.budgets:
                bumps = peak_positions(profile, budget, HALF_WIDTH)
                ab = recon.copy()
                ab[:, bumps] = x[:, bumps]                                # force residual to zero
                hit = objective_terms(ab, x, window_fn, args.device)

                ctrl = []
                interior = np.setdiff1d(np.arange(1, WINDOW - 1), bumps)
                for _ in range(N_RANDOM_DRAWS):
                    pos = rng.choice(interior, size=budget, replace=False)
                    rc = recon.copy()
                    rc[:, pos] = x[:, pos]
                    ctrl.append(objective_terms(rc, x, window_fn, args.device)["log_psd"])
                ctrl_mean = float(np.mean(ctrl))

                rows.append({
                    "cell": cell, "seed": seed, "window_fn": window_fn, "budget": budget,
                    "positions": ",".join(map(str, bumps)),
                    "edge_selected": bool(0 in bumps or WINDOW - 1 in bumps),
                    # absolutes, not just deltas: `log_psd_ablated` on hann0p3_fbwd IS the constant
                    # roadmap task Y9-F needs -- the spectral loss an impulse-free reconstruction
                    # achieves, i.e. the floor below which further reduction is only purchasable by
                    # planting an impulse. Fills constants.spectral_floor.value for the aux_clip cell.
                    "base_log_psd": base["log_psd"],
                    "log_psd_ablated": hit["log_psd"],
                    "base_time_mse": base["time_mse"],
                    "d_log_psd_bump_pct": 100 * (hit["log_psd"] - base["log_psd"]) / abs(base["log_psd"]),
                    "d_log_psd_rand_pct": 100 * (ctrl_mean - base["log_psd"]) / abs(base["log_psd"]),
                    "d_time_mse_bump_pct": 100 * (hit["time_mse"] - base["time_mse"]) / base["time_mse"],
                    "d_hf_bump_pct": 100 * (hit["hf_time"] - base["hf_time"]) / base["hf_time"],
                })
            log.info(f"  {cell} s{seed}: " + " | ".join(
                f"b{r['budget']} {r['d_log_psd_bump_pct']:+.0f}%/{r['d_log_psd_rand_pct']:+.0f}%"
                for r in rows[-len(args.budgets):]))
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    agg = out.groupby(["cell", "budget"]).agg(
        log_psd_bump=("d_log_psd_bump_pct", "mean"), bump_sd=("d_log_psd_bump_pct", "std"),
        log_psd_rand=("d_log_psd_rand_pct", "mean"), time_mse=("d_time_mse_bump_pct", "mean"),
        edge=("edge_selected", "mean"))
    agg["asymmetry"] = agg.log_psd_bump / agg.log_psd_rand
    order = [c for c in ["exp07_comb0p3_fbwd", "exp07_hann0p3_fbwd", "exp09_aux_impulse_pen",
                         "exp09_aux_dpss", "exp09_aux_none"] if c in out.cell.unique()]

    print("\n=== ablate the top-error positions: does the cell's OWN log-PSD term get WORSE? (%)")
    print(agg.loc[order].round(2).to_string())
    print("\n=== ASYMMETRY (bump / random) by budget -- read it where the random control is still small")
    piv = agg.reset_index().pivot(index="cell", columns="budget", values="asymmetry")
    print(piv.loc[order].round(2).to_string())
    print("\n'edge' = fraction of seeds whose selected positions include a window endpoint;")
    print("comb0p3 MUST be 1.0 or the endpoint-padding fix has regressed.")
    print("readout: asymmetry >> 1 => the structure was a PURCHASE; ~1 => honest residual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
