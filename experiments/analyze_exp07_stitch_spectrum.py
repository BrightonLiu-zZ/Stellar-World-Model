"""Pre-exp08 CHK-2b: the window-stitch harmonic comb, promoted out of a scratchpad one-off.

Decoding a light curve in independent 256-cadence windows and laying the outputs end to end injects a
discontinuity at every boundary. A train of impulses spaced 256 cadences apart is a harmonic comb at
1/256 cycles per cadence -- 2.8125 c/d at the TESS 2-min cadence -- and its multiples. That comb is in
the reconstruction and not in the input, and per-sample MSE cannot see it.

The number this replaces came from one seed and 60 stars in a session scratchpad, and is quoted in a
figure that has already left the project (the Yue Ma update, 2026-08-05): 26.2x rect / 24.0x hann /
1.6x input. This script gives it a reproducible home at 6 seeds x >=200 strips x both arms.

Two things to read carefully when interpreting the output.

  The comb is NOT a probe cost. Probes read mu; they never see a reconstruction, and no term in the
  training objective spans a window boundary. The comb is therefore a property of a display/eval
  convention, cosmetic by construction for any claim about transfer. It is measured here as PAPER
  HYGIENE -- the number is already outside the project and needs to be reproducible -- and explicitly
  not as evidence that the artifact costs downstream score. That question is CHK-2a's per-star
  centre-severity correlation, and only that.

  The taper was never going to fix it. An impulse at position 0 and an impulse at position 130 produce
  the SAME comb frequencies and differ only in phase, so 26.2 -> 24.0 is "unchanged", not an 8% win.

  `untrained` is the control that decides what kind of thing the comb is (F16: an attribution needs an
  immune arm, not just a magnitude). An untrained decoder also emits independent per-window outputs. If
  it combs too, the comb is ARCHITECTURAL -- a consequence of windowed decoding that no aux-term
  redesign touches. If it does not, the comb is learned and belongs to the objective.

Writes experiments/exp07_stitch_spectrum_{strips,summary}.csv. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp07_stitch_spectrum.py
    python experiments/analyze_exp07_stitch_spectrum.py --n-strips 20 --seeds 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from swm.eval.skyline import _make_untrained
from swm.models import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp07_stitch")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
WINDOW = 256
CADENCE_MIN = 2.0
STITCH_CPD = 24.0 * 60.0 / (WINDOW * CADENCE_MIN)  # 2.8125 c/d, the window-repetition rate
DEFAULT_CELLS = ["exp07_hann0p3_fbwd", "exp07_comb0p3_fbwd", "exp07_hann0p3_off", "exp07_comb0p3_off"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5]
N_HARMONICS = 12
MIN_WINDOWS = 16  # a strip must span at least this many stitch periods for the comb to be resolvable


def load_model(cell: str, seed: int, device: str) -> WorldModel:
    """Rebuild one run's world model from its selected checkpoint (eval mode)."""
    path = ROOT / "experiments" / cell / "models" / f"B_seed{seed}" / "best_recon_aux.pt"
    assert path.exists(), f"missing checkpoint {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg, mc = ckpt["cfg"], ckpt["cfg"]["model"]
    model = WorldModel(in_ch=1, enc_channels=list(mc["enc_channels"]), kernel_size=int(mc["kernel_size"]),
                       z_dim=int(mc["z_dim"]), window=int(cfg["data"]["window"]),
                       gru_hidden=int(mc["gru_hidden"]), gru_layers=int(mc["gru_layers"]),
                       dyn_mode=str(mc.get("dyn_mode", "fwd"))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def make_untrained(reference_cell: str, device: str, seed: int) -> WorldModel:
    """Capacity-matched random-init model, geometry read off a trained checkpoint of the same sweep."""
    ckpt = torch.load(ROOT / "experiments" / reference_cell / "models" / "B_seed0" / "best_recon_aux.pt",
                      map_location="cpu", weights_only=False)
    mc = ckpt["cfg"]["model"]
    return _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                           int(ckpt["cfg"]["data"]["window"]), int(mc["gru_hidden"]), int(mc["gru_layers"]),
                           device, seed=seed)


def load_strips(n_strips: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Contiguous within-segment flux strips from the packed test split, one per sampled segment.

    A packed segment is gap-guarded and uniformly sampled by construction (ADR-0003), so laying its
    windows end to end reproduces exactly the strip the reconstruction figure shows -- with no segment
    boundary inside it, which would put a real discontinuity where we are measuring a spurious one.
    """
    index = pd.read_parquet(PACKED / "test_index.parquet")
    index = index[index["n_win"] >= MIN_WINDOWS]
    total = int(pd.read_parquet(PACKED / "test_index.parquet")["n_win"].sum())
    memmap = np.memmap(PACKED / "test_windows.dat", dtype=np.float32, mode="r", shape=(total, WINDOW))
    pick = rng.permutation(len(index))[:n_strips]
    strips = []
    for row in index.iloc[pick].itertuples():
        block = np.array(memmap[int(row.row_start):int(row.row_start) + int(row.n_win)], dtype=np.float32)
        strips.append(block.reshape(-1))  # (n_win * 256,) contiguous cadences
    return strips


@torch.no_grad()
def reconstruct(strip: np.ndarray, model: WorldModel, device: str) -> np.ndarray:
    """Encode/decode in independent 256-cadence windows, then lay the windows back end to end."""
    win = strip.reshape(-1, WINDOW, 1)
    x = torch.from_numpy(np.ascontiguousarray(win)).float().to(device)
    mu, _ = model.encoder(x)
    return model.decoder(mu)[:, :, 0].cpu().numpy().reshape(-1)


def comb_contrast(y: np.ndarray) -> float:
    """On-harmonic over off-harmonic amplitude, in index space where harmonics land exactly on a bin.

    Measured in cycles/cadence rather than cycles/day: the stitch harmonics sit at exactly k/256, which
    is an integer bin of an N-sample FFT whenever N is a multiple of 256. That removes the tolerance
    band a frequency-space version needs and with it the arbitrary choice of its halfwidth.
    """
    y = y - y.mean()
    n = (len(y) // WINDOW) * WINDOW
    amp = np.abs(np.fft.rfft(y[:n])) * 2.0 / n
    step = n // WINDOW  # bins per stitch harmonic
    on, off = [], []
    for h in range(1, N_HARMONICS + 1):
        centre = h * step
        if centre + 3 * step >= len(amp):
            break
        on.append(amp[max(centre - 1, 0):centre + 2].max())
        neighbourhood = np.r_[amp[centre - 3 * step:centre - step], amp[centre + step:centre + 3 * step]]
        off.append(np.median(neighbourhood))
    return float(np.median(on) / np.median(off)) if on else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description="Window-stitch harmonic comb contrast, per run.")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--n-strips", type=int, default=200, help="Test-split segments sampled (>=200 for the record).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-seed", type=int, default=0, help="Strip sample RNG; fixed, this is a measurement.")
    ap.add_argument("--out-prefix", default=None, help="Default: experiments/exp07_stitch_spectrum")
    args = ap.parse_args()

    prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "experiments" / "exp07_stitch_spectrum"
    # fixed sample seed on purpose: this is a measurement over a star population, not a display sample,
    # so the CLAUDE.md fresh-sample rule (which governs plotted examples) does not apply.
    strips = load_strips(args.n_strips, np.random.default_rng(args.sample_seed))
    log.info(f"{len(strips)} strips, {np.median([len(s) for s in strips]):.0f} median cadences, device={args.device}")

    rows = [{"arm_kind": "input", "cell": "input", "seed": -1, "strip": i, "contrast": comb_contrast(s)}
            for i, s in enumerate(strips)]

    runs: list[tuple[str, str, int]] = [("untrained", "untrained", s) for s in args.seeds]
    runs += [("trained", cell, seed) for cell in args.cells for seed in args.seeds]
    for kind, cell, seed in tqdm(runs, desc="runs"):
        model = (make_untrained(args.cells[0], args.device, seed) if kind == "untrained"
                 else load_model(cell, seed, args.device))
        for i, strip in enumerate(strips):
            rows.append({"arm_kind": kind, "cell": cell, "seed": seed, "strip": i,
                         "contrast": comb_contrast(reconstruct(strip, model, args.device))})
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    strips_frame = pd.DataFrame(rows)
    strips_frame.to_csv(f"{prefix}_strips.csv", index=False)
    summary = (strips_frame.groupby(["arm_kind", "cell", "seed"])["contrast"]
               .agg(median="median", p90=lambda s: s.quantile(0.90), n="size").reset_index())
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    log.info(f"wrote {prefix}_{{strips,summary}}.csv")

    print("\nharmonic-comb contrast at the 256-cadence stitch rate (2.8125 c/d), median over strips:")
    print(summary.groupby(["arm_kind", "cell"])["median"].agg(["mean", "std", "size"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
