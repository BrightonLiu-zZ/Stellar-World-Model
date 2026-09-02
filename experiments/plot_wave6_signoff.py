"""Wave-6 visual sign-off: reconstructions from the switch candidate beside the incumbent recipe.

G17-artifact, like G9-artifact before it, has TWO clauses and the number is only one of them. The
other is a human looking at a fresh random sample of reconstructed windows. That clause exists because
of F18: a taper was once reported as having REMOVED the impulse when it had only relocated it from the
window edge to the window centre, and the severity number of the day was blind to the move. So the
number never passes alone, and this script exists to make the other clause cheap to satisfy.

Two panels per figure:
  top     a fresh random sample of test windows, truth vs reconstruction, candidate and incumbent on
          the same axes and the same stars -- what a reader would actually see.
  bottom  the mean squared error at each of the 256 within-window positions, averaged over seeds. This
          is the severity metric's raw material: a planted impulse is a spike at one address, and the
          eye is a better judge of "is that a spike or is that noise" than a max-over-position ratio.

The sample is UNSEEDED on purpose (project rule): every re-run draws different stars, so a clean
verdict has to survive re-drawing rather than being a lucky panel. The seeded line is kept beside it,
commented, for when a specific figure has to be reproduced.

Run (repo root, swm env, PYTHONPATH=src):
    python experiments/plot_wave6_signoff.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so this runs from the repo root too
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot is imported)

from analyze_exp09_artifact import CELLS, WINDOW, load_model, load_test_windows, position_profile  # noqa: E402

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = "exp09_dpss_impulse_w0p025"
REF = "exp07_hann0p3_fbwd"
ARMS = [(CANDIDATE, "candidate  w=0.025 dpss+kurtosis", "tab:blue"),
        (REF, "incumbent  hann0p3_fbwd", "tab:red")]
N_SHOW = 6
PROFILE_SEEDS = [0, 1, 2, 3, 4, 5]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def reconstruct(cell: str, seed: int, x: np.ndarray) -> np.ndarray:
    """Reconstruct a small batch of windows through one run's frozen encoder/decoder."""
    model = load_model(cell, seed, CELLS[cell], DEVICE)
    chunk = torch.from_numpy(x).unsqueeze(-1).to(DEVICE)   # (n, 256, 1)
    mu, _ = model.encoder(chunk)
    return model.decoder(mu)[:, :, 0].float().cpu().numpy()  # (n, 256)


def main() -> int:
    windows = load_test_windows(None)
    rng = np.random.default_rng()
    # rng = np.random.default_rng(0)  # uncomment for a reproducible sample
    pick = rng.choice(windows.shape[0], size=N_SHOW, replace=False)
    sample = windows[pick]
    log.info(f"sampled windows {sorted(pick.tolist())} of {windows.shape[0]}")

    recon = {cell: reconstruct(cell, 0, sample) for cell, _, _ in ARMS}
    profiles = {}
    for cell, _, _ in ARMS:
        stack = [position_profile(load_model(cell, s, CELLS[cell], DEVICE), windows, DEVICE,
                                  desc=f"{cell}:s{s}") for s in PROFILE_SEEDS]
        profiles[cell] = np.mean(stack, axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i in range(N_SHOW):
        ax = axes.flat[i]
        ax.plot(sample[i], color="0.35", lw=1.4, label="flux (truth)")
        for cell, label, colour in ARMS:
            ax.plot(recon[cell][i], color=colour, lw=1.0, alpha=0.9, label=label)
        ax.set_title(f"test window {pick[i]}", fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    ax = axes.flat[6]
    for cell, label, colour in ARMS:
        ax.plot(profiles[cell], color=colour, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_title("mean MSE by within-window position, 6-seed mean (log scale)", fontsize=9)
    ax.set_xlabel("position in window")
    ax.legend(fontsize=7)

    ax = axes.flat[7]
    for cell, label, colour in ARMS:
        p = profiles[cell]
        ax.plot(p / np.median(p[16:240]), color=colour, lw=1.2, label=label)
    ax.axhline(1.5, color="0.5", ls=":", lw=1, label="G9-artifact 1.5x")
    ax.axhline(2.0, color="0.2", ls="--", lw=1, label="G17-artifact 2.0x")
    ax.set_title("the severity metric itself: profile / interior median", fontsize=9)
    ax.set_xlabel("position in window")
    ax.legend(fontsize=7)

    ax = axes.flat[8]
    residual = {cell: sample - recon[cell] for cell, _, _ in ARMS}
    for cell, label, colour in ARMS:
        ax.plot(np.abs(residual[cell]).mean(axis=0), color=colour, lw=1.2, label=label)
    ax.set_title(f"mean |residual| over the {N_SHOW} sampled windows", fontsize=9)
    ax.set_xlabel("position in window")
    ax.legend(fontsize=7)

    fig.suptitle("exp09 wave 6 -- visual sign-off clause of G17-artifact (fresh unseeded sample)",
                 fontsize=12)
    fig.tight_layout()
    out = ROOT / "experiments" / "exp09_wave6_signoff.png"
    fig.savefig(out, dpi=110)
    log.info(f"wrote {out}")

    interior = {c: float(np.median(p[16:240])) for c, p in profiles.items()}
    table = pd.DataFrame([{"cell": c, "interior_mse": interior[c],
                           "peak_ratio": float(profiles[c].max() / interior[c]),
                           "peak_pos": int(profiles[c].argmax())} for c, _, _ in ARMS])
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    raise SystemExit(main())
