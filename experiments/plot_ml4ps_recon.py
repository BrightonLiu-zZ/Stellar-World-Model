"""Appendix figure: what the two spectral-auxiliary recipes actually reconstruct.

The paper says the reconstruction objective develops a localised within-window artifact and that the
Hann taper RELOCATED it rather than removing it (F27). Until now that claim had no picture. This
script draws one:

  rows 1-2   two stars, input flux against each recipe's reconstruction, one column per star,
             BOTH ROWS OF A COLUMN ON ONE y-SCALE so the two decoders are directly comparable;
  row 3      the position-resolved reconstruction error inside a window, divided by the interior
             median, mean over six pre-training seeds -- the same quantity the pre-registered
             artifact gate maxima over, read off experiments/exp07_edge_profiles_6seed.parquet.

Both recipes are the forward+backward dynamics arm, so the only difference between the two rows is
the taper on the auxiliary log-power-spectrum term: `comb0p3` computes it on a rectangular window,
`hann0p3` (the shipped recipe) on a Hann-tapered one.

The two display stars are drawn at random -- one pulsator, one quiet star -- from the test split of
the first pool, which no probe or pre-training run has fitted. `--seed` fixes the draw so the figure
regenerates identically; the drawn TICs are written beside the PDF.

Run (swm env, from the repo root):
    PYTHONPATH=src python experiments/plot_ml4ps_recon.py
    PYTHONPATH=src python experiments/plot_ml4ps_recon.py --seed 3 --png   # look at another pair
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from swm.models import WorldModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("plot_recon")

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "experiments"
SEQ_DIR = ROOT / "processed" / "sequences"
PROFILES = EXPS / "exp07_edge_profiles_6seed.parquet"

WINDOW = 256
N_WINDOWS_SHOWN = 3          # 768 cadences ~ 25.6 h; two interior window seams are visible
INTERIOR = (16, 240)         # the reference band the exp07/exp09 ratio is measured against
CADENCE_DAYS = 2.0 / 1440.0
SEEDS = [0, 1, 2, 3, 4, 5]
PANEL_SEED = 0
CKPT = "best_recon_aux"      # the checkpoint every exp07 number in the paper is read from

ARMS = {
    "exp07_comb0p3_fbwd": ("rectangular window", "tab:red"),
    "exp07_hann0p3_fbwd": ("Hann taper (shipped)", "tab:blue"),
}
_NPZ_RE = re.compile(r"^TIC(\d+)_s\d+_seg\d+_run\d+\.npz$")


def load_model(cell: str, seed: int, device: str) -> WorldModel:
    """Rebuild one run's world model from its selected checkpoint, in eval mode.

    strict=False: a fwd_bwd checkpoint carries a backward-GRU head the plain constructor does not
    build. Only the encoder and decoder are used here and both load exactly; a silently missing
    encoder weight would instead surface as a shape error, not as a quiet zero.
    """
    path = EXPS / cell / "models" / f"B_seed{seed}" / f"{CKPT}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = WorldModel(in_ch=c["model"]["in_ch"], enc_channels=c["model"]["enc_channels"],
                       kernel_size=c["model"]["kernel_size"], z_dim=c["model"]["z_dim"],
                       window=int(c["data"]["window"]), gru_hidden=c["model"]["gru_hidden"],
                       gru_layers=c["model"]["gru_layers"]).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model


def first_segment_files() -> dict[int, str]:
    """One scandir pass over the corpus: the alphabetically first stored segment of each star."""
    first: dict[int, str] = {}
    with os.scandir(SEQ_DIR) as it:
        for entry in it:
            m = _NPZ_RE.match(entry.name)
            if m is None:
                continue
            tic = int(m.group(1))
            if tic not in first or entry.name < first[tic]:
                first[tic] = entry.name
    return first


def load_strip(tic: int, files: dict[int, str]) -> np.ndarray:
    """The star's first stored segment, flattened back into one continuous flux strip.

    The corpus stores segments as 1024-cadence rows; the models were trained on 256-cadence windows,
    so the strip is re-cut below. Flux is already MAD-normalised per segment by the build pipeline.
    """
    payload = np.load(SEQ_DIR / files[tic])
    return payload["windows"].reshape(-1).astype(np.float32)


def reconstruct(flux: np.ndarray, model: WorldModel, device: str) -> np.ndarray:
    """Encode and decode a strip in the 256-cadence windows the model was trained on.

    Each window goes through independently and the outputs are laid back end to end, which is what
    makes a within-window defect visible as a repeating feature of the strip.
    """
    usable = (len(flux) // WINDOW) * WINDOW
    windows = np.ascontiguousarray(flux[:usable].reshape(-1, WINDOW, 1))   # (n_win, 256, 1)
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        mu, _ = model.encoder(x)
        recon = model.decoder(mu)                                          # (n_win, 256, 1)
    return recon[:, :, 0].cpu().numpy().reshape(-1)


def position_profiles() -> dict[str, np.ndarray]:
    """Per-seed, per-position reconstruction MSE divided by that seed's interior median.

    Normalising each seed by its own interior median before anything else is what makes the curves
    comparable across recipes: the two arms sit at different absolute reconstruction levels, and the
    gate is a ratio, not a level. Seeds are returned unaveraged because the Hann peak sits at a
    different position in every seed, so a seed-mean curve is materially lower at the peak than the
    per-seed maximum the gate is read on -- averaging first would print a number the gate disagrees
    with.
    """
    if not PROFILES.exists():
        raise FileNotFoundError(f"missing {PROFILES}; run experiments/analyze_exp07_diagnostics.py")
    frame = pd.read_parquet(PROFILES)
    out = {}
    for cell in ARMS:
        stack = []
        for _, group in frame[frame["cell"] == cell].groupby("seed"):
            profile = group.sort_values("pos")["mse"].to_numpy()
            stack.append(profile / np.median(profile[INTERIOR[0]:INTERIOR[1]]))
        ratios = np.vstack(stack)
        if len(ratios) != len(SEEDS):
            raise AssertionError(f"{cell}: {len(ratios)} seeds in the profile table, expected {len(SEEDS)}")
        out[cell] = ratios
    return out


def pick_stars(rng: np.random.Generator, available: set[int]) -> list[tuple[int, str]]:
    """One pulsator and one quiet star from the first pool's test split, in that order.

    The test split is the right place to draw from: no probe was fitted on it and the encoder never
    saw it, so nothing about these two curves is a training-set memory.
    """
    pool = pd.read_parquet(ROOT / "processed" / "subset" / "subset_tics.parquet")
    picks = []
    for stratum in ["pulsating", "quiet"]:
        candidates = pool[(pool["split"] == "test") & (pool["stratum"] == stratum)]["tic_id"]
        candidates = np.array(sorted(set(candidates.astype(int)) & available))
        picks.append((int(rng.choice(candidates)), stratum))
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description="Appendix reconstruction figure: rectangular vs Hann.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the two display stars.")
    ap.add_argument("--out", default="paper/figures/figD_recon.pdf")
    ap.add_argument("--png", action="store_true", help="Also write a PNG copy for on-screen review.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = first_segment_files()
    rng = np.random.default_rng(args.seed)
    stars = pick_stars(rng, set(files))
    log.info(f"display stars: {stars}")

    models = {cell: load_model(cell, PANEL_SEED, args.device) for cell in ARMS}
    profiles = position_profiles()

    n_show = N_WINDOWS_SHOWN * WINDOW
    hours = np.arange(n_show) * CADENCE_DAYS * 24.0

    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.45, wspace=0.18)

    provenance = []
    for col, (tic, stratum) in enumerate(stars):
        flux = load_strip(tic, files)[:n_show]
        recons = {cell: reconstruct(flux, model, args.device)[:n_show] for cell, model in models.items()}
        span = np.concatenate([flux] + [r for r in recons.values()])
        lo, hi = float(span.min()), float(span.max())
        pad = 0.06 * (hi - lo)
        for row, (cell, (label, colour)) in enumerate(ARMS.items()):
            ax = fig.add_subplot(grid[row, col])
            ax.plot(hours, flux, color="0.65", lw=0.7, label="input")
            ax.plot(hours, recons[cell], color=colour, lw=0.8, label="reconstruction")
            for boundary in range(WINDOW, n_show, WINDOW):
                ax.axvline(hours[boundary], color="0.4", ls=":", lw=0.6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(hours[0], hours[-1])
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"TIC {tic} ({stratum})", fontsize=8)
            else:
                ax.set_xlabel("hours from segment start", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\nflux (MAD)", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
            provenance.append({"tic_id": tic, "stratum": stratum, "cell": cell,
                               "y_low": lo - pad, "y_high": hi + pad,
                               "cadences_shown": n_show})

    ax = fig.add_subplot(grid[2, :])
    positions = np.arange(WINDOW)
    for cell, (label, colour) in ARMS.items():
        ratios = profiles[cell]
        for row in ratios:
            ax.plot(positions, row, color=colour, lw=0.5, alpha=0.45)
        ax.plot(positions, ratios.mean(axis=0), color=colour, lw=1.3, label=label)
    ax.axhline(1.0, color="0.4", ls=":", lw=0.6)
    ax.axhline(1.5, color="k", ls="--", lw=0.8)
    ax.annotate("pre-registered gate, 1.5x", xy=(40, 1.5), xytext=(40, 1.72),
                fontsize=6.5, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.55, 60)
    ax.set_xlim(0, WINDOW - 1)
    ax.set_xlabel("position within the 256-cadence window", fontsize=8)
    ax.set_ylabel("reconstruction MSE\n/ interior median", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if args.png:
        fig.savefig(out_path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for cell in ARMS:
        ratios = profiles[cell]
        per_seed_max = ratios.max(axis=1)
        rows.append({"cell": cell,
                     # the gate statistic: maximum over position PER SEED, then averaged
                     "max_ratio_gate": float(per_seed_max.mean()),
                     "max_ratio_gate_sd": float(per_seed_max.std(ddof=1)),
                     "argmax_pos_per_seed": " ".join(str(int(p)) for p in ratios.argmax(axis=1)),
                     # the plotted curve's own maximum, lower where the peak wanders between seeds
                     "max_ratio_seed_mean_curve": float(ratios.mean(axis=0).max()),
                     "edge_ratio": float(np.maximum(ratios[:, 0], ratios[:, -1]).mean()),
                     "centre_ratio": float(ratios[:, 112:145].max(axis=1).mean()),
                     "n_seeds": len(SEEDS)})
    summary = pd.DataFrame(rows)
    prov_path = ROOT / "paper" / "build" / "figD_recon_data.csv"
    pd.concat([summary, pd.DataFrame(provenance)], axis=0, ignore_index=True).to_csv(prov_path, index=False)
    log.info(f"wrote {out_path} and {prov_path}")
    print(summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
