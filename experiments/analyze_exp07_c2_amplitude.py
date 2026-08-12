"""exp07 pre-check C2: does the log-PSD recipe escape the amplitude meter?

exp05 forensics section 2 measured star-level mu under the `comb` recipe and found it is largely one
scalar: PC1 carries 92.6% of the variance, the participation ratio sits at 1.16-1.57, and four
amplitude statistics linearly reproduce 96.6% of PC1, so roughly 89% of mu's total variance is an
amplitude meter. That measurement was never run on the `lpsd` cells, and exp07's aux-term 2x2 is
premised on the aux form being able to move it.

This script runs the identical structure measurement on both recipes:

  exp05_comb_off / exp05_comb_fbwd_c1p0   combined aux @ 0.3, free_bits 0.0 (the forensics baseline)
  exp05_lpsd_off / exp05_lpsd_multi_c1p0  log_psd aux @ 0.1, free_bits 0.02

Per arm and seed: PC1 explained-variance share, participation ratio (effective dimensionality), the R^2
of an amplitude-only regression onto PC1, and `amp_var_frac`, the variance-weighted R^2 of the same four
scalars regressed onto ALL 128 mu dimensions. The last is the headline generalisation of the "89%"
figure, which was evr_pc1 * r2_amp_pc1 on a single arm; both are reported so the new numbers are
directly comparable to the quoted ones.

Two footings are reported side by side because the request and the precedent differ: `pooled` is
train + test, exactly what forensics section 2 measured, and `test` is the test split alone. The verdict
is read off `pooled` to keep it on the precedent's footing, with `test` as the robustness column.

The lpsd cells vary free_bits AND the aux type together (base_lpsd.yaml sets both), so a difference
here cannot be attributed to the aux form alone. That is what exp07's fb=0 log_psd cell separates; this
check only sharpens the prediction.

Writes experiments/exp07_c2_amplitude.csv. Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp07_c2_amplitude.py
    python experiments/analyze_exp07_c2_amplitude.py --seeds 0 --limit-stars 300
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from swm.eval.skyline import encoder_mu_table
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

TASKS = ("pulsating", "eb", "rotation", "transit")
MU_COLS = [f"mu{j}" for j in range(128)]
# Scale and roughness only, no periodogram feature: residualizing against this basis cannot remove
# frequency information, so what it explains is genuinely an amplitude effect (forensics 2.2 basis).
AMPLITUDE_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"]

DEFAULT_CELLS = ["exp05_comb_off", "exp05_comb_fbwd_c1p0", "exp05_lpsd_off", "exp05_lpsd_multi_c1p0"]
WINDOW = 256 # every exp05 cell shares the exp01 geometry


def load_model(ckpt_path: Path, device: str) -> tuple[WorldModel, dict]:
    """Rebuild one run's world model from its checkpoint, strict-loading the backward head when present."""
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


def participation_ratio(x: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Effective dimensionality of a star-by-dimension matrix, plus its explained-variance spectrum.
    Sum of eigenvalues squared over sum of squared eigenvalues answers "how many directions does the
    variance actually occupy", and is invariant to a global rescaling unlike an absolute threshold.
    Identical estimator to analyze_exp06_mu_structure so the comb rows must reproduce section 2.
    """
    centred = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(centred, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals[::-1], 0.0, None)
    pr = float(eigvals.sum() ** 2 / (eigvals ** 2).sum())
    return pr, eigvals / max(eigvals.sum(), 1e-12)


def pc_scores(x: np.ndarray, k: int) -> np.ndarray:
    """Project a star-by-dimension matrix onto its own top-k principal components."""
    centred = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:k].T # (n_stars, k)


def amplitude_variance_fraction(amp: np.ndarray, mu: np.ndarray) -> float:
    """
    Fraction of mu's TOTAL variance that the four amplitude scalars linearly reproduce.
    Regresses all 128 dimensions on the basis at once and pools the residual and total sums of squares
    across dimensions, which weights each dimension by how much variance it actually holds. This is the
    proper form of the "~89% amplitude meter" figure, which was read off PC1 alone.
    """
    fitted = LinearRegression().fit(amp, mu).predict(amp)
    sse = float(((mu - fitted) ** 2).sum())
    sst = float(((mu - mu.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def structure_row(table: pd.DataFrame, footing: str, n_pcs: int, meta: dict) -> dict:
    """All amplitude-dominance statistics for one arm on one split footing."""
    mu = table[MU_COLS].to_numpy()
    amp = table[AMPLITUDE_COLS].to_numpy()
    pr, evr = participation_ratio(mu)
    pcs = pc_scores(mu, n_pcs)
    row = {**meta, "footing": footing, "n_stars": int(len(table)),
           "participation_ratio": pr, "amp_var_frac": amplitude_variance_fraction(amp, mu)}
    for j in range(n_pcs):
        row[f"evr_pc{j + 1}"] = float(evr[j])
        row[f"r2_amp_pc{j + 1}"] = float(LinearRegression().fit(amp, pcs[:, j]).score(amp, pcs[:, j]))
    row["evr1_x_r2amp1"] = row["evr_pc1"] * row["r2_amp_pc1"] # the form the quoted 89% took
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="exp07 C2: amplitude dominance of mu, comb vs lpsd")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--n-pcs", type=int, default=3)
    ap.add_argument("--limit-stars", type=int, default=0, help="smoke test: truncate each split")
    ap.add_argument("--feature-cache", default="experiments/exp06_features_cache.parquet")
    ap.add_argument("--out", default="experiments/exp07_c2_amplitude.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subset = pd.read_parquet(ROOT / "processed/subset/subset_tics.parquet")
    feats = pd.read_parquet(ROOT / args.feature_cache)
    packed = ROOT / "experiments" / args.cells[0] / "packed"

    arms = []
    cfg = None
    for cell in args.cells:
        for seed in args.seeds:
            path = ROOT / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            assert path.exists(), f"missing {path}"
            arms.append((cell, seed, path))
    log.info(f"device {device} | {len(arms)} arms")

    rows = []
    for cell, seed, path in tqdm(arms, desc="arms", total=len(arms)):
        model, cfg = load_model(path, device)
        mu = encoder_mu_table(model, packed, WINDOW, subset, TASKS, MU_COLS, device)
        table = mu.merge(feats[["tic_id", "split", *AMPLITUDE_COLS]], on=["tic_id", "split"], how="inner")
        assert len(table) == len(mu), "a star with mu is missing from the engineered feature table"
        if args.limit_stars:
            table = pd.concat([g.head(args.limit_stars) for _, g in table.groupby("split")], ignore_index=True)

        recipe = "lpsd" if "lpsd" in cell else "comb"
        meta = {"cell": cell.replace("exp05_", ""), "recipe": recipe, "seed": seed,
                "aux_type": cfg["train"]["recon_aux"]["type"],
                "aux_weight": float(cfg["train"]["recon_aux"]["weight"]),
                "free_bits": float(cfg["train"]["free_bits"]),
                "dyn_on": float(cfg["train"]["lambda_dyn"]) > 0}
        rows.append(structure_row(table, "pooled", args.n_pcs, meta))
        rows.append(structure_row(table[table["split"] == "test"], "test", args.n_pcs, meta))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # The capacity-matched untrained reference, on the same fixed init every other arm in this project uses.
    torch.manual_seed(0)
    untrained = WorldModel(
        in_ch=1, enc_channels=list(cfg["model"]["enc_channels"]), kernel_size=int(cfg["model"]["kernel_size"]),
        z_dim=int(cfg["model"]["z_dim"]), window=WINDOW,
        gru_hidden=int(cfg["model"]["gru_hidden"]), gru_layers=int(cfg["model"]["gru_layers"]),
    ).to(device).eval()
    mu = encoder_mu_table(untrained, packed, WINDOW, subset, TASKS, MU_COLS, device)
    table = mu.merge(feats[["tic_id", "split", *AMPLITUDE_COLS]], on=["tic_id", "split"], how="inner")
    if args.limit_stars:
        table = pd.concat([g.head(args.limit_stars) for _, g in table.groupby("split")], ignore_index=True)
    meta = {"cell": "untrained", "recipe": "untrained", "seed": 0, "aux_type": "none",
            "aux_weight": np.nan, "free_bits": np.nan, "dyn_on": False}
    rows.append(structure_row(table, "pooled", args.n_pcs, meta))
    rows.append(structure_row(table[table["split"] == "test"], "test", args.n_pcs, meta))

    structure = pd.DataFrame(rows)
    structure.to_csv(ROOT / args.out, index=False)

    pd.set_option("display.width", 240)
    for footing in ["pooled", "test"]:
        view = structure[structure["footing"] == footing]
        print(f"\n[{footing}] seed-mean amplitude dominance")
        print(view.groupby(["cell", "recipe"], sort=False)[
            ["evr_pc1", "participation_ratio", "r2_amp_pc1", "evr1_x_r2amp1", "amp_var_frac"]
        ].mean().round(4).to_string())
    print("\namp_var_frac is the fraction of mu's total variance the four amplitude scalars reproduce.")
    print("The comb rows must reproduce forensics section 2 (evr_pc1 0.926 / PR 1.16 / r2_amp 0.967).")
    log.info(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
