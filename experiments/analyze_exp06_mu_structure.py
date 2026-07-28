"""exp06 pre-design forensics, part 2: is the latent an amplitude meter?

Two exp05 findings point the same way from opposite directions. Section B3 measured the participation
ratio of star-level mu at 1.19 (lambda=0) and 1.79 (lambda=66), meaning the representation's variance
lives in one to two effective directions, and found the dominant dimension correlating moderately with
every task at once. Section C3b then measured the probes against nuisance statistics on true negatives
only, and found the transit probe tracking point-to-point scatter at rho = +0.70 and the rotation probe
tracking it at rho = -0.91. Both readings say the same thing: much of what the probe uses may be a
single "how variable is this star" scalar rather than anything about stellar physics.

Neither section tested it. This script does, on the frozen v1 subset, over all four seeds (B3 was one):

  structure   PCA spectrum and participation ratio of star-level mu, per arm and seed, plus how much of
              each leading PC a small set of amplitude scalars explains. Participation ratio is used
              rather than the active-unit count because it is invariant to rescaling, which the KL-based
              count is not (see B1's note on lpsd's free_bits floor inflating it to 128/128).

  probe       the decisive test. Regress every mu dimension on the amplitude scalars, fit on train only,
              and re-run the identical logistic probe on the residual. PR-AUC that survives is signal the
              amplitude scalars do not already carry. If the paired criterion-1 delta does not survive,
              exp05's gain is an amplitude-calibration gain and exp06's objective has to target phase or
              frequency content instead.

  amp_only    the converse bound: the same probe on the four amplitude scalars alone. C3b's rho = -0.91
              predicts this nearly reproduces the rotation probe from four numbers.

The amplitude basis is deliberately free of periodicity: the periodogram features (peak frequencies,
band powers, spectral entropy and centroid) are all excluded, so residualizing removes scale and
roughness information only, and anything frequency-related survives by construction.

Writes experiments/exp06_mu_structure.csv and experiments/exp06_mu_probe.csv. The engineered feature
table is cached to experiments/exp06_features_cache.parquet since it is deterministic and slow.
Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp06_mu_structure.py
    python experiments/analyze_exp06_mu_structure.py --seeds 0 --limit-stars 400
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

from swm.eval.features import FEATURE_NAMES
from swm.eval.skyline import encoder_mu_table, feature_table, logistic_scores
from swm.models import WorldModel

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

TASKS = ("pulsating", "eb", "rotation", "transit")
MU_COLS = [f"mu{j}" for j in range(128)]

# Scale and roughness only. Every periodogram feature is excluded on purpose, so residualizing against
# this basis cannot remove frequency information -- what survives is genuinely not an amplitude effect.
AMPLITUDE_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"]

OFF_CELL = "exp05_comb_off"
DYN_CELL = "exp05_comb_fbwd_c1p0"


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
    The participation ratio (sum of eigenvalues squared over sum of squared eigenvalues) answers "how
    many directions does the variance actually occupy", and is invariant to a global rescaling, unlike
    an absolute standard-deviation threshold. A value near 1 means one dimension carries everything.
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


def explained_by(features: np.ndarray, target: np.ndarray) -> float:
    """R^2 of an ordinary least-squares fit from engineered features to one latent direction."""
    model = LinearRegression().fit(features, target)
    return float(model.score(features, target))


def residualize(table: pd.DataFrame, cols: list[str], basis: list[str]) -> pd.DataFrame:
    """
    Remove everything the amplitude basis can linearly predict from each latent dimension.
    The regression is fitted on the TRAIN split only and applied to both, so the test residual carries
    no information that leaked from test labels or test feature distributions.
    """
    out = table.copy()
    train = table[table["split"] == "train"]
    fitter = LinearRegression().fit(train[basis].to_numpy(), train[cols].to_numpy())
    out[cols] = table[cols].to_numpy() - fitter.predict(table[basis].to_numpy())
    return out


def probe_all_tasks(table: pd.DataFrame, cols: list[str], arm: str, seed: int, readout: str) -> list[dict]:
    """Run the protocol-matched logistic probe for every v1 task and return one row per task."""
    rows = []
    for task in TASKS:
        _, y, scores = logistic_scores(table, cols, task)
        rows.append({"arm": arm, "seed": seed, "readout": readout, "task": task,
                     "pr_auc": float(average_precision_score(y, scores)),
                     "n_test": int(len(y)), "n_pos": int(y.sum())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 pre-design forensics: what does mu encode?")
    ap.add_argument("--cells", nargs="+", default=[OFF_CELL, DYN_CELL])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--ckpt", default="best_recon_aux")
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--limit-stars", type=int, default=0, help="smoke test: truncate each split")
    ap.add_argument("--feature-cache", default="experiments/exp06_features_cache.parquet")
    ap.add_argument("--out-structure", default="experiments/exp06_mu_structure.csv")
    ap.add_argument("--out-probe", default="experiments/exp06_mu_probe.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset = pd.read_parquet(ROOT / "processed/subset/subset_tics.parquet")
    packed = ROOT / "experiments" / args.cells[0] / "packed"
    window = 256

    cache = ROOT / args.feature_cache
    if cache.exists():
        feats = pd.read_parquet(cache)
        log.info(f"loaded cached engineered features {cache} ({len(feats)} stars)")
    else:
        feats = feature_table(packed, window, subset, TASKS)
        feats.to_parquet(cache, index=False)
        log.info(f"wrote engineered feature cache {cache} ({len(feats)} stars)")

    structure_rows: list[dict] = []
    probe_rows: list[dict] = []
    arms: list[tuple[str, int, WorldModel]] = []
    cfg = None
    for cell in args.cells:
        for seed in args.seeds:
            path = ROOT / "experiments" / cell / "models" / f"{args.variant}_seed{seed}" / f"{args.ckpt}.pt"
            if not path.exists():
                log.warning(f"{cell} seed {seed}: no {args.ckpt}.pt; skipped")
                continue
            model, cfg = load_model(path, device)
            arms.append((cell.replace("exp05_", ""), seed, model))
    assert cfg is not None, "no checkpoints loaded - check --cells/--seeds"

    torch.manual_seed(0) # fixed init, matching every other untrained reference in this project
    untrained = WorldModel(
        in_ch=1, enc_channels=list(cfg["model"]["enc_channels"]), kernel_size=int(cfg["model"]["kernel_size"]),
        z_dim=int(cfg["model"]["z_dim"]), window=window,
        gru_hidden=int(cfg["model"]["gru_hidden"]), gru_layers=int(cfg["model"]["gru_layers"]),
    ).to(device).eval()
    arms.append(("untrained", 0, untrained))

    for arm, seed, model in tqdm(arms, desc="arms", total=len(arms)):
        mu = encoder_mu_table(model, packed, window, subset, TASKS, MU_COLS, device)
        table = mu.merge(feats[["tic_id", "split", *FEATURE_NAMES]], on=["tic_id", "split"], how="inner")
        assert len(table) == len(mu), "a star with mu is missing from the engineered feature table"
        if args.limit_stars:
            table = pd.concat([g.head(args.limit_stars) for _, g in table.groupby("split")], ignore_index=True)

        x = table[MU_COLS].to_numpy()
        pr, evr = participation_ratio(x)
        pcs = pc_scores(x, args.n_pcs)
        amp = table[AMPLITUDE_COLS].to_numpy()
        allf = table[FEATURE_NAMES].to_numpy()
        row = {"arm": arm, "seed": seed, "participation_ratio": pr,
               "max_dim_sd": float(x.std(axis=0).max()),
               "dominant_dim": MU_COLS[int(np.argmax(x.std(axis=0)))]}
        for j in range(args.n_pcs):
            row[f"evr_pc{j + 1}"] = float(evr[j])
            row[f"r2_amp_pc{j + 1}"] = explained_by(amp, pcs[:, j])
            row[f"r2_all_pc{j + 1}"] = explained_by(allf, pcs[:, j])
        structure_rows.append(row)

        resid = residualize(table, MU_COLS, AMPLITUDE_COLS)
        probe_rows += probe_all_tasks(table, MU_COLS, arm, seed, "mu")
        probe_rows += probe_all_tasks(resid, MU_COLS, arm, seed, "mu_resid_amp")

    probe_rows += probe_all_tasks(feats, AMPLITUDE_COLS, "features", 0, "amp_only")
    probe_rows += probe_all_tasks(feats, FEATURE_NAMES, "features", 0, "all25")

    structure = pd.DataFrame(structure_rows)
    probe = pd.DataFrame(probe_rows)
    structure.to_csv(ROOT / args.out_structure, index=False)
    probe.to_csv(ROOT / args.out_probe, index=False)
    log.info(f"\nwrote {ROOT / args.out_structure} and {ROOT / args.out_probe}")

    pd.set_option("display.width", 220)
    print()
    print(structure.groupby("arm")[["participation_ratio", "evr_pc1", "r2_amp_pc1", "r2_all_pc1"]]
          .mean().round(4).to_string())
    print()
    print(probe.pivot_table(index="task", columns=["arm", "readout"], values="pr_auc").round(4).to_string())
    print("\nPR-AUC that survives the mu_resid_amp readout is signal the four amplitude scalars do not")
    print("already carry. Compare amp_only to mu to see how much of the probe four numbers reproduce.")


if __name__ == "__main__":
    main()
