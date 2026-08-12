"""Pre-exp08 CHK-3 + CHK-4: what channel the probes read, and what the dynamics arm's extra units carry.

Both questions need the same per-star mu tables on the same cells and seeds, so they share one encode
pass here rather than two.

CHK-3 (open question Q2) -- what does mu encode?
    exp06 measured that star-level mu is dominated by one direction (PC1 = 93% of variance, R^2 0.97
    against amplitude scalars alone) and that probing mu with the AMPLITUDE basis projected out costs
    eb 0.755 -> 0.599 and rotation 0.531 -> 0.379 while pulsating survives (0.817 -> 0.788). That basis
    was deliberately periodicity-free, so frequency content survived by construction and the measurement
    cannot speak to the fusion claim. Nobody has run the probe against the FULL engineered basis, which
    is the number "SSL carries something engineered features do not" actually rests on.

    Two nested bases make the increment readable:
        B_amp   scale and roughness only (exp06's four scalars) -- removing it cannot remove periodicity
        B_full  all 25 T'DA-style features, periodogram structure included
    and five readouts per task: mu, mu perp B_amp, mu perp B_full, B_full alone, and B_full (+) mu --
    the fusion cell. Every residualisation is fitted on TRAIN only.

CHK-4 (open question Q11) -- what do the one-to-three extra dynamics units buy?
    At ep100 the dynamics arm holds barely more latent capacity than its own dyn-off arm (hann0p3 5.8 vs
    5.0) and still wins every task, while a cell holding 18 units scores mid-table (F25). So the win is
    not capacity. Three hypotheses: (a) new content, (b) the same content in a more linearly readable
    direction, (c) a scale change the probe's StandardScaler exploits.

    The obvious estimator does not separate them. Cross-RECIPE, linearly predicting one cell's mu from
    another's leaves resid_var_frac = 0.005 and collapses every probe to base rate; with 5-8 active dims
    sharing one dominant amplitude direction that collapse is near-guaranteed for any pair, cross-arm
    included. So this measures four things instead:

        cca         canonical correlations cross-ARM within recipe, against the within-cell CROSS-SEED
                    pair as the null distribution. Without that null a 47-degree principal angle means
                    nothing, because two seeds of the SAME cell are not identical either.
        residual    the probe on mu_fbwd perp mu_off AND on mu_off perp mu_fbwd. Both are low; the
                    ASYMMETRY is the signal, and it is paired by seed. Symmetric -> (b). Favouring
                    fbwd -> (a).
        scaling     hypothesis (c) directly: refit the identical probe on the dyn-off mu with its active
                    dimensions whitened, and refit both arms with the StandardScaler ablated. If
                    whitened-off closes the gap to fbwd, the dynamics benefit is a conditioning effect
                    and "world model" is the wrong description of it.
        dim_ridge   per-dimension R^2 from B_full onto each active mu dimension, which is what NAMES the
                    directions the other three isolate.

    Pairs are always within recipe (hann_fbwd vs hann_off, comb_fbwd vs comb_off), never hann_fbwd vs
    comb_off: that would span both design axes at once, which is F21 in its original form.

Writes experiments/exp07_channel_{probe,feature_map}.csv and exp07_dynunits_{cca,residual,scaling,dims}.csv.
Run (repo root, swm env, PYTHONPATH=src):
    python experiments/analyze_exp07_mu_channel.py
    python experiments/analyze_exp07_mu_channel.py --seeds 0 --limit-stars 400 --parts channel
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import average_precision_score, r2_score
from tqdm.auto import tqdm

from swm.eval.features import FEATURE_NAMES
from swm.eval.skyline import encoder_mu_table, feature_table, logistic_scores
from swm.models import WorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("exp07_mu_channel")

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "experiments" / "exp01_window256_seq16" / "packed"
WINDOW = 256
TASKS = ("pulsating", "eb", "rotation", "transit")
MU_COLS = [f"mu{j}" for j in range(128)]
# Scale and roughness only; every periodogram feature excluded on purpose (exp06 AMPLITUDE_COLS).
AMP_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"]
RECIPES = ["hann0p3", "comb0p3"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5]
FEATURE_CACHE = ROOT / "experiments" / "exp06_features_cache.parquet"
RIDGE_ALPHAS = np.logspace(-3, 4, 15)
ACTIVE_SD = 1e-3  # a mu dimension below this train-split SD is collapsed and carries nothing


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


def residualize(table: pd.DataFrame, cols: list[str], basis: list[str]) -> pd.DataFrame:
    """Remove everything `basis` linearly predicts from each column, fitting on the TRAIN split only."""
    out = table.copy()
    train = table[table["split"] == "train"]
    fitter = LinearRegression().fit(train[basis].to_numpy(), train[cols].to_numpy())
    out[cols] = table[cols].to_numpy() - fitter.predict(table[basis].to_numpy())
    return out


def probe_rows(table: pd.DataFrame, cols: list[str], cell: str, seed: int, readout: str) -> list[dict]:
    """Protocol-matched logistic probe over every v1 task, one row per task."""
    rows = []
    for task in TASKS:
        _, y, scores = logistic_scores(table, cols, task)
        rows.append({"cell": cell, "seed": seed, "readout": readout, "task": task,
                     "pr_auc": float(average_precision_score(y, scores)),
                     "n_test": int(len(y)), "n_pos": int(y.sum())})
    return rows


def active_dims(table: pd.DataFrame) -> list[str]:
    """mu dimensions with non-negligible train-split spread; the rest are posterior-collapsed."""
    train = table[table["split"] == "train"][MU_COLS].to_numpy()
    return [c for c, sd in zip(MU_COLS, train.std(axis=0)) if sd > ACTIVE_SD]


# --------------------------------------------------------------------------------------- CHK-3

def channel_probes(table: pd.DataFrame, cell: str, seed: int) -> list[dict]:
    """The five readouts that locate the probe-carrying signal between mu and the engineered basis."""
    rows = probe_rows(table, MU_COLS, cell, seed, "mu")
    rows += probe_rows(residualize(table, MU_COLS, AMP_COLS), MU_COLS, cell, seed, "mu_resid_amp")
    rows += probe_rows(residualize(table, MU_COLS, list(FEATURE_NAMES)), MU_COLS, cell, seed, "mu_resid_full")
    rows += probe_rows(table, list(FEATURE_NAMES), cell, seed, "features_only")
    rows += probe_rows(table, list(FEATURE_NAMES) + MU_COLS, cell, seed, "features_plus_mu")
    return rows


def feature_map(table: pd.DataFrame, cell: str, seed: int) -> list[dict]:
    """Both directions between mu and the engineered basis, scored on the held-out split.

    mu -> feature says how much of each engineered quantity the latent has re-encoded. feature -> mu
    says how much of the latent the engineered basis can reconstruct. Ridge is the linear reading;
    the boosted-tree version bounds how much is there under any smooth readout, so a low ridge R^2 with
    a high GBM R^2 would mean "present but not linearly available" rather than "absent".
    """
    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    cols = active_dims(table)
    rows = []
    for feature in FEATURE_NAMES:
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(train[cols].to_numpy(), train[feature].to_numpy())
        gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            train[cols].to_numpy(), train[feature].to_numpy())
        rows.append({"cell": cell, "seed": seed, "direction": "mu_to_feature", "target": feature,
                     "r2_ridge": float(r2_score(test[feature], ridge.predict(test[cols].to_numpy()))),
                     "r2_gbm": float(r2_score(test[feature], gbm.predict(test[cols].to_numpy())))})
    for dim in cols:
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(train[FEATURE_NAMES].to_numpy(), train[dim].to_numpy())
        gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            train[FEATURE_NAMES].to_numpy(), train[dim].to_numpy())
        rows.append({"cell": cell, "seed": seed, "direction": "feature_to_mu", "target": dim,
                     "r2_ridge": float(r2_score(test[dim], ridge.predict(test[FEATURE_NAMES].to_numpy()))),
                     "r2_gbm": float(r2_score(test[dim], gbm.predict(test[FEATURE_NAMES].to_numpy())))})
    return rows


# --------------------------------------------------------------------------------------- CHK-4

def canonical_correlations(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Canonical correlations between two star-by-dimension matrices, via QR on their top-k PCs.

    Reduced to k principal components first because each mu has only 5-8 dimensions with any spread out
    of 128; a full-rank CCA on the raw matrices would report correlations between numerical noise.
    """
    def top_pcs(x: np.ndarray) -> np.ndarray:
        centred = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return centred @ vt[:k].T
    qa, _ = np.linalg.qr(top_pcs(a))
    qb, _ = np.linalg.qr(top_pcs(b))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def cca_rows(mu_a: pd.DataFrame, mu_b: pd.DataFrame, pair: str, label: str, seed: int, k: int) -> list[dict]:
    """Canonical correlations and principal angles for one pair, computed on the TEST split."""
    a = mu_a[mu_a["split"] == "test"][MU_COLS].to_numpy()
    b = mu_b[mu_b["split"] == "test"][MU_COLS].to_numpy()
    corrs = canonical_correlations(a, b, k)
    return [{"pair": pair, "label": label, "seed": seed, "k": k, "component": i + 1,
             "cancorr": float(c), "angle_deg": float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))}
            for i, c in enumerate(corrs)]


def residual_probe_rows(target: pd.DataFrame, control: pd.DataFrame, label: str, seed: int) -> list[dict]:
    """Probe `target` mu after linearly predicting it away from `control` mu, fitted on train only.

    Run in BOTH directions by the caller. Each direction alone is uninformative (the residual is always
    small); the paired difference between the two is the quantity.
    """
    ctrl_cols = [f"ctrl{j}" for j in range(len(MU_COLS))]
    control_aligned = control.set_index(["tic_id", "split"]).loc[
        pd.MultiIndex.from_frame(target[["tic_id", "split"]])][MU_COLS].to_numpy()
    # concat rather than 128 inserts: assigning column-by-column fragments the frame and pandas warns
    joined = pd.concat([target.reset_index(drop=True),
                        pd.DataFrame(control_aligned, columns=ctrl_cols)], axis=1)
    resid = residualize(joined, MU_COLS, ctrl_cols)
    var_frac = float(resid[MU_COLS].to_numpy().var() / max(joined[MU_COLS].to_numpy().var(), 1e-12))
    rows = probe_rows(resid, MU_COLS, label, seed, "residual")
    for row in rows:
        row["resid_var_frac"] = var_frac
    return rows


def scaling_rows(table: pd.DataFrame, cell: str, seed: int) -> list[dict]:
    """Hypothesis (c): is the dynamics benefit a conditioning effect the StandardScaler could supply?

    `mu_whitened` rescales each active dimension to unit train variance BEFORE the probe's own scaler,
    which is a no-op for a scaled linear probe and therefore a null control -- it has to come out equal
    to `mu`, and if it does not, the probe pipeline is not doing what the tables assume. `mu_unscaled`
    ablates the scaler instead, which is the reading that actually varies: if the arms differ far more
    without the scaler than with it, the trained-vs-untrained ordering the tables report is partly a
    conditioning artefact rather than a representation one.
    """
    cols = active_dims(table)
    train = table[table["split"] == "train"]
    sd = train[cols].to_numpy().std(axis=0)
    whitened = table.copy()
    whitened[cols] = table[cols].to_numpy() / np.maximum(sd, 1e-12)
    rows = probe_rows(whitened, cols, cell, seed, "mu_whitened")
    rows += probe_rows(table, cols, cell, seed, "mu_active")
    for task in TASKS:
        # scaler-ablated probe: same estimator, raw mu columns, so any difference is pure conditioning
        from sklearn.linear_model import LogisticRegression
        tr, te = table[table["split"] == "train"], table[table["split"] == "test"]
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
            tr[cols].to_numpy(), tr[task].to_numpy())
        scores = clf.predict_proba(te[cols].to_numpy())[:, 1]
        rows.append({"cell": cell, "seed": seed, "readout": "mu_unscaled", "task": task,
                     "pr_auc": float(average_precision_score(te[task].to_numpy(), scores)),
                     "n_test": int(len(te)), "n_pos": int(te[task].sum())})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="CHK-3 what mu encodes + CHK-4 what the dyn units carry.")
    ap.add_argument("--recipes", nargs="+", default=RECIPES)
    ap.add_argument("--pairs", nargs="+", default=None, metavar="CELL_A:CELL_B",
                    help="explicit cell pairs, A = arm under test, B = control (e.g. "
                         "exp08_linear:exp07_hann0p3_off); overrides the exp07 --recipes naming. "
                         "In the outputs the pair label is CELL_A and the direction strings keep "
                         "the exp07 vocabulary (fbwd_perp_off = A perp B).")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--parts", nargs="+", default=["channel", "dyn"], choices=["channel", "dyn"])
    ap.add_argument("--no-feature-map", action="store_true",
                    help="channel part: skip the per-dimension mu<->feature ridge/GBM map (it fits one "
                         "GBM per ACTIVE dim per seed, which explodes on wide latents like frozen_fbwd's "
                         "~100 units); the probe readouts including the fusion cell still run")
    ap.add_argument("--cca-k", type=int, default=8, help="PCs retained per arm before the CCA.")
    ap.add_argument("--limit-stars", type=int, default=0, help="Truncate each split (smoke).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-prefix", default=None, help="Default: experiments/exp07")
    args = ap.parse_args()

    prefix = Path(args.out_prefix) if args.out_prefix else ROOT / "experiments" / "exp07"
    subset = pd.read_parquet(ROOT / "processed" / "subset" / "subset_tics.parquet")

    if FEATURE_CACHE.exists():
        feats = pd.read_parquet(FEATURE_CACHE)
        log.info(f"loaded engineered feature cache ({len(feats)} stars)")
    else:
        feats = feature_table(PACKED, WINDOW, subset, TASKS)
        feats.to_parquet(FEATURE_CACHE, index=False)
        log.info(f"wrote engineered feature cache ({len(feats)} stars)")

    # (label, cell_under_test, control_cell); the exp07 default reproduces the published CHK-3/CHK-4
    # runs exactly, --pairs generalizes the same estimators to arbitrary arm pairs (exp08 ladder).
    if args.pairs:
        pairs = [(p.split(":")[0], p.split(":")[0], p.split(":")[1]) for p in args.pairs]
    else:
        pairs = [(recipe, f"exp07_{recipe}_fbwd", f"exp07_{recipe}_off") for recipe in args.recipes]
    cells = list(dict.fromkeys([c for _, a, b in pairs for c in (a, b)]))

    tables: dict[tuple[str, int], pd.DataFrame] = {}
    channel_rows: list[dict] = []
    featmap_rows: list[dict] = []
    for cell in cells:
        for seed in tqdm(args.seeds, desc=f"{cell}"):
            model = load_model(cell, seed, args.device)
            mu = encoder_mu_table(model, PACKED, WINDOW, subset, TASKS, MU_COLS, args.device)
            table = mu.merge(feats[["tic_id", "split", *FEATURE_NAMES]], on=["tic_id", "split"],
                             how="inner")
            assert len(table) == len(mu), "a star with mu is missing from the engineered feature table"
            if args.limit_stars:
                table = pd.concat([g.head(args.limit_stars) for _, g in table.groupby("split")],
                                  ignore_index=True)
            tables[(cell, seed)] = table
            if "channel" in args.parts:
                channel_rows += channel_probes(table, cell, seed)
                if not args.no_feature_map:
                    featmap_rows += feature_map(table, cell, seed)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if "channel" in args.parts:
        probe = pd.DataFrame(channel_rows)
        probe.to_csv(f"{prefix}_channel_probe.csv", index=False)
        pd.DataFrame(featmap_rows).to_csv(f"{prefix}_channel_feature_map.csv", index=False)
        log.info(f"wrote {prefix}_channel_{{probe,feature_map}}.csv")
        print("\nCHK-3 probe by readout (mean over cells and seeds):")
        print(probe.pivot_table(index="readout", columns="task", values="pr_auc").round(3).to_string())

    if "dyn" in args.parts:
        cca, resid, scale = [], [], []
        for label, cell_a, cell_b in pairs:
            for seed in args.seeds:
                a, b = tables[(cell_a, seed)], tables[(cell_b, seed)]
                cca += cca_rows(a, b, "cross_arm", label, seed, args.cca_k)
                # direction strings keep the exp07 vocabulary: fbwd_perp_off means A perp B
                resid += [r | {"direction": "fbwd_perp_off", "recipe": label}
                          for r in residual_probe_rows(a, b, cell_a, seed)]
                resid += [r | {"direction": "off_perp_fbwd", "recipe": label}
                          for r in residual_probe_rows(b, a, cell_b, seed)]
                scale += [r | {"recipe": label, "arm": "fbwd"} for r in scaling_rows(a, cell_a, seed)]
                scale += [r | {"recipe": label, "arm": "off"} for r in scaling_rows(b, cell_b, seed)]
            # the null distribution any cross-arm angle has to beat: two SEEDS of the same cell, which
            # are not identical either. Without it a 47-degree principal angle is uninterpretable.
            for arm, cell in (("fbwd", cell_a), ("off", cell_b)):
                for i, seed_a in enumerate(args.seeds):
                    for seed_b in args.seeds[i + 1:]:
                        cca += cca_rows(tables[(cell, seed_a)], tables[(cell, seed_b)],
                                        f"same_cell_{arm}", label, seed_a, args.cca_k)
        pd.DataFrame(cca).to_csv(f"{prefix}_dynunits_cca.csv", index=False)
        residual = pd.DataFrame(resid)
        residual.to_csv(f"{prefix}_dynunits_residual.csv", index=False)
        scaling = pd.DataFrame(scale)
        scaling.to_csv(f"{prefix}_dynunits_scaling.csv", index=False)
        log.info(f"wrote {prefix}_dynunits_{{cca,residual,scaling}}.csv")
        print("\nCHK-4 residual probe, both directions (mean PR-AUC over seeds):")
        print(residual.pivot_table(index=["recipe", "direction"], columns="task", values="pr_auc")
              .round(3).to_string())
        print("\nCHK-4 scaling control (mean PR-AUC over seeds):")
        print(scaling.pivot_table(index=["recipe", "arm", "readout"], columns="task", values="pr_auc")
              .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
