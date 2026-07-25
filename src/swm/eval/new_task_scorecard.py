"""Task 2 (plan 2026-07-22, D3/D4) — the new-task probe scorecard.

Scores the expanded probe menu on the frozen-encoder mu the leader arms already produced, one linear
readout per task shape (plan D3):
  detection binary   osc_giant, solar_like_osc, flare(Level A) -- logistic, mean + window_score(MIL),
                     metric PR-AUC (the pool prevalence is inflated, so PR-AUC is the honest read)
  contrastive binary rgb_vs_heb -- logistic on the catalog-only population, metric ROC-AUC (balanced)
  regression         numax_hon, numax_hatt, dnu_hatt, prot_kounkel -- Ridge, metric R2/RMSE/Spearman
                     (numax and dnu regressed in log10, the asteroseismic scaling)

Every cell is scored for each trained seed (exp03 leader best_recon_aux) and for the capacity-matched
untrained arm; the headline number is the trained-minus-untrained gap with a 3-seed SE. rotation_period
(TARS) rides the existing v1 packed subset, not the new pool. Rows append to
experiments/new_task/new_task_scorecard.csv.

Run (swm env, from repo root, PYTHONPATH=src; needs new_task_extract caches first):
    python -m swm.eval.new_task_scorecard
    python -m swm.eval.new_task_scorecard --arms seed0 untrained   # quick pair
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from swm.eval.readout_sweep import (build_model_from_ckpt, cached_mu, fit_readout_scores, pool_stars,
                                    window_score_scores)
from swm.eval.new_task_extract import LEADER_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("new_task_scorecard")

repo_root = Path(__file__).resolve().parents[3]

# task shape specs: (name, shape, column, poolings, log_target, extra population mask)
DETECTION = [("osc_giant", "osc_giant"), ("solar_like_osc", "solar_like_osc"), ("flare", "flare")]
REGRESSION = [("numax_hon", "numax_hon", True), ("numax_hatt", "numax_hatt", True),
              ("dnu_hatt", "dnu_hatt", True), ("prot_kounkel", "prot_kounkel", False)]


def load_cache(cache_path: Path) -> dict[str, tuple[list[int], list[np.ndarray]]]:
    """Read a new_task_extract .npz into {split: (tics, per-star window-mu blocks)}, the readout_sweep layout."""
    payload = np.load(cache_path, allow_pickle=False)
    result = {}
    for split in ["train", "val", "test"]:
        flat = payload[f"{split}_mu"]
        counts = payload[f"{split}_counts"]
        tics = payload[f"{split}_tics"].tolist()
        blocks = []
        start = 0
        for count in counts:
            blocks.append(flat[start : start + int(count)])
            start += int(count)
        result[split] = (tics, blocks)
    return result


def label_frame() -> pd.DataFrame:
    """One tic-keyed frame carrying every new-pool task column (detection flags 0/1, regression targets, rgb)."""
    ext = pd.read_csv(repo_root / "labels" / "new_task_labels_star.csv")
    ext["tic_id"] = ext["tic_id"].astype(int)
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["flare"] = pd.to_numeric(canon["flare_ever"], errors="coerce").fillna(0).astype(int)
    frame = ext.merge(canon[["tic_id", "flare"]], on="tic_id", how="outer")
    for flag in ["osc_giant", "solar_like_osc", "flare"]:
        frame[flag] = pd.to_numeric(frame[flag], errors="coerce").fillna(0).astype(int)
    return frame.set_index("tic_id")


def y_for(tics: list[int], labels: pd.DataFrame, column: str) -> np.ndarray:
    """Column values aligned to `tics` (0 for tics absent from the label frame; used for detection flags)."""
    series = labels[column].reindex(tics)
    return series.to_numpy()


def score_regression(x_train, y_train, x_test, y_test) -> dict:
    """Ridge (standardized, CV alpha) on mean-pooled mu; returns R2/RMSE/Spearman on the held-out target."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)  # learn mean/std on train only (no leakage)
    x_te = scaler.transform(x_test)
    reg = RidgeCV(alphas=np.logspace(-2, 3, 10))  # L2 linear probe, alpha picked by leave-one-out CV
    reg.fit(x_tr, y_train)
    pred = reg.predict(x_te)
    rho = spearmanr(y_test, pred).statistic
    return {"r2": float(r2_score(y_test, pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "spearman": float(rho)}


def score_detection(mu: dict, labels: pd.DataFrame, column: str) -> list[dict]:
    """Detection binary over the whole pool: logistic on mean-pool and on window_score(MIL); PR-AUC + ROC-AUC."""
    train_tics, train_blocks = mu["train"]
    test_tics, test_blocks = mu["test"]
    y_train = y_for(train_tics, labels, column).astype(int)
    y_test = y_for(test_tics, labels, column).astype(int)
    rows = []
    x_train = pool_stars(train_blocks, "mean")
    x_test = pool_stars(test_blocks, "mean")
    for pooling in ["mean", "window_score"]:
        if pooling == "mean":
            scores = fit_readout_scores("logistic", x_train, y_train, x_test)
        else:
            scores = window_score_scores("logistic", train_blocks, y_train, test_blocks)
        rows.append({"pooling": pooling, "pr_auc": float(average_precision_score(y_test, scores)),
                     "roc_auc": float(roc_auc_score(y_test, scores)),
                     "n_test_pos": int(y_test.sum()), "n_test": int(len(y_test))})
    return rows


def score_contrastive(mu: dict, labels: pd.DataFrame) -> list[dict]:
    """RGB(1) vs HeB(0) on the Sreenivas-only population: logistic on mean-pool; ROC-AUC primary."""
    rows = []
    pooled = {}
    y = {}
    for split in ["train", "test"]:
        tics, blocks = mu[split]
        target = labels["rgb_vs_heb"].reindex(tics)
        keep = target.notna().to_numpy()
        feats = pool_stars(blocks, "mean")[keep]
        pooled[split] = feats
        y[split] = target[keep].astype(int).to_numpy()
    scores = fit_readout_scores("logistic", pooled["train"], y["train"], pooled["test"])
    rows.append({"pooling": "mean", "roc_auc": float(roc_auc_score(y["test"], scores)),
                 "pr_auc": float(average_precision_score(y["test"], scores)),
                 "n_test_pos": int(y["test"].sum()), "n_test": int(len(y["test"]))})
    return rows


def score_regression_task(mu: dict, labels: pd.DataFrame, column: str, log_target: bool) -> list[dict]:
    """One regression probe on the catalog-only population (prot capped at 5.7 d), Ridge on mean-pool mu."""
    pooled = {}
    y = {}
    for split in ["train", "test"]:
        tics, blocks = mu[split]
        target = labels[column].reindex(tics)
        keep = target.notna()
        if column == "prot_kounkel":
            gt57 = labels["prot_kounkel_gt57"].reindex(tics).fillna(1).astype(int)
            keep = keep & (gt57 == 0)  # headline: periods inside one segment length
        keep = keep.to_numpy()
        vals = target[keep].to_numpy(dtype=float)
        if log_target:
            vals = np.log10(vals)
        pooled[split] = pool_stars(blocks, "mean")[keep]
        y[split] = vals
    metrics = score_regression(pooled["train"], y["train"], pooled["test"], y["test"])
    metrics.update({"pooling": "mean", "n_test": int(len(y["test"])), "log_target": log_target})
    return [metrics]


def rotation_subset_mu(arm: str, device: str) -> dict:
    """Build/load the v1 packed-subset first-segment mu for one arm (rotation_period rides this, not the pool)."""
    packed = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
    cache_dir = repo_root / "experiments" / "new_task" / "subset_mu_cache"
    cache_path = cache_dir / f"{arm}.npz"
    model = None
    if not cache_path.exists():
        import torch
        from swm.eval.skyline import _make_untrained
        base = torch.load(LEADER_DIR / "B_seed0" / "best_recon_aux.pt", map_location="cpu", weights_only=False)
        if arm == "untrained":
            mc = base["cfg"]["model"]
            model = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                                    256, int(mc["gru_hidden"]), int(mc["gru_layers"]), device)
        else:
            ck = torch.load(LEADER_DIR / f"B_seed{arm.replace('seed', '')}" / "best_recon_aux.pt",
                            map_location="cpu", weights_only=False)
            model, _ = build_model_from_ckpt(ck, device)
    return cached_mu(cache_path, model, packed, 256, device, desc=f"rot[{arm}]")


def score_rotation_period(arm: str, device: str) -> list[dict]:
    """rotation_period (TARS, P<=5 d) regression on the frozen v1 subset for one trained seed arm."""
    mu = rotation_subset_mu(arm, device)
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["rotation"] = pd.to_numeric(canon["rotation"], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    keep_tics = canon.loc[(canon["rotation"] == 1) & canon["rotation_period"].notna()
                          & (canon["rotation_period"] <= 5), ["tic_id", "rotation_period"]]
    lookup = keep_tics.set_index("tic_id")["rotation_period"]
    pooled = {}
    y = {}
    for split in ["train", "test"]:
        tics, blocks = mu[split]
        target = lookup.reindex(tics)
        keep = target.notna().to_numpy()
        pooled[split] = pool_stars(blocks, "mean")[keep]
        y[split] = target[keep].to_numpy(dtype=float)
    metrics = score_regression(pooled["train"], y["train"], pooled["test"], y["test"])
    metrics.update({"pooling": "mean", "n_test": int(len(y["test"])), "log_target": False})
    return [metrics]


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the new-task probe menu on frozen-encoder mu.")
    ap.add_argument("--arms", nargs="+", default=["seed0", "seed1", "seed2", "untrained"])
    ap.add_argument("--cache-dir", default=None, help="Default: experiments/new_task/mu_cache")
    ap.add_argument("--with-rotation", action="store_true", help="Also score rotation_period on the v1 subset.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="Default: experiments/new_task/new_task_scorecard.csv")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / "experiments" / "new_task" / "mu_cache"
    labels = label_frame()
    rows = []
    for arm in args.arms:
        mu = load_cache(cache_dir / f"{arm}.npz")
        for name, column in DETECTION:
            for cell in score_detection(mu, labels, column):
                rows.append({"arm": arm, "task": name, "shape": "detection", **cell})
        for cell in score_contrastive(mu, labels):
            rows.append({"arm": arm, "task": "rgb_vs_heb", "shape": "contrastive", **cell})
        for name, column, log_target in REGRESSION:
            for cell in score_regression_task(mu, labels, column, log_target):
                rows.append({"arm": arm, "task": name, "shape": "regression", **cell})
        if args.with_rotation:
            for cell in score_rotation_period(arm, args.device):
                rows.append({"arm": arm, "task": "rotation_period", "shape": "regression", **cell})
        log.info(f"scored arm {arm}")

    result = pd.DataFrame(rows)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else repo_root / "experiments" / "new_task" / "new_task_scorecard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(rows)} new rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
