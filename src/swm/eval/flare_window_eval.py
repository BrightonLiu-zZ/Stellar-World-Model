"""Task 2 (plan 2026-07-22, D7 Level B) — flare localization eval on window-level mu.

Does the frozen encoder mu separate in-flare windows from clean windows of the same stars? This is the
flare analogue of transit_window_eval: fit a logistic readout on window-level mu with the true
per-window flare labels (flare_window_labels.py, quarantine -1 dropped), then score window-level PR-AUC
on the held-out split, trained seeds vs the capacity-matched untrained arm. A within-star AUC (rank
in-flare above clean inside each star) isolates localization from any star-level confound. Reuses the
new-task pool mu cache row-for-row (identical replay order), so no extra GPU pass.

Run (swm env, from repo root, PYTHONPATH=src; needs new_task_extract + flare_window_labels first):
    python -m swm.eval.flare_window_eval
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from swm.eval.new_task_scorecard import load_cache
from swm.eval.readout_sweep import fit_readout_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("flare_window_eval")

repo_root = Path(__file__).resolve().parents[3]


def window_rows(mu: dict, labels: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack (window mu, flare label, star tic) for flaring stars in one split; quarantine -1 kept for masking."""
    tics, blocks = mu[split]
    tic_to_block = {}
    for tic, block in zip(tics, blocks):
        tic_to_block[int(tic)] = block
    label_by_star = labels[labels["split"] == split].groupby("tic_id")
    x_rows = []
    y_rows = []
    star_rows = []
    for tic, group in label_by_star:
        block = tic_to_block.get(int(tic))
        if block is None:
            continue
        lab = group.sort_values("win_idx")["label"].to_numpy()
        assert block.shape[0] == len(lab), f"TIC {tic}: mu has {block.shape[0]} windows, labels {len(lab)}"
        x_rows.append(block)
        y_rows.append(lab)
        star_rows.append(np.full(len(lab), int(tic)))
    x = np.concatenate(x_rows, axis=0)
    y = np.concatenate(y_rows)
    stars = np.concatenate(star_rows)
    return x, y, stars


def within_star_auc(scores: np.ndarray, y: np.ndarray, stars: np.ndarray) -> float:
    """Mean per-star ROC-AUC ranking in-flare above clean windows, over stars that have both classes."""
    aucs = []
    for tic in np.unique(stars):
        mask = stars == tic
        ys = y[mask]
        if ys.min() == ys.max():
            continue
        aucs.append(roc_auc_score(ys, scores[mask]))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def main() -> int:
    ap = argparse.ArgumentParser(description="Flare localization eval on window-level mu.")
    ap.add_argument("--arms", nargs="+", default=["seed0", "seed1", "seed2", "untrained"])
    ap.add_argument("--cache-dir", default=None, help="Default: experiments/new_task/mu_cache")
    ap.add_argument("--labels", default=None, help="Default: labels/qc/flare_window_labels_pool.parquet")
    ap.add_argument("--out", default=None, help="Default: experiments/new_task/flare_window_eval.csv")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / "experiments" / "new_task" / "mu_cache"
    labels_path = Path(args.labels) if args.labels else repo_root / "labels" / "qc" / "flare_window_labels_pool.parquet"
    labels = pd.read_parquet(labels_path)
    labels["tic_id"] = labels["tic_id"].astype(int)

    rows = []
    for arm in args.arms:
        mu = load_cache(cache_dir / f"{arm}.npz")
        x_train, y_train, _ = window_rows(mu, labels, "train")
        x_test, y_test, stars_test = window_rows(mu, labels, "test")
        keep_tr = y_train >= 0  # drop quarantine windows from the fit
        keep_te = y_test >= 0
        scores = fit_readout_scores("logistic", x_train[keep_tr], y_train[keep_tr], x_test[keep_te])
        yt = y_test[keep_te]
        rows.append({
            "arm": arm,
            "window_pr_auc": float(average_precision_score(yt, scores)),
            "window_roc_auc": float(roc_auc_score(yt, scores)),
            "within_star_auc": within_star_auc(scores, yt, stars_test[keep_te]),
            "n_test_pos": int((yt == 1).sum()), "n_test": int(len(yt)),
        })
        log.info(f"{arm}: window PR-AUC {rows[-1]['window_pr_auc']}, within-star AUC {rows[-1]['within_star_auc']}")

    result = pd.DataFrame(rows)
    result["run_id"] = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else repo_root / "experiments" / "new_task" / "flare_window_eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        result = pd.concat([pd.read_csv(out_path), result], ignore_index=True)
    result.to_csv(out_path, index=False)
    log.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
