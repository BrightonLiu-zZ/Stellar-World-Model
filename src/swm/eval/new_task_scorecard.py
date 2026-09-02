"""Task 2 (plan 2026-07-22, D3/D4) — the new-task probe scorecard.

Scores the expanded probe menu on the frozen-encoder mu the leader arms already produced, one linear
readout per task shape (plan D3):
  detection binary   osc_giant, solar_like_osc, flare(Level A) -- logistic, mean + window_score(MIL),
                     metric PR-AUC (the pool prevalence is inflated, so PR-AUC is the honest read)
  contrastive binary rgb_vs_heb -- logistic on the catalog-only population, metric ROC-AUC (balanced)
  regression         numax_hon, numax_hatt, dnu_hatt, prot_kounkel -- Ridge, metric R2/RMSE/Spearman
                     (numax and dnu regressed in log10, the asteroseismic scaling)

Two further probes ride the frozen v1 packed subset rather than the pool, behind their own flags:
rotation_period (TARS, --with-rotation) and Ijspeert+2024 OBA eclipsing binaries (--with-ijspeert).

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from swm.eval.readout_sweep import (build_model_from_ckpt, cached_mu, fit_readout_scores, pool_stars,
                                    window_score_scores)
from swm.eval.new_task_extract import LEADER_DIR, arm_seed

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


def score_regression(x_train, y_train, x_test, y_test, regressor: str = "ridge",
                     random_state: int = 0) -> tuple[dict, np.ndarray]:
    """Ridge (standardized, CV alpha) on mean-pooled mu; returns (R2/RMSE/Spearman, held-out predictions).

    `regressor="gbm"` swaps in gradient-boosted trees -- used by the A2 engineered-feature ceiling and by
    C3 (roadmap arm 7), never by the frozen-probe headline (the v1 linear-probe lock stands).
    `regressor="mlp"` is C3's second nonlinear family; it is standardized like ridge because a net is not
    scale-invariant, and it is NEVER a headline readout either.
    `random_state` defaults to 0, the value formerly hardcoded, so A2 and every published number that
    rides this function are bit-identical. Ridge is deterministic and ignores it.
    """
    if regressor == "gbm":
        reg = HistGradientBoostingRegressor(random_state=random_state)
        reg.fit(x_train, y_train)  # trees are scale-invariant, no standardization
        pred = reg.predict(x_test)
    elif regressor == "mlp":
        scaler = StandardScaler()
        reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, early_stopping=True,
                           random_state=random_state)
        reg.fit(scaler.fit_transform(x_train), y_train)  # fit the scaler on train only (no leakage)
        pred = reg.predict(scaler.transform(x_test))
    else:
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train)  # learn mean/std on train only (no leakage)
        x_te = scaler.transform(x_test)
        reg = RidgeCV(alphas=np.logspace(-2, 3, 10))  # L2 linear probe, alpha picked by leave-one-out CV
        reg.fit(x_tr, y_train)
        pred = reg.predict(x_te)
    rho = spearmanr(y_test, pred).statistic
    metrics = {"r2": float(r2_score(y_test, pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
               "spearman": float(rho)}
    return metrics, pred


def _emit(sink: list | None, task: str, pooling: str, tics, y, scores) -> None:
    """Record per-star held-out predictions for a scored cell, when the caller asked for them.

    The aggregate CSV cannot support a paired star-bootstrap CI (plan 2026-07-25, decision 10) and cannot
    draw a PR curve; both need the raw test-set predictions, so they are dumped alongside on request.
    """
    if sink is None:
        return
    sink.append({"task": task, "pooling": pooling, "tics": np.asarray(tics),
                 "y": np.asarray(y), "scores": np.asarray(scores)})


def score_detection(mu: dict, labels: pd.DataFrame, column: str, readout: str = "logistic",
                    poolings: tuple[str, ...] = ("mean", "window_score"),
                    sink: list | None = None, random_state: int = 0) -> list[dict]:
    """Detection binary over the whole pool: logistic on mean-pool and on window_score(MIL); PR-AUC + ROC-AUC."""
    train_tics, train_blocks = mu["train"]
    test_tics, test_blocks = mu["test"]
    y_train = y_for(train_tics, labels, column).astype(int)
    y_test = y_for(test_tics, labels, column).astype(int)
    rows = []
    x_train = pool_stars(train_blocks, "mean")
    x_test = pool_stars(test_blocks, "mean")
    for pooling in poolings:
        if pooling == "mean":
            scores = fit_readout_scores(readout, x_train, y_train, x_test, random_state)
        else:
            # window_score keeps its frozen random_state=0: MIL numbers are published and must not gain
            # a seed axis silently. C3 scores poolings=("mean",) only, so this branch is never its path.
            scores = window_score_scores(readout, train_blocks, y_train, test_blocks)
        rows.append({"pooling": pooling, "pr_auc": float(average_precision_score(y_test, scores)),
                     "roc_auc": float(roc_auc_score(y_test, scores)),
                     "n_test_pos": int(y_test.sum()), "n_test": int(len(y_test))})
        _emit(sink, column, pooling, test_tics, y_test, scores)
    return rows


def score_contrastive(mu: dict, labels: pd.DataFrame, readout: str = "logistic",
                      sink: list | None = None, random_state: int = 0) -> list[dict]:
    """RGB(1) vs HeB(0) on the Sreenivas-only population: logistic on mean-pool; ROC-AUC primary."""
    rows = []
    pooled = {}
    y = {}
    kept_tics = None
    for split in ["train", "test"]:
        tics, blocks = mu[split]
        target = labels["rgb_vs_heb"].reindex(tics)
        keep = target.notna().to_numpy()
        feats = pool_stars(blocks, "mean")[keep]
        pooled[split] = feats
        y[split] = target[keep].astype(int).to_numpy()
        if split == "test":
            kept_tics = np.asarray(tics)[keep]
    scores = fit_readout_scores(readout, pooled["train"], y["train"], pooled["test"], random_state)
    rows.append({"pooling": "mean", "roc_auc": float(roc_auc_score(y["test"], scores)),
                 "pr_auc": float(average_precision_score(y["test"], scores)),
                 "n_test_pos": int(y["test"].sum()), "n_test": int(len(y["test"]))})
    _emit(sink, "rgb_vs_heb", "mean", kept_tics, y["test"], scores)
    return rows


def score_regression_task(mu: dict, labels: pd.DataFrame, column: str, log_target: bool,
                          regressor: str = "ridge", sink: list | None = None,
                          random_state: int = 0) -> list[dict]:
    """One regression probe on the catalog-only population (prot capped at 5.7 d), Ridge on mean-pool mu."""
    pooled = {}
    y = {}
    kept_tics = None
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
        if split == "test":
            kept_tics = np.asarray(tics)[keep]
    metrics, pred = score_regression(pooled["train"], y["train"], pooled["test"], y["test"], regressor,
                                     random_state)
    metrics.update({"pooling": "mean", "n_test": int(len(y["test"])), "log_target": log_target})
    _emit(sink, column, "mean", kept_tics, y["test"], pred)
    return [metrics]


def subset_mu(arm: str, device: str, cache_dir: Path | None = None, ckpt_dir: Path | None = None,
              ckpt: str = "best_recon_aux") -> dict:
    """Build/load the v1 packed-subset first-segment mu for one arm (rotation_period + ijspeert ride this).

    exp05 arms already have this cache on disk under experiments/exp05_*/models/B_seed<N>/extracted/, so
    the caller normally hardlinks those in as <arm>.npz and no encoder pass runs here at all.
    `ckpt` defaults to best_recon_aux, the value formerly hardcoded, so every published cache is
    bit-identical; exp09 cells must pass best_recon_only (ADR-0012 A3 -- their sweep axis is the aux
    term, so an aux-bearing selector would compare cells at checkpoints chosen by different rules).
    """
    packed = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
    cache_dir = cache_dir or repo_root / "experiments" / "new_task" / "subset_mu_cache"
    ckpt_dir = ckpt_dir or LEADER_DIR
    cache_path = cache_dir / f"{arm}.npz"
    model = None
    if not cache_path.exists():
        import torch
        from swm.eval.skyline import _make_untrained
        base = torch.load(ckpt_dir / "B_seed0" / f"{ckpt}.pt", map_location="cpu", weights_only=False)
        if arm.startswith("untrained"):
            mc = base["cfg"]["model"]
            init_seed = 0 if arm == "untrained" else arm_seed(arm)  # untrained_s<N>: one init per seed (F17)
            model = _make_untrained(list(mc["enc_channels"]), int(mc["kernel_size"]), int(mc["z_dim"]),
                                    256, int(mc["gru_hidden"]), int(mc["gru_layers"]), device,
                                    seed=init_seed)
        else:
            ck = torch.load(ckpt_dir / f"B_seed{arm_seed(arm)}" / f"{ckpt}.pt",
                            map_location="cpu", weights_only=False)
            model, _ = build_model_from_ckpt(ck, device)
    return cached_mu(cache_path, model, packed, 256, device, desc=f"subset[{arm}]")


def score_ijspeert(arm: str, device: str, cache_dir: Path | None = None, ckpt_dir: Path | None = None,
                   readout: str = "logistic", sink: list | None = None) -> list[dict]:
    """Ijspeert+2024 OBA eclipsing-binary detection on the frozen v1 subset for one arm."""
    return score_ijspeert_from_mu(subset_mu(arm, device, cache_dir, ckpt_dir), readout, sink)


def score_ijspeert_from_mu(mu: dict, readout: str = "logistic", sink: list | None = None,
                           random_state: int = 0) -> list[dict]:
    """Ijspeert+2024 OBA-type eclipsing-binary detection on the frozen v1 subset (plan 2026-07-25).

    Two populations are scored. `ijspeert` uses the whole subset. `ijspeert_excl_villanova` first removes
    every Villanova EB from BOTH classes, so the probe cannot win by re-learning the v1 `eb` task -- 753 of
    the 1,903 corpus Ijspeert TICs are also Villanova EBs, and on the v1 subset the overlap is heavier
    still (74 of 93 test positives). This never touches the `eb` label (ADR-0007).
    """
    cat = pd.read_csv(repo_root / "labels" / "external" / "ijspeert2024_bright.csv")
    positives = set(cat["TIC"].astype(int))
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    villanova = set(canon.loc[pd.to_numeric(canon["eb"], errors="coerce").fillna(0) == 1, "tic_id"])

    rows = []
    for task, drop_eb in [("ijspeert", False), ("ijspeert_excl_villanova", True)]:
        pooled = {}
        y = {}
        kept_tics = None
        for split in ["train", "test"]:
            tics, blocks = mu[split]
            arr = np.array([int(t) for t in tics])
            keep = np.array([t not in villanova for t in arr]) if drop_eb else np.ones(len(arr), bool)
            pooled[split] = pool_stars(blocks, "mean")[keep]
            y[split] = np.array([int(t in positives) for t in arr[keep]])
            if split == "test":
                kept_tics = arr[keep]
        scores = fit_readout_scores(readout, pooled["train"], y["train"], pooled["test"], random_state)
        rows.append({"task": task, "shape": "detection", "pooling": "mean",
                     "pr_auc": float(average_precision_score(y["test"], scores)),
                     "roc_auc": float(roc_auc_score(y["test"], scores)),
                     "n_test_pos": int(y["test"].sum()), "n_test": int(len(y["test"]))})
        _emit(sink, task, "mean", kept_tics, y["test"], scores)
    return rows


def score_rotation_period(arm: str, device: str, cache_dir: Path | None = None, ckpt_dir: Path | None = None,
                          regressor: str = "ridge", sink: list | None = None) -> list[dict]:
    """rotation_period (TARS, P<=5 d) regression on the frozen v1 subset for one trained seed arm."""
    return score_rotation_period_from_mu(subset_mu(arm, device, cache_dir, ckpt_dir), regressor, sink)


def score_rotation_period_from_mu(mu: dict, regressor: str = "ridge", sink: list | None = None,
                                  random_state: int = 0) -> list[dict]:
    """rotation_period (TARS, P<=5 d) regression on the frozen v1 subset, given that subset's mu."""
    canon = pd.read_csv(repo_root / "labels" / "variability_labels_star.csv")
    canon["tic_id"] = canon["tic_id"].astype(int)
    canon["rotation"] = pd.to_numeric(canon["rotation"], errors="coerce").fillna(0).astype(int)
    canon["rotation_period"] = pd.to_numeric(canon["rotation_period"], errors="coerce")
    keep_tics = canon.loc[(canon["rotation"] == 1) & canon["rotation_period"].notna()
                          & (canon["rotation_period"] <= 5), ["tic_id", "rotation_period"]]
    lookup = keep_tics.set_index("tic_id")["rotation_period"]
    pooled = {}
    y = {}
    kept_tics = None
    for split in ["train", "test"]:
        tics, blocks = mu[split]
        target = lookup.reindex(tics)
        keep = target.notna().to_numpy()
        pooled[split] = pool_stars(blocks, "mean")[keep]
        y[split] = target[keep].to_numpy(dtype=float)
        if split == "test":
            kept_tics = np.asarray(tics)[keep]
    metrics, pred = score_regression(pooled["train"], y["train"], pooled["test"], y["test"], regressor,
                                     random_state)
    metrics.update({"pooling": "mean", "n_test": int(len(y["test"])), "log_target": False})
    _emit(sink, "rotation_period", "mean", kept_tics, y["test"], pred)
    return [metrics]


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the new-task probe menu on frozen-encoder mu.")
    ap.add_argument("--arms", nargs="+", default=["seed0", "seed1", "seed2", "untrained"])
    ap.add_argument("--cache-dir", default=None, help="Default: experiments/new_task/mu_cache")
    ap.add_argument("--with-rotation", action="store_true", help="Also score rotation_period on the v1 subset.")
    ap.add_argument("--with-ijspeert", action="store_true", help="Also score Ijspeert+2024 on the v1 subset.")
    ap.add_argument("--subset-cache-dir", default=None,
                    help="v1-subset mu cache for the subset-borne tasks. Default: experiments/new_task/subset_mu_cache")
    ap.add_argument("--ckpt-dir", default=None,
                    help="models/ dir used only if a subset cache must be built. Default: the exp03 leader.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="Default: experiments/new_task/new_task_scorecard.csv")
    ap.add_argument("--dump-scores", default=None,
                    help="Parquet path for per-star held-out predictions (needed by the star-bootstrap CI).")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else repo_root / "experiments" / "new_task" / "mu_cache"
    subset_cache_dir = Path(args.subset_cache_dir) if args.subset_cache_dir else None
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    labels = label_frame()
    rows = []
    dumps = []
    for arm in args.arms:
        sink = [] if args.dump_scores else None
        mu = load_cache(cache_dir / f"{arm}.npz")
        for name, column in DETECTION:
            for cell in score_detection(mu, labels, column, sink=sink):
                rows.append({"arm": arm, "task": name, "shape": "detection", **cell})
        for cell in score_contrastive(mu, labels, sink=sink):
            rows.append({"arm": arm, "task": "rgb_vs_heb", "shape": "contrastive", **cell})
        for name, column, log_target in REGRESSION:
            for cell in score_regression_task(mu, labels, column, log_target, sink=sink):
                rows.append({"arm": arm, "task": name, "shape": "regression", **cell})
        if args.with_rotation:
            for cell in score_rotation_period(arm, args.device, subset_cache_dir, ckpt_dir, sink=sink):
                rows.append({"arm": arm, "task": "rotation_period", "shape": "regression", **cell})
        if args.with_ijspeert:
            for cell in score_ijspeert(arm, args.device, subset_cache_dir, ckpt_dir, sink=sink):
                rows.append({"arm": arm, **cell})
        if sink is not None:
            for entry in sink:
                dumps.append(pd.DataFrame({"arm": arm, "task": entry["task"], "pooling": entry["pooling"],
                                           "tic_id": entry["tics"], "y": entry["y"], "score": entry["scores"]}))
        log.info(f"scored arm {arm}")

    if dumps:
        dump_path = Path(args.dump_scores)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(dumps, ignore_index=True).to_parquet(dump_path, index=False)
        log.info(f"wrote per-star scores {dump_path}")

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
