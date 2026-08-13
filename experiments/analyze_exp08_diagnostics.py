"""exp08 diagnostics: the three quantities the audit notebook needs and no artifact on disk holds.

The exp08 ladder writeup rests on 36 runs whose probe scores, menu scores and signature statistics are
all already cached as CSV. Three things it asserts are NOT cached anywhere, and each one gates a
different attack-list item:

  selection   Which epoch did each run actually ship? The probes read `best_recon_aux.pt`, whose epoch
              lives inside the checkpoint and nowhere else. Attack #6 (selection sanity: is the shipped
              epoch past beta warmup, do the 1-unit smooth arms really show 1 unit) and attack #5 (stop
              epoch vs probe score within a cell, the cosine-LR-exposure worry) both need that epoch
              joined against the run's own W&B curve.

  fingerprint Did every menu arm read its own checkpoint? `new_task_scorecard` takes a single
              --ckpt-dir for the whole fan, so a missing subset cache is silently rebuilt from the wrong
              cell's weights (the trap run_exp08_prechecks.ps1 documents). The caches carry no
              provenance fields, so identity has to be reconstructed: a content hash catches any two
              arms sharing weights, and the per-arm latent width ties each cache back to the arm whose
              curve it should match.

  refit       Are the published probe numbers reproducible from the cached mu, or only from the summary
              CSV that was written beside them? This stage re-fits the v1 probe on the two load-bearing
              cells (exp08_linear, exp08_frozen, the arms carrying "linear ~ GRU" and "frozen ~ GRU")
              with the estimator re-implemented here from its protocol description rather than imported,
              so a bug in the shared helper cannot reproduce itself.

Everything else the notebook does is seconds of pandas over CSVs that already exist, and it does it
inline where the arithmetic is visible: an audit whose SEs are computed behind an import is not an audit.

Run (repo root, swm env, PYTHONPATH=src):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp08_diagnostics.py
    python experiments/analyze_exp08_diagnostics.py --stages selection --cells exp08_linear
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]

LADDER_CELLS = ["exp08_smooth", "exp08_smooth_half", "exp08_smooth_lo",
                "exp08_linear", "exp08_frozen", "exp08_frozen_lo"]
END_CELLS = ["exp07_hann0p3_off", "exp07_hann0p3_fbwd"] # the ladder ends, reused from exp07
SEEDS = [0, 1, 2, 3, 4, 5]
REFIT_CELLS = ["exp08_linear", "exp08_frozen"] # the two arms carrying the equivalence claims
TASKS = ("pulsating", "eb", "rotation", "transit")
AMPLITUDE_COLS = ["p2p_scatter_ratio", "depth_5_95", "mad", "iqr"] # the basis mean_resid removes

CURVE_DIRS = {"exp08": ROOT / "experiments" / "exp08_forensics" / "curves_exp08",
              "exp07": ROOT / "experiments" / "exp07_forensics" / "curves_exp07"}
SHARED_MU = ROOT / "experiments" / "exp07_forensics" / "mu_cache" # v1 probe mu, all exp08 cells
SUBSET_MU = ROOT / "experiments" / "exp08_prechecks" / "subset_mu_cache" # menu v1-subset mu
POOL_MU = ROOT / "experiments" / "exp08_prechecks" / "mu_cache" # menu new-task pool mu

# A cache dim counts as carrying signal when its across-star spread clears this. The training-time
# n_active_units is a KL criterion instead, so the two counts are related but not equal; only the
# fully collapsed arms (1 unit) are expected to agree exactly.
ACTIVE_STD = 0.05


def curve_path(cell: str, seed: int) -> Path:
    """Locate a run's W&B per-epoch dump, which lives under its own experiment's forensics dir."""
    prefix = cell.split("_")[0]
    return CURVE_DIRS[prefix] / f"{cell}_B_seed{seed}.csv"


def cell_config(cell: str) -> dict:
    """Read a cell's generated Hydra yaml so lambda / dyn_mode / warmup come from the config, not a table."""
    exp = cell.split("_")[0]
    path = ROOT / "src" / "swm" / "configs" / "experiment" / exp / f"{cell}.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ----------------------------------------------------------------------------------------------------
# stage: selection - which epoch shipped, and what the run looked like there
# ----------------------------------------------------------------------------------------------------
def selection_row(cell: str, seed: int, ckpt: str, curve: pd.DataFrame, cfg: dict) -> dict | None:
    """
    Read one checkpoint's shipped epoch and describe the run at that epoch from its own curve.
    `best_recon_aux` is the checkpoint every exp08 probe and mu cache was built from; `best` is carried
    alongside because the two disagree by up to 75 epochs and the notebook states that as a caveat.
    """
    path = ROOT / "experiments" / cell / "models" / f"B_seed{seed}" / f"{ckpt}.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False) # weights stay on CPU, only metadata is read
    epoch = int(state["epoch"])
    row = {"cell": cell, "seed": seed, "ckpt": ckpt, "epoch": epoch,
           "best_val": float(state["best_val"]), "best_select": float(state["best_select"]),
           "lambda_dyn": cfg["train"]["lambda_dyn"], "dyn_mode": cfg["model"].get("dyn_mode", "fwd_bwd"),
           "beta_warmup_epochs": cfg["train"]["beta_warmup_epochs"],
           "max_epochs": cfg["train"]["max_epochs"], "patience": cfg["train"]["patience"],
           "first_epoch": int(curve["epoch"].min()), "last_epoch": int(curve["epoch"].max()),
           "n_curve_rows": len(curve)}
    at = curve[curve["epoch"] == epoch]
    row["curve_has_epoch"] = len(at) == 1
    if len(at) == 1:
        for col, name in [("val/n_active_units", "val_n_active_units"), ("val/monitor_recon_aux", "val_monitor_recon_aux"),
                          ("val/recon", "val_recon"), ("val/dyn", "val_dyn"), ("val/aux", "val_aux"),
                          ("val/mu_var", "val_mu_var"), ("lr", "lr")]:
            row[name] = float(at[col].iloc[0])
    return row


def run_selection_stage(cells: list[str], seeds: list[int], out: Path) -> None:
    """Harvest the shipped epoch of both checkpoints for every run and join it to that run's curve."""
    rows = []
    jobs = []
    for cell in cells:
        for seed in seeds:
            jobs.append((cell, seed))
    for cell, seed in tqdm(jobs, desc="selection", total=len(jobs)):
        path = curve_path(cell, seed)
        assert path.exists(), f"missing curve dump {path}"
        curve = pd.read_csv(path)
        # F22 wants epoch 0 present, but one reused exp07 reference run was resumed and its dump lost
        # the prefix. That is a provenance fact the notebook reports, so it is recorded, not asserted.
        cfg = cell_config(cell)
        for ckpt in ["best_recon_aux", "best"]:
            row = selection_row(cell, seed, ckpt, curve, cfg)
            if row is not None:
                rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out} ({len(rows)} rows)")


# ----------------------------------------------------------------------------------------------------
# stage: fingerprint - is each cached mu the arm it claims to be
# ----------------------------------------------------------------------------------------------------
def fingerprint_row(kind: str, path: Path, mu_key: str, tic_key: str) -> dict:
    """
    Summarise one cached mu by content hash and latent width.
    The hash makes a wrong-checkpoint fallback visible as two arms sharing bytes; the width ties the
    cache back to the arm whose training curve reports a matching number of surviving units.
    """
    cached = np.load(path)
    mu = cached[mu_key]
    spread = mu.std(axis=0)
    digest = hashlib.sha1(np.ascontiguousarray(mu).tobytes()).hexdigest()
    return {"kind": kind, "name": path.stem, "n_stars": mu.shape[0], "z_dim": mu.shape[1],
            "n_tics": len(cached[tic_key]), "sha1": digest,
            "dims_active": int((spread > ACTIVE_STD).sum()), "dims_any": int((spread > 0.01).sum()),
            "spread_max": float(spread.max()), "spread_median": float(np.median(spread)),
            "total_var": float((spread ** 2).sum())}


def run_fingerprint_stage(out: Path) -> None:
    """Fingerprint every mu cache the exp08 tables were computed from, across all three cache dirs."""
    jobs = []
    for path in sorted(SUBSET_MU.glob("*.npz")):
        jobs.append(("subset", path, "test_mu", "test_tics"))
    for path in sorted(POOL_MU.glob("*.npz")):
        jobs.append(("pool", path, "test_mu", "test_tics"))
    for path in sorted(SHARED_MU.glob("exp0*.npz")):
        jobs.append(("shared", path, "mean_test", "tics_test"))
    rows = []
    for kind, path, mu_key, tic_key in tqdm(jobs, desc="fingerprint", total=len(jobs)):
        rows.append(fingerprint_row(kind, path, mu_key, tic_key))
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out} ({len(rows)} rows)")


# ----------------------------------------------------------------------------------------------------
# stage: refit - reproduce the published v1 probe from cached mu with an independent estimator
# ----------------------------------------------------------------------------------------------------
def probe_pr_auc(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, test_y: np.ndarray) -> float:
    """
    Fit the v1 linear probe and score it, re-implemented from the protocol rather than imported.
    Protocol: standardize on train only, logistic regression with balanced class weights, PR-AUC on test.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x) # mean/std learned on train only (no leakage)
    x_test = scaler.transform(test_x)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x_train, train_y)
    scores = clf.predict_proba(x_test)[:, 1]
    return float(average_precision_score(test_y, scores))


def amplitude_residual(train_x: np.ndarray, test_x: np.ndarray, train_amp: np.ndarray,
                       test_amp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove what the amplitude basis linearly predicts from every latent dim, fitted on train only."""
    fitter = LinearRegression().fit(train_amp, train_x)
    return train_x - fitter.predict(train_amp), test_x - fitter.predict(test_amp)


def refit_run(cell: str, seed: int, labels: pd.DataFrame) -> list[dict]:
    """Re-fit all three v1 readouts for one run from its cached star-level mu."""
    cached = np.load(SHARED_MU / f"{cell}_seed{seed}.npz")
    train = pd.DataFrame({"tic_id": cached["tics_train"]}).merge(labels, on="tic_id", how="inner")
    test = pd.DataFrame({"tic_id": cached["tics_test"]}).merge(labels, on="tic_id", how="inner")
    assert len(train) == len(cached["tics_train"]) and len(test) == len(cached["tics_test"]), \
        f"{cell} seed{seed}: a cached star is missing from the feature cache"
    mean_train, mean_test = cached["mean_train"], cached["mean_test"]
    std_train, std_test = cached["std_train"], cached["std_test"]
    resid_train, resid_test = amplitude_residual(mean_train, mean_test,
                                                 train[AMPLITUDE_COLS].to_numpy(),
                                                 test[AMPLITUDE_COLS].to_numpy())
    arms = {"mean": (mean_train, mean_test),
            "mean_resid": (resid_train, resid_test),
            "mean_std": (np.concatenate([mean_train, std_train], axis=1),
                         np.concatenate([mean_test, std_test], axis=1))}
    rows = []
    for pooling, (x_train, x_test) in arms.items():
        for task in TASKS:
            pr_auc = probe_pr_auc(x_train, x_test, train[task].to_numpy(), test[task].to_numpy())
            rows.append({"cell": cell, "seed": seed, "pooling": pooling, "task": task, "pr_auc": pr_auc})
    return rows


def run_refit_stage(cells: list[str], seeds: list[int], labels: pd.DataFrame, out: Path) -> None:
    """Re-fit the load-bearing cells so the published summary can be checked against a second estimator."""
    jobs = []
    for cell in cells:
        for seed in seeds:
            jobs.append((cell, seed))
    rows = []
    for cell, seed in tqdm(jobs, desc="refit", total=len(jobs)):
        rows.extend(refit_run(cell, seed, labels))
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info(f"wrote {out} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="exp08 diagnostics: shipped epochs, cache provenance, probe re-fit")
    ap.add_argument("--stages", nargs="+", default=["selection", "fingerprint", "refit"])
    ap.add_argument("--cells", nargs="+", default=LADDER_CELLS + END_CELLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--refit-cells", nargs="+", default=REFIT_CELLS)
    ap.add_argument("--feature-cache", default="experiments/exp06_features_cache.parquet")
    ap.add_argument("--out-prefix", default="experiments/exp08_diag")
    ap.add_argument("--force", action="store_true", help="recompute stages whose output already exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    outputs = {"selection": Path(f"{args.out_prefix}_selection.csv"),
               "fingerprint": Path(f"{args.out_prefix}_cache_fingerprint.csv"),
               "refit": Path(f"{args.out_prefix}_refit.csv")}
    for stage, out in outputs.items():
        if stage not in args.stages:
            continue
        if out.exists() and not args.force:
            log.info(f"skip {stage}: {out} exists (--force to recompute)")
            continue
        if stage == "selection":
            run_selection_stage(args.cells, args.seeds, out)
        if stage == "fingerprint":
            run_fingerprint_stage(out)
        if stage == "refit":
            labels = pd.read_parquet(ROOT / args.feature_cache)[["tic_id", *TASKS, *AMPLITUDE_COLS]]
            run_refit_stage(args.refit_cells, args.seeds, labels, out)


if __name__ == "__main__":
    main()
