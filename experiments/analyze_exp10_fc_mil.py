"""F-C, MIL half (exp10 forensics) -- does the frozen window_score operator add to the engineered features?

THE QUESTION. F-C's other variants ask whether a richer per-star SUMMARY of mu (std, quantiles, max)
beats the published mean. This half asks the sharper version: does the localized channel survive when
mu is reduced all the way to ONE number per star produced by the MIL operator, and is that number
something the 25 engineered features do not already have? `eb` and `transit` only, because those are the
two tasks the MIL work ever found a pooling gain on (ADR-0008-lite).

THE OPERATOR IS FROZEN. `window_score` fits a logistic readout on window-level mu with each star's label
broadcast to its windows and takes the max over a star's windows. Its random_state stays 0: those
numbers are published and must not gain a seed axis. Only the OUTER readout (the one that sees
`features (+) window_score`) carries a seed.

THE ONE DEVIATION, stated rather than buried. Concatenating the MIL score as a column needs a value for
the TRAIN stars too, and the published operator's train scores are in-sample -- the outer readout would
see a column that is honest on test and inflated on train, which biases the fusion arm in its own
favour. The train column is therefore built OUT-OF-FOLD (5 folds over stars, so a star's windows never
straddle a fold), while the TEST column is the published operator verbatim. The `ws_only` arm reported
beside it is the published MIL number itself and is unaffected.

UNTRAINED TWIN, mandatory. The MIL work measured the untrained encoder gaining +0.066 on transit from
the pooling operator ALONE. Without its twin this forensic cannot tell an operator effect from a
representation effect, so every arm here has one.

DECISION RULE (pre-registered 2026-08-29, and shared with the window-statistic half):
    R-C1  a variant beats `features (+) mean` by > 2*SE on `transit` or `eb` under GBM AND its gain
          exceeds its untrained twin's gain --> the localized channel is a real mu asset that
          mean-pooling destroys. Otherwise the localized story stays transit-only at `mean`, as now.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, ~30 min with --jobs 6):
    PYTHONUNBUFFERED=1 python experiments/analyze_exp10_fc_mil.py
    python experiments/analyze_exp10_fc_mil.py --arms hann0p3_fbwd_s0 untrained --tasks eb
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import align_features, arm_parts, load_mu_cache, stacked  # noqa: E402
from swm.eval.new_task_ceiling import cached_subset_features  # noqa: E402
from swm.eval.readout_sweep import fit_readout_scores, window_score_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_fc_mil")

source_home = repo_root / "experiments" / "exp08_menu_channel" / "subset_mu_cache"
out_home = repo_root / "experiments" / "exp10_forensics" / "fc_mil"
default_arms = [f"hann0p3_fbwd_s{s}" for s in range(6)] + ["untrained"]
n_folds = 5  # out-of-fold folds for the TRAIN column only; the test column is the published operator


def v1_labels(tics: dict[str, list[int]], task: str) -> dict[str, np.ndarray]:
    """Star-level 0/1 labels for one v1 task, in the cache's own star order."""
    frame = (pd.read_parquet(repo_root / "experiments" / "exp06_features_cache.parquet")
             [["tic_id", task]].drop_duplicates("tic_id").set_index("tic_id"))
    out = {}
    for split in ["train", "test"]:
        values = frame[task].reindex(tics[split])
        assert values.notna().all(), f"{task}/{split}: v1 label table does not cover the cache stars"
        out[split] = values.to_numpy(dtype=int)
    return out


def oof_train_column(blocks: list[np.ndarray], y: np.ndarray, seed: int) -> np.ndarray:
    """
    The MIL window score for every TRAIN star, computed on folds that never saw that star.
    Splitting on stars rather than windows is the point: a star whose other windows trained the operator
    would get an in-sample score, which is exactly the inflation this column exists to avoid.
    """
    scores = np.zeros(len(blocks), dtype=float)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_index, held_index in splitter.split(np.zeros(len(y)), y):
        fit_blocks = []
        for i in fit_index:
            fit_blocks.append(blocks[i])
        held_blocks = []
        for i in held_index:
            held_blocks.append(blocks[i])
        scores[held_index] = window_score_scores("logistic", fit_blocks, y[fit_index], held_blocks)
    return scores


def score_arm_task(arm: str, task: str, feats: dict, cache_dir: Path) -> list[dict]:
    """
    Every arm set for one (encoder arm, task): features alone, the MIL score alone, and their fusion.
    Returns one row per readout family; the outer readout's random_state is the encoder seed, which is
    what keeps the fusion delta paired against the engineered arm (the C3b fix).
    """
    mu = load_mu_cache(cache_dir / f"{arm}.npz")
    tics = {"train": list(mu["train"][0]), "test": list(mu["test"][0])}
    aligned = align_features(feats, mu)  # reads the star list only, so the per-window cache is fine here
    x_feat = {"train": stacked(aligned, "train"), "test": stacked(aligned, "test")}
    y = v1_labels(tics, task)
    _, seed = arm_parts(arm)

    ws = {"train": oof_train_column(mu["train"][1], y["train"], seed).reshape(-1, 1),
          "test": window_score_scores("logistic", mu["train"][1], y["train"],
                                      mu["test"][1]).reshape(-1, 1)}

    tables = {"features_only": x_feat, "ws_only": ws,
              "features_plus_ws": {"train": np.concatenate([x_feat["train"], ws["train"]], axis=1),
                                   "test": np.concatenate([x_feat["test"], ws["test"]], axis=1)}}
    rows = []
    for readout_family, readout in [("linear", "logistic"), ("gbm", "gbm")]:
        for arm_set, table in tables.items():
            scores = fit_readout_scores(readout, table["train"], y["train"], table["test"], seed)
            rows.append({"arm": arm, "seed": seed, "task": task, "readout_family": readout_family,
                         "arm_set": arm_set, "n_col": table["train"].shape[1],
                         "pr_auc": float(average_precision_score(y["test"], scores)),
                         "n_test": int(len(y["test"])), "n_test_pos": int(y["test"].sum())})
    return rows


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Fusion delta over the engineered arm, paired per seed, per family and per encoder arm family."""
    rows = []
    for (task, readout_family), group in probe.groupby(["task", "readout_family"], sort=False):
        base = group[group["arm_set"] == "features_only"].set_index("arm")["pr_auc"]
        for arm_family, sub in group.groupby(group["arm"].str.replace(r"_s\d+$", "", regex=True),
                                             sort=False):
            for arm_set in ["ws_only", "features_plus_ws"]:
                cells = sub[sub["arm_set"] == arm_set]
                if cells.empty:
                    continue
                deltas = []
                for _, cell in cells.iterrows():
                    deltas.append(float(cell["pr_auc"]) - float(base[cell["arm"]]))
                values = np.array(deltas)
                sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                rows.append({"task": task, "readout_family": readout_family, "arm_family": arm_family,
                             "arm_set": arm_set, "n_seeds": len(values),
                             "features_only": float(base[cells["arm"]].mean()),
                             "score_mean": float(cells["pr_auc"].mean()),
                             "delta_mean": float(values.mean()), "delta_sd": sd,
                             "delta_2se": 2 * sd / np.sqrt(len(values)) if len(values) > 1 else np.nan,
                             "n_test": int(cells["n_test"].max()),
                             "n_test_pos": int(cells["n_test_pos"].max())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="F-C MIL half: the frozen window_score operator in fusion.")
    ap.add_argument("--arms", nargs="+", default=default_arms)
    ap.add_argument("--tasks", nargs="+", default=["eb", "transit"], choices=["eb", "transit"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else source_home
    out_dir = Path(args.out_dir) if args.out_dir else out_home
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = cached_subset_features(repo_root / "experiments" / "exp01_window256_seq16" / "packed")
    cells = [(arm, task) for arm in args.arms for task in args.tasks]
    results = Parallel(n_jobs=args.jobs)(
        delayed(score_arm_task)(arm, task, feats, cache_dir) for arm, task in
        tqdm(cells, desc="arm x task", total=len(cells)))

    rows = []
    for block in results:
        rows.extend(block)
    probe = pd.DataFrame(rows)
    probe["cache_dir"] = str(cache_dir)
    probe.to_csv(out_dir / "fc_mil_probe.csv", index=False)

    summary = summarize(probe)
    summary.to_csv(out_dir / "fc_mil_summary.csv", index=False)
    print("\nF-C MIL: PR-AUC delta over the 25 engineered features, readout mean:")
    print(summary[["task", "readout_family", "arm_family", "arm_set", "features_only", "score_mean",
                   "delta_mean", "delta_2se"]].round(4).to_string(index=False))
    log.info(f"wrote {out_dir}/fc_mil_{{probe,summary}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
