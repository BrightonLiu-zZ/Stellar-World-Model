"""Stage 2 step 2: the frozen linear probe and the A-vs-B results table.

For each primary task (transit, eb, pulsating) fit a logistic regression on the per-star mu vectors
of the train split and evaluate on the test split, both drawn from the frozen star-disjoint split.
Features are standardized on the train split only. PR-AUC is the primary separability metric under
imbalance; ROC-AUC, F1, and macro-F1 are reported alongside. Results are written per run and upserted
into a single results_table.csv so running A then B builds the headline comparison.

Run (from repo src/ on PYTHONPATH, in the swm env):
    python -m swm.eval.probe variant=B seed=0
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

primary_tasks = ["transit", "eb", "pulsating"]


def probe_one_task(data: pd.DataFrame, mu_cols: list[str], task: str) -> dict[str, float] | None:
    """
    Fit one logistic-regression probe for a single binary task and score it on the test split.
    Standardizes features on train only (no leakage), uses balanced class weights for the rare
    positives, and returns the separability metrics. Returns None if either split lacks both classes.
    """
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    y_train = train[task].to_numpy()
    y_test = test[task].to_numpy()
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        log.warning(f"task {task}: a split lacks both classes; skipping")
        return None

    x_train = train[mu_cols].to_numpy()
    x_test = test[mu_cols].to_numpy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train) # learn mean/std on train only
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000) # balanced: upweight rare positives
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_test)[:, 1] # P(positive)
    pred = (proba >= 0.5).astype(int)

    return {
        "task": task,
        "n_train_pos": int(y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test": int(len(y_test)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_name = f"{cfg.variant_name}_seed{cfg.seed}"
    star_mu_path = Path(cfg.paths.models_dir) / run_name / "extracted" / "star_mu.parquet"
    assert star_mu_path.exists(), f"run swm.eval.extract for {run_name} first; missing {star_mu_path}"
    star_mu = pd.read_parquet(star_mu_path)

    subset_path = Path(cfg.paths.subset_dir) / "subset_tics.parquet"
    assert subset_path.exists(), f"run swm.data.subset first; missing {subset_path}"
    labels = pd.read_parquet(subset_path)

    mu_cols = []
    for col in star_mu.columns:
        if col.startswith("mu"):
            mu_cols.append(col)
    data = star_mu.merge(labels[["tic_id"] + primary_tasks], on="tic_id", how="inner")

    rows = []
    for task in primary_tasks:
        result = probe_one_task(data, mu_cols, task)
        if result is None:
            continue
        result["variant"] = cfg.variant_name
        result["seed"] = int(cfg.seed)
        rows.append(result)

    results = pd.DataFrame(rows)
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / f"probe_{run_name}.csv", index=False)
    log.info(f"[{run_name}] probe results:\n{results}")

    combined_path = results_dir / "results_table.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        keep = ~((existing["variant"] == cfg.variant_name) & (existing["seed"] == int(cfg.seed)))
        combined = pd.concat([existing[keep], results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(combined_path, index=False)
    log.info(f"updated {combined_path}")


if __name__ == "__main__":
    main()
