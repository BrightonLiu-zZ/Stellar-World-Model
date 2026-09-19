"""XGBoost readout for the wave-3 nonlinear baseline (Yue Ma revision plan 2026-09-15, item 2).

The question it answers: does the fusion gain survive when the readout on the engineered features is
allowed to be nonlinear? sklearn's HistGradientBoosting already answers it (`gbm` family, C3b); this is
the same measurement under the estimator she asked for by name, so the paper can print whichever of
the two gives the engineered arm its stronger score.

Recipe, fixed in advance and identical for every arm (features / mu / features (+) mu):
    grid        max_depth {3, 5} x learning_rate {0.05, 0.1}   (her plan's grid, nothing else)
    budget      up to 1000 rounds, early stopping after 50 rounds without validation improvement
    validation  25 % of the TRAIN split, stratified for classification, keyed on `random_state` so both
                arms of one seed are selected on the same rows; the test split is never touched
    selection   the config with the best validation score (PR-AUC / RMSE); its early-stopped model is
                the one that predicts the test rows -- no refit on the full train split, same as F-F
    imbalance   scale_pos_weight = n_neg / n_pos on the fitting rows
Trees are scale-invariant, so nothing is standardised.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

GRID = [{"max_depth": d, "learning_rate": lr} for d in (3, 5) for lr in (0.05, 0.1)]
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.25


def xgb_predict(shape: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                random_state: int = 0) -> np.ndarray:
    """Held-out predictions from the grid-selected, early-stopped XGBoost model.

    `shape` is "classify" (returns P(positive)) or "regress" (returns the prediction).
    """
    if shape not in ("classify", "regress"):
        raise ValueError(f"unknown shape {shape}")
    stratify = y_train if shape == "classify" else None
    x_fit, x_val, y_fit, y_val = train_test_split(  # (n_fit, D), (n_val, D)
        x_train, y_train, test_size=VAL_FRACTION, random_state=random_state, stratify=stratify)
    best_score, best_model = -np.inf, None
    for config in GRID:
        if shape == "classify":
            n_pos = int(y_fit.sum())
            model = XGBClassifier(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                  eval_metric="aucpr", scale_pos_weight=(len(y_fit) - n_pos) / max(n_pos, 1),
                                  tree_method="hist", random_state=random_state, n_jobs=8,
                                  verbosity=0, **config)
        else:
            model = XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                 eval_metric="rmse", tree_method="hist", random_state=random_state,
                                 n_jobs=8, verbosity=0, **config)
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        score = float(model.best_score)  # aucpr (higher better) or rmse (lower better)
        if shape == "regress":
            score = -score
        if score > best_score:
            best_score, best_model = score, model
    if best_model is None:
        raise RuntimeError("no XGBoost config produced a validation score")
    if shape == "classify":
        return best_model.predict_proba(x_test)[:, 1]
    return best_model.predict(x_test)
