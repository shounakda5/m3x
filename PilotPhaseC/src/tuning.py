"""
tuning.py -- Lightweight hyperparameter search, run once before the full
cross-validation, so the CV + final model aren't just trained on whatever
defaults happened to be written down first.

Deliberately cheap: a handful of candidates (config.TUNE_GRID), each
trained on ONE group-aware train/val split (not full CV -- that would
multiply the cost by n_folds for comparatively little extra signal at this
dataset size) at a reduced epoch/patience budget (config.TUNE_EPOCHS/
TUNE_PATIENCE). The winning candidate is then handed to the caller, which
uses it for the full CV and final-model training at the full epoch budget.

The validation split respects the CV group key (Survey_ID) via
GroupShuffleSplit, for the same leakage-prevention reason the main CV uses
grouped folds -- a tuning decision made by peeking at leaked validation
data would be exactly the kind of "wrong ML habit" this phase is trying to
avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import CLASSES, TUNE_GRID, TUNE_EPOCHS, TUNE_PATIENCE, RANDOM_STATE
from PilotPhaseC.src.common import (
    fit_channel_scaler, apply_channel_scaler,
    fit_target_scaler, apply_target_scaler, invert_target_scaler,
    classifier_sample_weights, regressor_sample_weights,
)
from PilotPhaseC.src.models import CNNClassifier, CNNRegressor


def _val_split(groups, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    idx = np.arange(len(groups))
    tr_idx, va_idx = next(gss.split(idx, groups=groups))
    return tr_idx, va_idx


def tune_classifier(
    X, y, groups, base_params: dict, random_state: int = RANDOM_STATE,
    period=None, lineage: str | None = None,
    classes: list[str] | None = None,
    class_names: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    tr_idx, va_idx = _val_split(groups, random_state)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    g_tr, g_va = groups[tr_idx], groups[va_idx]
    p_tr = period[tr_idx] if period is not None else None

    mean, std = fit_channel_scaler(X_tr)
    X_tr_s = apply_channel_scaler(X_tr, mean, std)
    X_va_s = apply_channel_scaler(X_va, mean, std)

    metric_classes = classes or CLASSES
    sw_tr = classifier_sample_weights(g_tr, y_tr, p_tr, lineage, classes=metric_classes)

    log = []
    best_params, best_score = None, -1.0
    for i, override in enumerate(TUNE_GRID):
        params = {**base_params, **override}
        model = CNNClassifier(params, class_names=class_names or metric_classes)
        model.fit(
            X_tr_s, y_tr, epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            sample_weight=sw_tr, groups=g_tr,
        )
        y_pred = model.predict(X_va_s)
        score = f1_score(y_va, y_pred, average="macro", labels=metric_classes, zero_division=0)
        log.append({"candidate": i, "override": override, "val_f1_macro_window": score})
        print(f"    [tune] candidate {i} {override or '(defaults)'} -> window val F1 macro = {score:.4f}")
        if score > best_score:
            best_score, best_params = score, params

    return best_params, log


def tune_regressor(
    X, y, groups, base_params: dict, random_state: int = RANDOM_STATE,
    period=None, lineage: str | None = None,
) -> tuple[dict, list[dict]]:
    tr_idx, va_idx = _val_split(groups, random_state)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    g_tr, g_va = groups[tr_idx], groups[va_idx]
    p_tr = period[tr_idx] if period is not None else None

    mean, std = fit_channel_scaler(X_tr)
    X_tr_s = apply_channel_scaler(X_tr, mean, std)
    X_va_s = apply_channel_scaler(X_va, mean, std)

    t_mean, t_std = fit_target_scaler(y_tr)
    y_tr_s = apply_target_scaler(y_tr, t_mean, t_std)
    sw_tr = regressor_sample_weights(g_tr, p_tr, lineage)

    log = []
    best_params, best_score = None, np.inf
    for i, override in enumerate(TUNE_GRID):
        params = {**base_params, **override}
        model = CNNRegressor(params)
        model.fit(
            X_tr_s, y_tr_s, epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            sample_weight=sw_tr, groups=g_tr,
        )
        y_pred = invert_target_scaler(model.predict(X_va_s), t_mean, t_std)
        rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
        log.append({"candidate": i, "override": override, "val_rmse_window": rmse})
        print(f"    [tune] candidate {i} {override or '(defaults)'} -> window val RMSE = {rmse:.4f}")
        if rmse < best_score:
            best_score, best_params = rmse, params

    return best_params, log
