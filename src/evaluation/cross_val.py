"""
K-fold cross-validation runner.

Fold strategy: StratifiedGroupKFold
  - Stratified  : each fold preserves the global class distribution, preventing
                  a fold from being dominated by a single label.
  - Group-aware : all SubWindow_3s rows that share the same Window_10s are kept
                  in the same fold.  Without this, the model would see correlated
                  sub-windows from the same 10-second epoch in both train and
                  test — effectively a form of data leakage.

Scaler discipline: StandardScaler is fit exclusively on the training portion of
each fold and then applied to the test portion.  Fitting on the full dataset
before splitting would leak test-set statistics into the scaler, inflating
reported accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import N_FOLDS, CLASSES
from src.data.preprocess import fit_scaler, apply_scaler


def run_cross_val(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Run stratified-group k-fold CV for a single model instance.

    Parameters
    ----------
    model   : instance of BaseClassifier
    X       : feature matrix (cleaned, not scaled)
    y       : label Series
    groups  : Window_10s Series for GroupKFold grouping
    n_folds : number of folds (default from config)

    Returns a list of per-fold result dicts.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx],        y.iloc[test_idx]

        if model.requires_scaling():
            scaler, X_train = fit_scaler(X_train)
            X_test = apply_scaler(scaler, X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_results.append({
            "fold":        fold,
            "accuracy":    accuracy_score(y_test, y_pred),
            "f1_macro":    f1_score(y_test, y_pred, average="macro",    zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "y_test":      y_test,
            "y_pred":      y_pred,
        })

        print(
            f"    Fold {fold}/{n_folds}  |  "
            f"acc={fold_results[-1]['accuracy']:.4f}  "
            f"f1_macro={fold_results[-1]['f1_macro']:.4f}"
        )

    return fold_results


def aggregate_results(fold_results: list[dict]) -> dict:
    """Compute mean/std metrics and pool predictions across all folds."""
    accs = [r["accuracy"]    for r in fold_results]
    f1s  = [r["f1_macro"]    for r in fold_results]
    wf1s = [r["f1_weighted"] for r in fold_results]

    all_y_test = pd.concat([r["y_test"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])

    return {
        "accuracy_mean":      np.mean(accs),
        "accuracy_std":       np.std(accs),
        "f1_macro_mean":      np.mean(f1s),
        "f1_macro_std":       np.std(f1s),
        "f1_weighted_mean":   np.mean(wf1s),
        "f1_weighted_std":    np.std(wf1s),
        "classification_report": classification_report(
            all_y_test, all_y_pred, labels=CLASSES, zero_division=0
        ),
        "confusion_matrix":  confusion_matrix(all_y_test, all_y_pred, labels=CLASSES),
        "fold_results":      fold_results,
    }
