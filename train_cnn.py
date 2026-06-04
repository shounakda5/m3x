"""
train_cnn.py — Train the 1D CNN on raw M3X sensor data.

The CNN operates on raw time-series windows (not on the engineered features
used by SVM / RF / XGBoost).  Each window is 112 samples × 9 channels
(≈ 4.5 s at 25 Hz).  Two windows are extracted per duty-cycle burst,
mirroring the two sub-windows of the feature-based pipeline.

Usage
-----
    python train_cnn.py
    python train_cnn.py --output_dir results/cnn_raw_baseline
    python train_cnn.py --s1 data/raw/session1_raw.csv \\
                        --s2 data/raw/session2_raw.csv \\
                        --labeled data/features/features_labeled_new.csv

CV strategy : 5-fold StratifiedGroupKFold, grouping by Block_ID (same
              policy as the feature-based models — all windows from one
              duty-cycle burst stay in the same fold).

Scaling     : per-channel z-normalisation fitted on each training fold,
              applied to the corresponding test fold.  The final scaler
              (fitted on the full dataset) is saved alongside the model.

Outputs (in --output_dir)
-------------------------
    fold_results.csv          per-fold accuracy and F1
    cnn_1d_model.pkl          serialized Keras model (joblib)
    cnn_1d_scaler.npz         per-channel mean and std (numpy)
    cnn_1d_model.tflite       TFLite float32 graph for MCU deployment
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from config import (
    CLASSES,
    LABELED_FEATURES_FILE,
    N_FOLDS,
    RAW_SESSION1_FILE,
    RAW_SESSION2_FILE,
    RESULTS_DIR,
)
from src.data.raw_preprocess import load_raw_cnn_dataset
from src.models.cnn_model import CNNClassifier


# ---------------------------------------------------------------------------
# Per-channel z-normalisation (fit on train, apply to test)
# ---------------------------------------------------------------------------

def _fit_channel_scaler(X_train: np.ndarray):
    """
    Compute per-channel mean and std over all training windows and timesteps.

    X_train : (N, T, C)
    Returns  : (mean, std)  each (1, 1, C)
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std( axis=(0, 1), keepdims=True)
    std  = np.where(std == 0, 1.0, std)
    return mean, std


def _apply_channel_scaler(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_raw_cross_val(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """StratifiedGroupKFold CV for the raw CNN."""
    le   = LabelEncoder().fit(y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        mean, std = _fit_channel_scaler(X_tr)
        X_tr = _apply_channel_scaler(X_tr, mean, std)
        X_te = _apply_channel_scaler(X_te, mean, std)

        model = CNNClassifier()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro",    labels=CLASSES, zero_division=0)
        f1w = f1_score(y_te, y_pred, average="weighted", labels=CLASSES, zero_division=0)

        fold_results.append({
            "fold":        fold,
            "accuracy":    acc,
            "f1_macro":    f1m,
            "f1_weighted": f1w,
            "y_test":      y_te,
            "y_pred":      y_pred,
        })
        print(f"    Fold {fold}/{n_folds}  |  acc={acc:.4f}  f1_macro={f1m:.4f}")

    return fold_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(fold_results: list[dict]) -> dict:
    accs = [r["accuracy"]    for r in fold_results]
    f1s  = [r["f1_macro"]    for r in fold_results]
    wf1s = [r["f1_weighted"] for r in fold_results]

    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])

    sep = "=" * 62
    print(f"\n{sep}")
    print("  CNN_1D (raw sensor windows)")
    print(sep)
    print(f"  Accuracy    : {np.mean(accs):.4f}  +/-  {np.std(accs):.4f}")
    print(f"  F1 Macro    : {np.mean(f1s):.4f}  +/-  {np.std(f1s):.4f}")
    print(f"  F1 Weighted : {np.mean(wf1s):.4f}  +/-  {np.std(wf1s):.4f}")
    print("\nPer-class report (predictions pooled across all folds):")
    print(classification_report(all_y_test, all_y_pred, labels=CLASSES, zero_division=0))
    print("Confusion matrix  (rows = true, cols = predicted)")
    print(f"Classes: {CLASSES}")
    print(confusion_matrix(all_y_test, all_y_pred, labels=CLASSES))

    return {
        "accuracy_mean":   np.mean(accs),
        "accuracy_std":    np.std(accs),
        "f1_macro_mean":   np.mean(f1s),
        "f1_macro_std":    np.std(f1s),
        "f1_weighted_mean":np.mean(wf1s),
        "f1_weighted_std": np.std(wf1s),
    }


def _save_fold_results(fold_results: list[dict], out_path: Path) -> None:
    rows = [
        {
            "model":       "CNN_1D",
            "fold":        r["fold"],
            "accuracy":    r["accuracy"],
            "f1_macro":    r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
        }
        for r in fold_results
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Per-fold results saved : {out_path}")


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def _export_tflite(keras_model, out_path: Path) -> None:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"TFLite model saved    : {out_path}  ({len(tflite_bytes)/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1D CNN on raw M3X sensor windows."
    )
    parser.add_argument("--s1",      default=str(RAW_SESSION1_FILE),
                        help="Session 1 raw CSV")
    parser.add_argument("--s2",      default=str(RAW_SESSION2_FILE),
                        help="Session 2 raw CSV")
    parser.add_argument("--labeled", default=str(LABELED_FEATURES_FILE),
                        help="Labeled feature CSV (for Block_ID → Label mapping)")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS,
                        help=f"CV folds (default: {N_FOLDS})")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "cnn_raw_baseline"),
                        help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build windowed dataset
    # ------------------------------------------------------------------
    print("Loading raw sensor data …")
    X, y, groups = load_raw_cnn_dataset(args.s1, args.s2, args.labeled)

    print(f"\nRunning {args.n_folds}-fold StratifiedGroupKFold cross-validation\n")
    print("-" * 40)
    print("Model : CNN_1D")

    # ------------------------------------------------------------------
    # 2. Cross-validation
    # ------------------------------------------------------------------
    fold_results = run_raw_cross_val(X, y, groups, n_folds=args.n_folds)
    _print_summary(fold_results)
    _save_fold_results(fold_results, out / "fold_results.csv")

    # ------------------------------------------------------------------
    # 3. Retrain on full dataset → serialize
    # ------------------------------------------------------------------
    print("\nRetraining on full dataset …")
    mean, std = _fit_channel_scaler(X)
    X_full    = _apply_channel_scaler(X, mean, std)

    final_model = CNNClassifier()
    final_model.fit(X_full, y)

    # Save Keras model via joblib
    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    print(f"Model saved           : {out / 'cnn_1d_model.pkl'}")

    # Save channel scaler (mean, std arrays)
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)
    print(f"Scaler saved          : {out / 'cnn_1d_scaler.npz'}")

    # ------------------------------------------------------------------
    # 4. TFLite export
    # ------------------------------------------------------------------
    _export_tflite(final_model._model, out / "cnn_1d_model.tflite")

    print("\nDone.")


if __name__ == "__main__":
    main()
