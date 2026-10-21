"""
train_mental_states.py -- ONE-OFF EXPERIMENT.

Trains a pooled (multi-participant) CNN on Shuail/Lehar/Emma/Aisha (501-504),
excluding Sophia and Jiang, using a DIFFERENT labeling scheme than the
standard pipeline: labels come directly from the survey's "Mental states"
column (one class per distinct value found), NOT from the Opportunity/
Pressure 4-class scheme. This does not change how labeling works for any
other model in this project -- see pilot_b/src/loader.py's
`label_column` parameter, which defaults to the standard scheme everywhere
else.

Known going in (checked before training): Shuail has zero non-blank
"Mental states" survey responses across all of his data, so he contributes
nothing despite being nominally in scope -- this run is effectively
Lehar + Emma + Aisha. There are only 2 distinct non-blank values in the
data, so this is a binary classification task.

Usage
-----
    python pilot_b/src/experiments/train_mental_states.py

Outputs (in pilot_b/results/cnn_mental_states/)
-------------------------------------------------
    dataset_summary.csv             per participant-day parsing/labeling diagnostics
    sampling_report.csv             per-participant/per-class balancing detail
    fold_results.csv                per-fold CV accuracy/F1
    general_validation_report.txt   pooled classification report + confusion matrix
    cnn_1d_model.pkl / .tflite       final model (trained on the full balanced pool)
    cnn_1d_model_int8.tflite         INT8 export
    cnn_1d_scaler.npz                per-channel mean/std
    int8_quantization_report.txt     size/RAM/quantization report
    SUMMARY.md                       consolidated write-up of the experiment
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import N_FOLDS
from pilot_b.config import PILOT_B_RESULTS_DIR, PILOT_B_RANDOM_STATE
from pilot_b.src.loader import load_pilot_b_dataset
from src.models.cnn_model import CNNClassifier
from train_cnn import _fit_channel_scaler, _apply_channel_scaler, _export_tflite
from pilot_b.src.train_generalized import uniform_sample_by_participant_total, PARTICIPANT_NAMES
from pilot_b.src.quantize import quantize, estimate_tensor_arena

PARTICIPANT_IDS = [501, 502, 503, 504]   # Shuail, Lehar, Emma, Aisha -- Sophia/Jiang excluded
LABEL_COLUMN = "mental state"


def _classification_report_str(y_true, y_pred, labels) -> str:
    lines = [
        f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}",
        f"F1 Macro    : {f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0):.4f}",
        f"F1 Weighted : {f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0):.4f}",
        "",
        classification_report(y_true, y_pred, labels=labels, zero_division=0),
        f"Confusion matrix (rows=true, cols=predicted)  classes={labels}",
        str(confusion_matrix(y_true, y_pred, labels=labels)),
    ]
    return "\n".join(lines)


def run_cv(X, y, groups, labels, n_folds):
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
        f1m = f1_score(y_te, y_pred, average="macro", labels=labels, zero_division=0)
        f1w = f1_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0)
        fold_results.append(dict(fold=fold, accuracy=acc, f1_macro=f1m, f1_weighted=f1w, y_test=y_te, y_pred=y_pred))
        print(f"    Fold {fold}/{n_folds}  |  acc={acc:.4f}  f1_macro={f1m:.4f}")
    return fold_results


def main():
    out = Path(PILOT_B_RESULTS_DIR) / "cnn_mental_states"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(PILOT_B_RANDOM_STATE)

    print(f"Loading Shuail/Lehar/Emma/Aisha data, labeled from '{LABEL_COLUMN}' (not Opportunity/Pressure) ...")
    X, y, groups, participants, day_summary = load_pilot_b_dataset(
        participant_ids=PARTICIPANT_IDS, label_column=LABEL_COLUMN,
    )
    pd.DataFrame(day_summary).to_csv(out / "dataset_summary.csv", index=False)

    labels = sorted(set(y))
    contributing = sorted(set(participants.tolist()))
    excluded = [pid for pid in PARTICIPANT_IDS if pid not in contributing]
    print(f"\nTotal windows: {X.shape[0]}")
    print(f"Classes found ({len(labels)}): {labels}")
    print(f"Class counts: {pd.Series(y).value_counts().to_dict()}")
    print(f"Participants contributing data: {[PARTICIPANT_NAMES.get(p, p) for p in contributing]}")
    if excluded:
        print(f"Participants with ZERO usable '{LABEL_COLUMN}' responses (excluded in practice): "
              f"{[PARTICIPANT_NAMES.get(p, p) for p in excluded]}")

    # ------------------------------------------------------------------
    # Balance per participant (equal total budget, same convention as the
    # standard generalized model) so no one person dominates the pool.
    # ------------------------------------------------------------------
    balance_mask, sampling_report = uniform_sample_by_participant_total(y, groups, participants, rng)
    sampling_report.to_csv(out / "sampling_report.csv", index=False)
    X_bal, y_bal, groups_bal = X[balance_mask], y[balance_mask], groups[balance_mask]
    print(f"\nAfter per-participant balancing: {X_bal.shape[0]} windows "
          f"(down from {X.shape[0]})  class counts: {pd.Series(y_bal).value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 5-fold grouped CV (standard reporting)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}\n{N_FOLDS}-fold StratifiedGroupKFold cross-validation\n{'=' * 62}")
    fold_results = run_cv(X_bal, y_bal, groups_bal, labels, N_FOLDS)
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1_macro"] for r in fold_results]
    f1ws = [r["f1_weighted"] for r in fold_results]
    print(f"\nAccuracy    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"F1 Macro    : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"F1 Weighted : {np.mean(f1ws):.4f} +/- {np.std(f1ws):.4f}")

    pd.DataFrame([{k: r[k] for k in ("fold", "accuracy", "f1_macro", "f1_weighted")} for r in fold_results]).to_csv(
        out / "fold_results.csv", index=False
    )
    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    report_str = _classification_report_str(all_y_test, all_y_pred, labels)
    print("\n" + report_str)
    (out / "general_validation_report.txt").write_text(report_str, encoding="utf-8")

    # ------------------------------------------------------------------
    # Final model on 100% of the balanced pool
    # ------------------------------------------------------------------
    print("\nRetraining final model on the full balanced pool ...")
    mean, std = _fit_channel_scaler(X_bal)
    X_scaled = _apply_channel_scaler(X_bal, mean, std)
    final_model = CNNClassifier()
    final_model.fit(X_scaled, y_bal)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)
    _export_tflite(final_model._model, out / "cnn_1d_model.tflite")

    print("Quantizing to INT8 ...")
    tflite_i8 = quantize(final_model._model, X_scaled, n_calib=min(200, len(X_scaled)))
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    mem = estimate_tensor_arena(tflite_i8)
    n_params = final_model._model.count_params()
    f32_size = (out / "cnn_1d_model.tflite").stat().st_size

    quant_report = "\n".join([
        "Pilot B Mental-States labeling experiment -- INT8 quantization report\n",
        f"Trainable params  : {n_params}",
        f"Float32 file size : {f32_size / 1024:.1f} KB",
        f"INT8 file size    : {len(tflite_i8) / 1024:.1f} KB",
        f"Estimated RAM (tensor arena): {mem['estimated_arena_bytes'] / 1024:.2f} KB",
        f"Largest single activation  : {mem['largest_single_activation_bytes'] / 1024:.2f} KB",
    ])
    (out / "int8_quantization_report.txt").write_text(quant_report, encoding="utf-8")
    print("\n" + quant_report)

    # ------------------------------------------------------------------
    # Consolidated summary
    # ------------------------------------------------------------------
    summary_lines = [
        "# Pilot B -- Mental-States Labeling Experiment (one-off)\n",
        "**This is an experiment and does not change how labeling works for any other model in this project.**\n",
        f"Participants in scope: Shuail (501), Lehar (502), Emma (503), Aisha (504) -- Sophia and Jiang excluded.\n",
        f"Labels come from the survey's `Mental states` column directly (one class per distinct value), "
        f"**not** from the Opportunity/Pressure scheme used everywhere else -- Opportunity and Pressure were not used at all.\n",
        "## Known limitation, checked before training\n",
        f"Shuail has zero non-blank `Mental states` responses across all of his survey data, so he contributes "
        f"**no windows** despite being nominally in scope. This run is effectively Lehar + Emma + Aisha. "
        f"There are only 2 distinct non-blank values in the data, so this is a **binary** classification task, not multi-class.\n",
        "## Dataset\n",
        f"- Full labeled dataset: **{X.shape[0]}** windows",
        f"- Classes: {labels}",
        f"- Class counts (full dataset): {pd.Series(y).value_counts().to_dict()}",
        f"- Participants contributing data: {[PARTICIPANT_NAMES.get(p, p) for p in contributing]}",
        f"- After per-participant balancing (training pool): **{X_bal.shape[0]}** windows, "
        f"class counts {pd.Series(y_bal).value_counts().to_dict()}\n",
        "## Results (5-fold grouped cross-validation)\n",
        f"- Accuracy    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}",
        f"- F1 Macro    : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        f"- F1 Weighted : {np.mean(f1ws):.4f} +/- {np.std(f1ws):.4f}\n",
        "See `general_validation_report.txt` for the full per-class report and confusion matrix.\n",
        "## Model files\n",
        f"- `cnn_1d_model.tflite` -- float32 ({f32_size/1024:.1f} KB)",
        f"- `cnn_1d_model_int8.tflite` -- INT8 ({len(tflite_i8)/1024:.1f} KB), estimated RAM ~{mem['estimated_arena_bytes']/1024:.1f} KB",
        f"- `cnn_1d_scaler.npz` -- per-channel mean/std used by the final model",
    ]
    (out / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nAll results written to {out}/")


if __name__ == "__main__":
    main()
