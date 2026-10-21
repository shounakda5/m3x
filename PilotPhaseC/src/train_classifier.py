"""
train_classifier.py -- Train, tune, cross-validate, and export a CNN
classifier for one Pilot Phase C lineage.

Pipeline
--------
    load windows (cached) -> tune (1 group-aware val split) ->
    5-fold StratifiedGroupKFold CV (grouped by Survey_ID) at the winning
    config -> retrain final model on 100% of the data -> export
    (.pkl, scaler.npz, .tflite float32, .tflite int8) -> manifest.json

Usage
-----
    python -m PilotPhaseC.src.train_classifier --lineage all_continuous
    python -m PilotPhaseC.src.train_classifier --lineage labeled_last5min
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import CLASSES, CNN_PARAMS, N_FOLDS, RANDOM_STATE, RESULTS_DIR
from PilotPhaseC.src.loader import load_lineage_dataset
from PilotPhaseC.src.common import fit_channel_scaler, apply_channel_scaler, export_tflite, quantize_int8, estimate_tensor_arena
from PilotPhaseC.src.models import CNNClassifier
from PilotPhaseC.src.tuning import tune_classifier


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _report_str(y_true, y_pred) -> str:
    lines = [
        f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}",
        f"F1 Macro    : {f1_score(y_true, y_pred, average='macro', labels=CLASSES, zero_division=0):.4f}",
        f"F1 Weighted : {f1_score(y_true, y_pred, average='weighted', labels=CLASSES, zero_division=0):.4f}",
        "",
        classification_report(y_true, y_pred, labels=CLASSES, zero_division=0),
        f"Confusion matrix (rows=true, cols=predicted)  classes={CLASSES}",
        str(confusion_matrix(y_true, y_pred, labels=CLASSES)),
    ]
    return "\n".join(lines)


def run_cv(X, y, groups, params, n_folds, random_state):
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        mean, std = fit_channel_scaler(X_tr)
        X_tr = apply_channel_scaler(X_tr, mean, std)
        X_te = apply_channel_scaler(X_te, mean, std)

        model = CNNClassifier(params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro", labels=CLASSES, zero_division=0)
        f1w = f1_score(y_te, y_pred, average="weighted", labels=CLASSES, zero_division=0)
        fold_results.append(dict(fold=fold, accuracy=acc, f1_macro=f1m, f1_weighted=f1w, y_test=y_te, y_pred=y_pred))
        print(f"    Fold {fold}/{n_folds}  |  acc={acc:.4f}  f1_macro={f1m:.4f}")
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Train a CNN classifier for one Pilot Phase C lineage.")
    parser.add_argument("--participant", default="ring01_pid104")
    parser.add_argument("--lineage", required=True, choices=["all_continuous", "labeled_last5min"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search, use CNN_PARAMS defaults.")
    args = parser.parse_args()

    out = Path(args.output_dir or (RESULTS_DIR / args.participant / args.lineage / "classification"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.lineage}' lineage ...")
    data = load_lineage_dataset(args.participant, args.lineage)
    print(f"  X: {data.X.shape}  groups: {len(np.unique(data.groups))}  windows dropped: {data.n_windows_dropped}")
    print(f"  Label counts: {pd.Series(data.y_label).value_counts().to_dict()}")

    if args.no_tune:
        best_params, tune_log = dict(CNN_PARAMS), []
        print("Skipping hyperparameter search (--no-tune); using CNN_PARAMS defaults.")
    else:
        print("\nTuning (1 group-aware validation split, reduced epoch budget) ...")
        best_params, tune_log = tune_classifier(data.X, data.y_label, data.groups, CNN_PARAMS)
        print(f"  Best config: {best_params}")

    print(f"\n{args.n_folds}-fold StratifiedGroupKFold cross-validation (grouped by Survey_ID) ...")
    fold_results = run_cv(data.X, data.y_label, data.groups, best_params, args.n_folds, RANDOM_STATE)

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
    cv_report = _report_str(all_y_test, all_y_pred)
    print("\n" + cv_report)
    (out / "cv_report.txt").write_text(cv_report, encoding="utf-8")

    print("\nRetraining final model on 100% of the data ...")
    mean, std = fit_channel_scaler(data.X)
    X_scaled = apply_channel_scaler(data.X, mean, std)
    final_model = CNNClassifier(best_params)
    final_model.fit(X_scaled, data.y_label)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)
    f32_size = export_tflite(final_model._model, out / "cnn_1d_model.tflite")
    print(f"  Float32 TFLite: {f32_size/1024:.1f} KB")

    print("Quantizing to INT8 ...")
    tflite_i8 = quantize_int8(final_model._model, X_scaled, RANDOM_STATE, n_calib=200)
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    mem = estimate_tensor_arena(tflite_i8)

    # Quantization-fidelity check: predictions on a random slice of the
    # training data, float32 model vs INT8 model (NOT a generalization
    # test -- just confirms INT8 didn't change predictions).
    import tensorflow as tf
    rng = np.random.RandomState(RANDOM_STATE)
    check_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    y_pred_f32 = final_model.predict(X_scaled[check_idx])

    interpreter = tf.lite.Interpreter(model_content=tflite_i8)
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    inp_scale, inp_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    x_i8 = np.clip(np.round(X_scaled[check_idx] / inp_scale + inp_zp), -128, 127).astype(np.int8)

    alphabetical_classes = sorted(set(data.y_label))
    y_pred_i8 = []
    for i in range(len(x_i8)):
        interpreter.set_tensor(inp_d["index"], x_i8[i:i + 1])
        interpreter.invoke()
        probs = (interpreter.get_tensor(out_d["index"])[0].astype(np.float32) - out_zp) * out_scale
        y_pred_i8.append(alphabetical_classes[int(np.argmax(probs))])
    y_pred_i8 = np.array(y_pred_i8)
    agreement = float(np.mean(y_pred_f32 == y_pred_i8))

    quant_report = "\n".join([
        f"Pilot Phase C classifier -- {args.participant}/{args.lineage} -- INT8 quantization report\n",
        f"Classes (alphabetical, model output order): {alphabetical_classes}",
        f"Trainable params  : {final_model._model.count_params()}",
        f"Float32 file size : {f32_size/1024:.1f} KB",
        f"INT8 file size    : {len(tflite_i8)/1024:.1f} KB",
        "",
        f"Float32 vs INT8 agreement (n={len(check_idx)}, own training data, NOT a generalization check): {agreement:.4f}",
        "",
        "Quantization parameters (hardcode in firmware):",
        f"  Input  scale={inp_scale:.8f}  zero_point={inp_zp}",
        f"  Output scale={out_scale:.8f}  zero_point={out_zp}",
        "",
        "Estimated RAM (tensor arena) -- best-effort, see common.py's estimate_tensor_arena caveat:",
        f"  Estimated peak arena       : {mem['estimated_arena_bytes']/1024:.2f} KB",
        f"  Largest single activation  : {mem['largest_single_activation_bytes']/1024:.2f} KB",
        f"  Total weight bytes (flash) : {mem['total_weight_bytes']/1024:.2f} KB",
    ])
    (out / "int8_quantization_report.txt").write_text(quant_report, encoding="utf-8")
    print("\n" + quant_report)

    manifest = {
        "task": "classification",
        "participant": args.participant,
        "lineage": args.lineage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_windows": int(data.X.shape[0]),
        "n_groups": int(len(np.unique(data.groups))),
        "n_windows_dropped": int(data.n_windows_dropped),
        "classes": CLASSES,
        "n_folds": args.n_folds,
        "tuned": not args.no_tune,
        "tune_log": tune_log,
        "best_params": best_params,
        "cv_metrics": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
            "f1_weighted_mean": float(np.mean(f1ws)), "f1_weighted_std": float(np.std(f1ws)),
        },
        "final_model": {
            "float32_tflite_kb": round(f32_size / 1024, 1),
            "int8_tflite_kb": round(len(tflite_i8) / 1024, 1),
            "float32_vs_int8_agreement": agreement,
            "estimated_arena_kb": round(mem["estimated_arena_bytes"] / 1024, 2),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nAll results written to {out}/")


if __name__ == "__main__":
    main()
