"""
train_regressor.py -- Train, tune, cross-validate, and export a CNN
regressor (predicting Stress_0to100) for one Pilot Phase C lineage.

Pipeline
--------
    load windows (cached) -> tune (1 group-aware val split) ->
    5-fold GroupKFold CV (grouped by Survey_ID -- no stratification, since
    the target is continuous) at the winning config -> retrain final model
    on 100% of the data -> export (.pkl, scaler.npz incl. target scaler,
    .tflite float32, .tflite int8) -> manifest.json

The model is trained against a z-scored version of Stress_0to100 (fit on
train fold only, exactly like the input-channel scaler) -- predictions are
inverted back to the original 0-100 scale before any metric is computed or
reported, so every number in fold_results.csv / cv_report.txt is in real
stress-score units.

Usage
-----
    python -m PilotPhaseC.src.train_regressor --lineage all_continuous
    python -m PilotPhaseC.src.train_regressor --lineage labeled_last5min
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PilotPhaseC.config import CNN_PARAMS, N_FOLDS, RANDOM_STATE, RESULTS_DIR
from PilotPhaseC.src.loader import load_lineage_dataset
from PilotPhaseC.src.common import (
    fit_channel_scaler, apply_channel_scaler,
    fit_target_scaler, apply_target_scaler, invert_target_scaler,
    export_tflite, quantize_int8, estimate_tensor_arena,
)
from PilotPhaseC.src.models import CNNRegressor
from PilotPhaseC.src.tuning import tune_regressor


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _metrics(y_true, y_pred) -> dict:
    return dict(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def _report_str(y_true, y_pred) -> str:
    m = _metrics(y_true, y_pred)
    return (
        f"MAE  : {m['mae']:.3f}\n"
        f"RMSE : {m['rmse']:.3f}\n"
        f"R^2  : {m['r2']:.4f}\n"
        f"(n={len(y_true)}, true range [{y_true.min():.1f}, {y_true.max():.1f}], "
        f"pred range [{y_pred.min():.1f}, {y_pred.max():.1f}])"
    )


def run_cv(X, y, groups, params, n_folds, random_state):
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        mean, std = fit_channel_scaler(X_tr)
        X_tr_s = apply_channel_scaler(X_tr, mean, std)
        X_te_s = apply_channel_scaler(X_te, mean, std)

        t_mean, t_std = fit_target_scaler(y_tr)
        y_tr_s = apply_target_scaler(y_tr, t_mean, t_std)

        model = CNNRegressor(params)
        model.fit(X_tr_s, y_tr_s)
        y_pred = invert_target_scaler(model.predict(X_te_s), t_mean, t_std)

        m = _metrics(y_te, y_pred)
        fold_results.append(dict(fold=fold, **m, y_test=y_te, y_pred=y_pred))
        print(f"    Fold {fold}/{n_folds}  |  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.4f}")
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Train a CNN regressor (Stress_0to100) for one Pilot Phase C lineage.")
    parser.add_argument("--participant", default="ring01_pid104")
    parser.add_argument("--lineage", required=True, choices=["all_continuous", "labeled_last5min"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search, use CNN_PARAMS defaults.")
    args = parser.parse_args()

    out = Path(args.output_dir or (RESULTS_DIR / args.participant / args.lineage / "regression"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.lineage}' lineage ...")
    data = load_lineage_dataset(args.participant, args.lineage)
    print(f"  X: {data.X.shape}  groups: {len(np.unique(data.groups))}  windows dropped: {data.n_windows_dropped}")
    print(f"  Stress_0to100 range: [{data.y_stress.min():.1f}, {data.y_stress.max():.1f}]")

    if args.no_tune:
        best_params, tune_log = dict(CNN_PARAMS), []
        print("Skipping hyperparameter search (--no-tune); using CNN_PARAMS defaults.")
    else:
        print("\nTuning (1 group-aware validation split, reduced epoch budget) ...")
        best_params, tune_log = tune_regressor(data.X, data.y_stress, data.groups, CNN_PARAMS)
        print(f"  Best config: {best_params}")

    print(f"\n{args.n_folds}-fold GroupKFold cross-validation (grouped by Survey_ID, no stratification -- continuous target) ...")
    fold_results = run_cv(data.X, data.y_stress, data.groups, best_params, args.n_folds, RANDOM_STATE)

    maes = [r["mae"] for r in fold_results]
    rmses = [r["rmse"] for r in fold_results]
    r2s = [r["r2"] for r in fold_results]
    print(f"\nMAE  : {np.mean(maes):.3f} +/- {np.std(maes):.3f}")
    print(f"RMSE : {np.mean(rmses):.3f} +/- {np.std(rmses):.3f}")
    print(f"R^2  : {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

    pd.DataFrame([{k: r[k] for k in ("fold", "mae", "rmse", "r2")} for r in fold_results]).to_csv(
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
    t_mean, t_std = fit_target_scaler(data.y_stress)
    y_scaled = apply_target_scaler(data.y_stress, t_mean, t_std)

    final_model = CNNRegressor(best_params)
    final_model.fit(X_scaled, y_scaled)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std, target_mean=t_mean, target_std=t_std)
    f32_size = export_tflite(final_model._model, out / "cnn_1d_model.tflite")
    print(f"  Float32 TFLite: {f32_size/1024:.1f} KB")

    print("Quantizing to INT8 ...")
    tflite_i8 = quantize_int8(final_model._model, X_scaled, RANDOM_STATE, n_calib=200)
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    mem = estimate_tensor_arena(tflite_i8)

    # Quantization-fidelity check (own training data, NOT a generalization
    # test): float32 vs INT8 predictions, inverted to real stress units.
    import tensorflow as tf
    rng = np.random.RandomState(RANDOM_STATE)
    check_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    y_pred_f32 = invert_target_scaler(final_model.predict(X_scaled[check_idx]), t_mean, t_std)

    interpreter = tf.lite.Interpreter(model_content=tflite_i8)
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    inp_scale, inp_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    x_i8 = np.clip(np.round(X_scaled[check_idx] / inp_scale + inp_zp), -128, 127).astype(np.int8)

    y_pred_i8_scaled = []
    for i in range(len(x_i8)):
        interpreter.set_tensor(inp_d["index"], x_i8[i:i + 1])
        interpreter.invoke()
        out_i8 = interpreter.get_tensor(out_d["index"])[0]
        y_pred_i8_scaled.append((out_i8.astype(np.float32) - out_zp) * out_scale)
    y_pred_i8 = invert_target_scaler(np.array(y_pred_i8_scaled).reshape(-1), t_mean, t_std)

    mae_f32_vs_i8 = float(np.mean(np.abs(y_pred_f32 - y_pred_i8)))

    quant_report = "\n".join([
        f"Pilot Phase C regressor -- {args.participant}/{args.lineage} -- INT8 quantization report\n",
        f"Target: Stress_0to100, z-scored for training (mean={t_mean:.3f}, std={t_std:.3f}) -- "
        f"model output is in that scaled space; invert with y = scaled*std + mean to get a 0-100 score.",
        f"Trainable params  : {final_model._model.count_params()}",
        f"Float32 file size : {f32_size/1024:.1f} KB",
        f"INT8 file size    : {len(tflite_i8)/1024:.1f} KB",
        "",
        f"Float32 vs INT8 mean abs. difference in real stress units (n={len(check_idx)}, "
        f"own training data, NOT a generalization check): {mae_f32_vs_i8:.3f}",
        "",
        "Quantization parameters (hardcode in firmware):",
        f"  Input  scale={inp_scale:.8f}  zero_point={inp_zp}",
        f"  Output scale={out_scale:.8f}  zero_point={out_zp}  (output is still in z-scored space -- invert with target_mean/target_std above)",
        "",
        "Estimated RAM (tensor arena) -- best-effort, see common.py's estimate_tensor_arena caveat:",
        f"  Estimated peak arena       : {mem['estimated_arena_bytes']/1024:.2f} KB",
        f"  Largest single activation  : {mem['largest_single_activation_bytes']/1024:.2f} KB",
        f"  Total weight bytes (flash) : {mem['total_weight_bytes']/1024:.2f} KB",
    ])
    (out / "int8_quantization_report.txt").write_text(quant_report, encoding="utf-8")
    print("\n" + quant_report)

    manifest = {
        "task": "regression",
        "participant": args.participant,
        "lineage": args.lineage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_windows": int(data.X.shape[0]),
        "n_groups": int(len(np.unique(data.groups))),
        "n_windows_dropped": int(data.n_windows_dropped),
        "target": "Stress_0to100",
        "target_scaler": {"mean": t_mean, "std": t_std},
        "n_folds": args.n_folds,
        "tuned": not args.no_tune,
        "tune_log": tune_log,
        "best_params": best_params,
        "cv_metrics": {
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        },
        "final_model": {
            "float32_tflite_kb": round(f32_size / 1024, 1),
            "int8_tflite_kb": round(len(tflite_i8) / 1024, 1),
            "float32_vs_int8_mae": mae_f32_vs_i8,
            "estimated_arena_kb": round(mem["estimated_arena_bytes"] / 1024, 2),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nAll results written to {out}/")


if __name__ == "__main__":
    main()
