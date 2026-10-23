"""
train_classifier.py -- Train, tune, cross-validate, and export a CNN
classifier for one Pilot Phase C lineage.

Window-level labels; Survey_ID for grouped CV and early-stopping holdout.
After CV, OOF logit biases are fit once for deployment (see config.py).
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
from PilotPhaseC.src.loader import load_lineage_dataset, load_pooled_lineage_dataset
from PilotPhaseC.src.common import (
    fit_channel_scaler,
    apply_channel_scaler,
    export_tflite,
    quantize_int8,
    estimate_tensor_arena,
    classifier_sample_weights,
    effective_classification_folds,
    tune_logit_biases,
    apply_logit_biases,
    predict_with_logit_biases,
)
from PilotPhaseC.src.models import CNNClassifier
from PilotPhaseC.src.tuning import tune_classifier


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _window_metrics(y_true, y_pred) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=CLASSES, zero_division=0)),
        "n_windows": int(len(y_true)),
    }
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    for c, f1v in zip(CLASSES, per_class_f1):
        m[f"f1_{c}"] = float(f1v)
    support = {c: int(np.sum(y_true == c)) for c in CLASSES}
    m.update({f"support_{c}": support[c] for c in CLASSES})
    return m


def _report_str(y_true, y_pred, title: str = "") -> str:
    lines = [title] if title else []
    m = _window_metrics(y_true, y_pred)
    lines.extend([
        f"Accuracy    : {m['accuracy']:.4f}",
        f"F1 Macro    : {m['f1_macro']:.4f}",
        f"F1 Weighted : {m['f1_weighted']:.4f}",
        f"Windows     : {m['n_windows']}",
        "",
        classification_report(y_true, y_pred, labels=CLASSES, zero_division=0),
        f"Confusion matrix (rows=true, cols=predicted)  classes={CLASSES}",
        str(confusion_matrix(y_true, y_pred, labels=CLASSES)),
    ])
    return "\n".join(lines)


def _argmax_with_biases(proba: np.ndarray, biases: np.ndarray) -> str | np.ndarray:
    classes = np.asarray(CLASSES, dtype=object)
    log_p = np.log(np.clip(proba, 1e-12, 1.0))
    idx = np.argmax(log_p + biases.reshape(1, -1), axis=1)
    out = classes[idx]
    return out[0] if out.shape == (1,) else out


def run_cv(X, y, groups, params, n_folds, random_state, period=None, lineage=None):
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr = groups[tr_idx]
        p_tr = period[tr_idx] if period is not None else None

        mean, std = fit_channel_scaler(X_tr)
        X_tr = apply_channel_scaler(X_tr, mean, std)
        X_te = apply_channel_scaler(X_te, mean, std)

        sw_tr = classifier_sample_weights(g_tr, y_tr, p_tr, lineage)
        model = CNNClassifier(params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr, groups=g_tr)
        y_proba = model.predict_proba(X_te)
        y_pred = model.predict(X_te)

        m = _window_metrics(y_te, y_pred)
        m["fold"] = fold
        m["n_train_windows"] = int(len(tr_idx))
        m["n_test_windows"] = int(len(te_idx))
        m["n_train_groups"] = int(len(np.unique(g_tr)))
        m["n_test_groups"] = int(len(np.unique(groups[te_idx])))
        m["y_test"] = y_te
        m["y_pred"] = y_pred
        m["y_proba"] = y_proba
        fold_results.append(m)
        print(
            f"    Fold {fold}/{n_folds}  |  windows test={m['n_test_windows']}  "
            f"acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
            f"f1_relax={m['f1_relax']:.3f}  f1_eu={m['f1_eustress']:.3f}  "
            f"f1_dis={m['f1_distress']:.3f}  f1_ed={m['f1_eustress+distress']:.3f}"
        )
    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Train a CNN classifier for one Pilot Phase C lineage.")
    parser.add_argument("--participant", default="ring01_pid104",
                        help="Participant key from config.PARTICIPANTS (ignored if --generalized).")
    parser.add_argument("--generalized", action="store_true",
                        help="Train on all registered participants pooled for this lineage.")
    parser.add_argument("--lineage", required=True, choices=["all_continuous", "labeled_last5min"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--force-reload", action="store_true", help="Rebuild dataset cache from Excel.")
    args = parser.parse_args()

    participant_key = "generalized" if args.generalized else args.participant
    out = Path(args.output_dir or (RESULTS_DIR / participant_key / args.lineage / "classification"))
    out.mkdir(parents=True, exist_ok=True)
    fold_dir = out / "fold_reports"
    fold_dir.mkdir(exist_ok=True)

    scope = "all registered participants" if args.generalized else args.participant
    print(f"Loading '{args.lineage}' lineage ({scope}) ...")
    if args.generalized:
        data = load_pooled_lineage_dataset(args.lineage, force_reload=args.force_reload)
    else:
        data = load_lineage_dataset(args.participant, args.lineage, force_reload=args.force_reload)
    print(f"  X: {data.X.shape}  groups: {len(np.unique(data.groups))}  windows dropped: {data.n_windows_dropped}")
    print(f"  Window label counts: {pd.Series(data.y_label).value_counts().to_dict()}")

    folds_used = effective_classification_folds(data.y_label, data.groups, args.n_folds)
    if folds_used < args.n_folds:
        print(f"  Using {folds_used}-fold CV (min groups per class limits folds; requested {args.n_folds}).")

    print("\nTuning (group-aware split, window-level macro-F1) ...")
    best_params, tune_log = tune_classifier(
        data.X, data.y_label, data.groups, dict(CNN_PARAMS),
        period=data.period, lineage=args.lineage,
    )
    print(f"  Best config: {best_params}")

    print(f"\n{folds_used}-fold StratifiedGroupKFold (window-level metrics, grouped by Survey_ID) ...")
    fold_results = run_cv(
        data.X, data.y_label, data.groups, best_params, folds_used, RANDOM_STATE,
        period=data.period, lineage=args.lineage,
    )

    scalar_keys = [k for k in fold_results[0] if k not in ("y_test", "y_pred", "y_proba")]
    pd.DataFrame([{k: r[k] for k in scalar_keys} for r in fold_results]).to_csv(
        out / "fold_results.csv", index=False
    )

    report_parts = ["Pilot Phase C classifier CV -- window-level labels and metrics\n"]
    for r in fold_results:
        title = (
            f"=== Fold {r['fold']}/{folds_used}  "
            f"(train windows={r['n_train_windows']}, test windows={r['n_test_windows']}, "
            f"test groups={r['n_test_groups']}) ==="
        )
        block = _report_str(r["y_test"], r["y_pred"], title=title)
        report_parts.append(block)
        (fold_dir / f"fold_{r['fold']}_cv_report.txt").write_text(block, encoding="utf-8")

    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_proba = np.concatenate([r["y_proba"] for r in fold_results], axis=0)
    logit_biases, _ = tune_logit_biases(all_y_test, all_y_proba, random_state=RANDOM_STATE)
    all_y_pred = apply_logit_biases(all_y_proba, logit_biases)
    pooled_oof = _window_metrics(all_y_test, all_y_pred)
    pooled = _report_str(all_y_test, all_y_pred, title="=== Pooled out-of-fold (calibrated) ===")
    report_parts.append(pooled)
    cv_report = "\n\n".join(report_parts)
    print("\n" + pooled)
    (out / "cv_report.txt").write_text(cv_report, encoding="utf-8")

    def _agg(key):
        vals = [r[key] for r in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = _agg("accuracy")
    f1m_m, f1m_s = _agg("f1_macro")
    f1w_m, f1w_s = _agg("f1_weighted")
    print(f"\nFold-mean CNN (pre-calibration): acc={acc_m:.4f}±{acc_s:.4f}  f1_macro={f1m_m:.4f}±{f1m_s:.4f}")

    print("\nRetraining final model on 100% of the data ...")
    mean, std = fit_channel_scaler(data.X)
    X_scaled = apply_channel_scaler(data.X, mean, std)
    sw_all = classifier_sample_weights(data.groups, data.y_label, data.period, args.lineage)
    final_model = CNNClassifier(best_params)
    final_model.fit(X_scaled, data.y_label, sample_weight=sw_all, groups=data.groups)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std, logit_biases=logit_biases)
    f32_size = export_tflite(final_model._model, out / "cnn_1d_model.tflite")
    print(f"  Float32 TFLite: {f32_size/1024:.1f} KB")

    print("Quantizing to INT8 ...")
    tflite_i8 = quantize_int8(final_model._model, X_scaled, RANDOM_STATE, n_calib=200)
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    mem = estimate_tensor_arena(tflite_i8)

    import tensorflow as tf
    rng = np.random.RandomState(RANDOM_STATE)
    check_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    y_pred_f32 = predict_with_logit_biases(final_model, X_scaled[check_idx], logit_biases)

    interpreter = tf.lite.Interpreter(model_content=tflite_i8)
    interpreter.allocate_tensors()
    inp_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    inp_scale, inp_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    x_i8 = np.clip(np.round(X_scaled[check_idx] / inp_scale + inp_zp), -128, 127).astype(np.int8)

    y_pred_i8 = []
    for i in range(len(x_i8)):
        interpreter.set_tensor(inp_d["index"], x_i8[i:i + 1])
        interpreter.invoke()
        probs = (interpreter.get_tensor(out_d["index"])[0].astype(np.float32) - out_zp) * out_scale
        y_pred_i8.append(_argmax_with_biases(probs.reshape(1, -1), logit_biases)[0])
    y_pred_i8 = np.array(y_pred_i8)
    agreement = float(np.mean(y_pred_f32 == y_pred_i8))

    quant_report = "\n".join([
        f"Pilot Phase C classifier -- {participant_key}/{args.lineage} -- INT8 quantization report\n",
        f"Classes (fixed taxonomy order): {CLASSES}",
        f"Logit biases (add to log-prob before argmax): {logit_biases.tolist()}",
        f"Trainable params  : {final_model._model.count_params()}",
        f"Float32 file size : {f32_size/1024:.1f} KB",
        f"INT8 file size    : {len(tflite_i8)/1024:.1f} KB",
        "",
        f"Float32 vs INT8 agreement (n={len(check_idx)}, training sample): {agreement:.4f}",
        "",
        "Quantization parameters (hardcode in firmware):",
        f"  Input  scale={inp_scale:.8f}  zero_point={inp_zp}",
        f"  Output scale={out_scale:.8f}  zero_point={out_zp}",
        "",
        "Estimated RAM (tensor arena):",
        f"  Estimated peak arena       : {mem['estimated_arena_bytes']/1024:.2f} KB",
        f"  Largest single activation  : {mem['largest_single_activation_bytes']/1024:.2f} KB",
        f"  Total weight bytes (flash) : {mem['total_weight_bytes']/1024:.2f} KB",
    ])
    (out / "int8_quantization_report.txt").write_text(quant_report, encoding="utf-8")

    from PilotPhaseC.config import (
        CLASSIFIER_CLASS_BALANCED_WEIGHTS,
        CLASSIFIER_SURVEY_EQUAL_WEIGHTS,
    )
    manifest = {
        "task": "classification",
        "participant": participant_key,
        "generalized": args.generalized,
        "lineage": args.lineage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_windows": int(data.X.shape[0]),
        "n_groups": int(len(np.unique(data.groups))),
        "n_windows_dropped": int(data.n_windows_dropped),
        "classes": CLASSES,
        "n_folds_requested": args.n_folds,
        "n_folds": folds_used,
        "training": {
            "label_level": "window",
            "survey_equal_sample_weights": CLASSIFIER_SURVEY_EQUAL_WEIGHTS,
            "class_balanced_sample_weights": CLASSIFIER_CLASS_BALANCED_WEIGHTS,
            "oof_logit_bias_calibration": True,
            "logit_biases": logit_biases.tolist(),
            "group_holdout_early_stopping": True,
        },
        "tuned": True,
        "tune_log": tune_log,
        "best_params": best_params,
        "cv_metrics": {
            "metric_level": "window",
            "fold_mean_accuracy": acc_m, "fold_std_accuracy": acc_s,
            "fold_mean_f1_macro": f1m_m, "fold_std_f1_macro": f1m_s,
            "fold_mean_f1_weighted": f1w_m, "fold_std_f1_weighted": f1w_s,
            "per_fold": [{k: r[k] for k in scalar_keys} for r in fold_results],
            "pooled_oof": pooled_oof,
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
