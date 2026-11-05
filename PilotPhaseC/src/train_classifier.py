"""
train_classifier.py -- Train, tune, cross-validate, and export a CNN
classifier for one Pilot Phase C lineage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
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

from PilotPhaseC.config import (
    CLASSES,
    CNN_PARAMS,
    LEGACY_FALLBACK_ACCURACY_THRESHOLD,
    LEGACY_FALLBACK_PARTICIPANTS,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    WINDOW_SAMPLES_LEGACY,
    WINDOW_SAMPLES_NEW,
)
from PilotPhaseC.src.loader import (
    DataLoadSpec,
    LineageDataset,
    default_load_spec,
    load_lineage_dataset,
    load_pooled_lineage_dataset,
)
from PilotPhaseC.src.labels import four_class_support, map_window_labels, training_class_list
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


def _window_metrics(y_true, y_pred, class_list: list[str]) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=class_list, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=class_list, zero_division=0)),
        "n_windows": int(len(y_true)),
    }
    per_class_f1 = f1_score(y_true, y_pred, labels=class_list, average=None, zero_division=0)
    for c, f1v in zip(class_list, per_class_f1):
        key = c.replace("+", "_plus_") if "+" in c else c
        m[f"f1_{key}"] = float(f1v)
        m[f"support_{key}"] = int(np.sum(y_true == c))
    return m


def _report_str(y_true, y_pred, class_list: list[str], title: str = "") -> str:
    lines = [title] if title else []
    m = _window_metrics(y_true, y_pred, class_list)
    lines.extend([
        f"Accuracy    : {m['accuracy']:.4f}",
        f"F1 Macro    : {m['f1_macro']:.4f}",
        f"F1 Weighted : {m['f1_weighted']:.4f}",
        f"Windows     : {m['n_windows']}",
        "",
        classification_report(y_true, y_pred, labels=class_list, zero_division=0),
        f"Confusion matrix (rows=true, cols=predicted)  classes={class_list}",
        str(confusion_matrix(y_true, y_pred, labels=class_list)),
    ])
    return "\n".join(lines)


def _argmax_with_biases(proba: np.ndarray, biases: np.ndarray, class_list: list[str]) -> np.ndarray:
    classes = np.asarray(class_list, dtype=object)
    log_p = np.log(np.clip(proba, 1e-12, 1.0))
    idx = np.argmax(log_p + biases.reshape(1, -1), axis=1)
    return classes[idx]


def run_cv(
    X, y, groups, params, n_folds, random_state, class_list, class_names,
    period=None, lineage=None,
):
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

        sw_tr = classifier_sample_weights(g_tr, y_tr, p_tr, lineage, classes=class_list)
        model = CNNClassifier(params, class_names=class_names)
        model.fit(X_tr, y_tr, sample_weight=sw_tr, groups=g_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        m = _window_metrics(y_te, y_pred, class_list)
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
            f"acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}"
        )
    return fold_results


def _output_subdir(label_mode: str) -> str:
    if label_mode == "four_class":
        return "classification_4class"
    if label_mode == "binary_good_bad":
        return "classification_2class"
    raise ValueError(label_mode)


def train_and_export(
    data: LineageDataset,
    *,
    participant_key: str,
    lineage: str,
    label_mode: str,
    generalized: bool,
    load_spec: DataLoadSpec,
    out: Path,
    n_folds: int,
    data_variant: str = "primary",
) -> dict:
    class_list = training_class_list(label_mode)
    class_names = CLASSES if label_mode == "four_class" else class_list
    y_train = map_window_labels(data.y_label, label_mode)

    out.mkdir(parents=True, exist_ok=True)
    fold_dir = out / "fold_reports"
    fold_dir.mkdir(exist_ok=True)

    print(f"  X: {data.X.shape}  groups: {len(np.unique(data.groups))}  dropped: {data.n_windows_dropped}")
    print(f"  drop_reasons: {data.drop_reasons}")
    print(f"  Training labels: {pd.Series(y_train).value_counts().to_dict()}")
    if label_mode == "four_class":
        print(f"  Underlying 4-class support: {four_class_support(data.y_label)}")

    folds_used = effective_classification_folds(y_train, data.groups, n_folds)
    if folds_used < n_folds:
        print(f"  Using {folds_used}-fold CV (requested {n_folds}).")

    print("\nTuning ...")
    best_params, tune_log = tune_classifier(
        data.X, y_train, data.groups, dict(CNN_PARAMS),
        period=data.period, lineage=lineage,
        classes=class_list, class_names=class_names,
    )

    print(f"\n{folds_used}-fold StratifiedGroupKFold ...")
    fold_results = run_cv(
        data.X, y_train, data.groups, best_params, folds_used, RANDOM_STATE,
        class_list, class_names, period=data.period, lineage=lineage,
    )

    scalar_keys = [k for k in fold_results[0] if k not in ("y_test", "y_pred", "y_proba")]
    pd.DataFrame([{k: r[k] for k in scalar_keys} for r in fold_results]).to_csv(
        out / "fold_results.csv", index=False
    )

    report_parts = [f"Pilot Phase C classifier ({label_mode}, {data_variant})\n"]
    for r in fold_results:
        title = f"=== Fold {r['fold']}/{folds_used} ==="
        block = _report_str(r["y_test"], r["y_pred"], class_list, title=title)
        report_parts.append(block)
        (fold_dir / f"fold_{r['fold']}_cv_report.txt").write_text(block, encoding="utf-8")

    all_y_test = np.concatenate([r["y_test"] for r in fold_results])
    all_y_proba = np.concatenate([r["y_proba"] for r in fold_results], axis=0)
    logit_biases, _ = tune_logit_biases(all_y_test, all_y_proba, classes=class_list, random_state=RANDOM_STATE)
    all_y_pred = apply_logit_biases(all_y_proba, logit_biases, classes=class_list)
    pooled_oof = _window_metrics(all_y_test, all_y_pred, class_list)
    pooled = _report_str(all_y_test, all_y_pred, class_list, title="=== Pooled OOF (calibrated) ===")
    report_parts.append(pooled)
    (out / "cv_report.txt").write_text("\n\n".join(report_parts), encoding="utf-8")
    print("\n" + pooled)

    def _agg(key):
        vals = [r[key] for r in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    acc_m, acc_s = _agg("accuracy")
    f1m_m, f1m_s = _agg("f1_macro")

    mean, std = fit_channel_scaler(data.X)
    X_scaled = apply_channel_scaler(data.X, mean, std)
    sw_all = classifier_sample_weights(data.groups, y_train, data.period, lineage, classes=class_list)
    final_model = CNNClassifier(best_params, class_names=class_names)
    final_model.fit(X_scaled, y_train, sample_weight=sw_all, groups=data.groups)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std, logit_biases=logit_biases)
    f32_size = export_tflite(final_model._model, out / "cnn_1d_model.tflite")

    tflite_i8 = quantize_int8(final_model._model, X_scaled, RANDOM_STATE, n_calib=200)
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    mem = estimate_tensor_arena(tflite_i8)

    import tensorflow as tf
    rng = np.random.RandomState(RANDOM_STATE)
    check_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    y_pred_f32 = predict_with_logit_biases(final_model, X_scaled[check_idx], logit_biases, classes=class_list)

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
        y_pred_i8.append(_argmax_with_biases(probs.reshape(1, -1), logit_biases, class_list)[0])
    agreement = float(np.mean(y_pred_f32 == np.array(y_pred_i8)))

    quant_report = "\n".join([
        f"Pilot Phase C classifier -- {participant_key}/{lineage} -- INT8 report",
        f"Label mode: {label_mode}",
        f"Model classes (output order): {class_list}",
        f"Window shape (T, C): {data.X.shape[1:]}",
        f"Logit biases: {logit_biases.tolist()}",
        f"Float32 TFLite: {f32_size/1024:.1f} KB",
        f"INT8 TFLite: {len(tflite_i8)/1024:.1f} KB",
        f"Float32 vs INT8 agreement (n={len(check_idx)}): {agreement:.4f}",
        f"Estimated tensor arena: {mem['estimated_arena_bytes']/1024:.2f} KB",
    ])
    (out / "int8_quantization_report.txt").write_text(quant_report, encoding="utf-8")

    from PilotPhaseC.config import CLASSIFIER_CLASS_BALANCED_WEIGHTS, CLASSIFIER_SURVEY_EQUAL_WEIGHTS

    manifest = {
        "task": "classification",
        "label_mode": label_mode,
        "model_classes": class_list,
        "four_class_support": four_class_support(data.y_label) if label_mode == "four_class" else None,
        "participant": participant_key,
        "generalized": generalized,
        "lineage": lineage,
        "data_variant": data_variant,
        "load_spec": asdict(load_spec),
        "window_shape": list(data.X.shape[1:]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_windows": int(data.X.shape[0]),
        "n_groups": int(len(np.unique(data.groups))),
        "n_windows_dropped": int(data.n_windows_dropped),
        "drop_reasons": data.drop_reasons,
        "n_folds_requested": n_folds,
        "n_folds": folds_used,
        "training": {
            "label_level": "window",
            "survey_equal_sample_weights": CLASSIFIER_SURVEY_EQUAL_WEIGHTS,
            "class_balanced_sample_weights": CLASSIFIER_CLASS_BALANCED_WEIGHTS,
            "oof_logit_bias_calibration": True,
            "logit_biases": logit_biases.tolist(),
        },
        "tuned": True,
        "tune_log": tune_log,
        "best_params": best_params,
        "cv_metrics": {
            "fold_mean_accuracy": acc_m,
            "fold_std_accuracy": acc_s,
            "fold_mean_f1_macro": f1m_m,
            "fold_std_f1_macro": f1m_s,
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
    print(f"\nResults written to {out}/")
    return manifest


def _load_data(
    participant: str,
    lineage: str,
    generalized: bool,
    load_spec: DataLoadSpec | None,
    force_reload: bool,
) -> tuple[LineageDataset, DataLoadSpec]:
    if generalized:
        data = load_pooled_lineage_dataset(lineage, force_reload=force_reload)
        spec = DataLoadSpec(target_window_samples=WINDOW_SAMPLES_LEGACY, file_set="legacy_pooled")
        return data, spec
    spec = load_spec or default_load_spec(participant, lineage)
    data = load_lineage_dataset(participant, lineage, force_reload=force_reload, load_spec=spec)
    return data, spec


def main():
    parser = argparse.ArgumentParser(description="Train a Pilot Phase C CNN classifier.")
    parser.add_argument("--participant", default="ring01_pid104")
    parser.add_argument("--generalized", action="store_true")
    parser.add_argument("--lineage", required=True, choices=["all_continuous", "labeled_last5min"])
    parser.add_argument(
        "--label-mode",
        default="four_class",
        choices=["four_class", "binary_good_bad"],
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument(
        "--include-legacy-upsampled",
        action="store_true",
        help="Include legacy 112-row files upsampled to 224 (Ring 3/4).",
    )
    parser.add_argument(
        "--no-legacy-fallback",
        action="store_true",
        help="Do not auto-retry with legacy upsampled if accuracy is low.",
    )
    args = parser.parse_args()

    participant_key = "generalized" if args.generalized else args.participant
    sub = _output_subdir(args.label_mode)
    out = Path(args.output_dir or (RESULTS_DIR / participant_key / args.lineage / sub))

    base_spec = None if args.generalized else default_load_spec(args.participant, args.lineage)
    if args.include_legacy_upsampled and base_spec is not None:
        base_spec = DataLoadSpec(
            target_window_samples=WINDOW_SAMPLES_NEW,
            file_set=base_spec.file_set,
            include_legacy_upsampled=True,
            apply_nonwear_filter=base_spec.apply_nonwear_filter,
        )

    scope = "generalized pool" if args.generalized else args.participant
    print(f"Loading {args.lineage} ({scope}) label_mode={args.label_mode} ...")
    data, spec = _load_data(args.participant, args.lineage, args.generalized, base_spec, args.force_reload)

    manifest = train_and_export(
        data,
        participant_key=participant_key,
        lineage=args.lineage,
        label_mode=args.label_mode,
        generalized=args.generalized,
        load_spec=spec,
        out=out,
        n_folds=args.n_folds,
        data_variant="primary",
    )

    acc = manifest["cv_metrics"]["pooled_oof"]["accuracy"]
    if (
        not args.generalized
        and not args.no_legacy_fallback
        and not args.include_legacy_upsampled
        and args.participant in LEGACY_FALLBACK_PARTICIPANTS
        and args.lineage == "all_continuous"
        and acc < LEGACY_FALLBACK_ACCURACY_THRESHOLD
    ):
        print(
            f"\n*** Accuracy {acc:.3f} < {LEGACY_FALLBACK_ACCURACY_THRESHOLD}; "
            f"retrying with legacy upsampled data ***\n"
        )
        fallback_spec = DataLoadSpec(
            target_window_samples=WINDOW_SAMPLES_NEW,
            file_set="new_format",
            include_legacy_upsampled=True,
            apply_nonwear_filter=True,
        )
        data_fb = load_lineage_dataset(
            args.participant, args.lineage, force_reload=args.force_reload, load_spec=fallback_spec,
        )
        out_fb = RESULTS_DIR / participant_key / args.lineage / f"{sub}_legacy_augmented"
        train_and_export(
            data_fb,
            participant_key=participant_key,
            lineage=args.lineage,
            label_mode=args.label_mode,
            generalized=False,
            load_spec=fallback_spec,
            out=out_fb,
            n_folds=args.n_folds,
            data_variant="legacy_augmented",
        )


if __name__ == "__main__":
    main()
