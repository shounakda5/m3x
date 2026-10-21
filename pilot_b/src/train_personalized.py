"""
train_personalized.py -- Train one CNN per Pilot B participant, using
only that person's own data (no cross-participant pooling/balancing).

Some participants' devices have an extra humidity sensor (currently Aisha
and Sophia); their personalized models use all 10 channels (HUMIDITY_
CHANNELS). Everyone else uses the standard 9-channel schema (RAW_CHANNELS).
This means those participants' models have a different input shape --
(112, 10) -- from everyone else's (112, 9). Keep that in mind for firmware/
inference code.

Some participants never reported certain stress classes at all (e.g. Lehar
and Aisha both have zero "distress" responses in their data). A personalized
model can only learn classes it has actually seen, so those models have a
3-class output instead of 4, and in a different order (LabelEncoder is
alphabetical over whatever classes are present). Each participant's actual
class list/order is recorded in that participant's own int8_quantization_
report.txt -- always read it before wiring up firmware for a specific
person's model.

There's no held-out generalization thread here (that only makes sense when
training on OTHER people's data, per the generalized-model work). Validation
is a per-participant 5-fold StratifiedGroupKFold, and the final shipped
model is retrained on 100% of that participant's own data.

Usage
-----
    python pilot_b/src/train_personalized.py

Outputs (in pilot_b/results/personalized/<name>/)
---------------------------------------------------
    dataset_summary.csv              per-day parsing/labeling diagnostics
    fold_results.csv                 per-fold CV accuracy/F1 (this person only)
    cv_report.txt                    pooled classification report + confusion matrix
    cnn_1d_model.pkl / .tflite        final model (trained on 100% of their data)
    cnn_1d_model_int8.tflite          INT8 export
    cnn_1d_scaler.npz                 per-channel mean/std
    int8_quantization_report.txt      size + quantization-fidelity report

Plus pilot_b/results/personalized/personalized_overview.csv summarizing all participants.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CLASSES, N_FOLDS, RAW_CHANNELS
from pilot_b.config import PILOT_B_RESULTS_DIR, PILOT_B_RANDOM_STATE
from pilot_b.src.loader import load_pilot_b_participant_dataset, HUMIDITY_CHANNELS, participant_has_humidity
from src.models.cnn_model import CNNClassifier
from train_cnn import _fit_channel_scaler, _apply_channel_scaler
from pilot_b.src.quantize import quantize, estimate_tensor_arena

PARTICIPANTS = [
    (501, "Shuail"),
    (502, "Lehar"),
    (503, "Emma"),
    (504, "Aisha"),
    (505, "Sophia"),
]


def _classification_report_str(y_true, y_pred, labels):
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


def run_person_cv(X, y, groups, labels, n_folds):
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


def train_one_participant(
    participant_id: int, name: str, out_root: Path, n_folds: int,
    detrend: bool = False, label_column: str | None = None,
) -> dict:
    tag = "".join([" [detrended]" if detrend else "", f" [labels={label_column}]" if label_column else ""])
    print(f"\n{'=' * 62}\n{name} ({participant_id}){tag}\n{'=' * 62}")

    has_humidity = participant_has_humidity(participant_id)
    channels = HUMIDITY_CHANNELS if has_humidity else RAW_CHANNELS
    X, y, groups, day_summary = load_pilot_b_participant_dataset(
        participant_id, channels=channels if has_humidity else None, detrend=detrend,
        label_column=label_column,
    )

    labels = sorted(set(y))
    # The "never observed" check only makes sense against the standard 4-class
    # scheme; skip it entirely for an alternate label_column (e.g. mental
    # states), where CLASSES's 4 strings would never match anyway.
    missing = [] if label_column else [c for c in CLASSES if c not in labels]
    print(f"Windows: {X.shape[0]}  channels: {X.shape[2]} {channels}")
    print(f"Classes present: {labels}" + (f"  (never observed: {missing})" if missing else ""))

    out = out_root / name.lower()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(day_summary).to_csv(out / "dataset_summary.csv", index=False)

    # Group-per-class floor -> reduce fold count if a class is too thin for n_folds
    gt = pd.DataFrame({"group": groups, "label": y}).drop_duplicates("group")
    min_groups_per_class = int(gt["label"].value_counts().min())
    folds_used = min(n_folds, min_groups_per_class)

    cv_accuracy = cv_f1_macro = None
    if folds_used < 2:
        print(f"  Not enough groups per class for CV (smallest class has {min_groups_per_class}); skipping CV.")
    else:
        if folds_used < n_folds:
            print(f"  Reducing folds {n_folds} -> {folds_used} (smallest class only has {min_groups_per_class} groups)")
        fold_results = run_person_cv(X, y, groups, labels, folds_used)
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["f1_macro"] for r in fold_results]
        f1ws = [r["f1_weighted"] for r in fold_results]
        cv_accuracy, cv_f1_macro = float(np.mean(accs)), float(np.mean(f1s))
        print(f"  CV Accuracy : {cv_accuracy:.4f} +/- {np.std(accs):.4f}")
        print(f"  CV F1 Macro : {cv_f1_macro:.4f} +/- {np.std(f1s):.4f}")

        pd.DataFrame([{k: r[k] for k in ("fold", "accuracy", "f1_macro", "f1_weighted")} for r in fold_results]).to_csv(
            out / "fold_results.csv", index=False
        )
        all_y_test = np.concatenate([r["y_test"] for r in fold_results])
        all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
        (out / "cv_report.txt").write_text(_classification_report_str(all_y_test, all_y_pred, labels), encoding="utf-8")

    print("  Retraining final model on 100% of this participant's data ...")
    mean, std = _fit_channel_scaler(X)
    X_scaled = _apply_channel_scaler(X, mean, std)

    final_model = CNNClassifier()
    final_model.fit(X_scaled, y)

    joblib.dump(final_model._model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)

    import tensorflow as tf
    tflite_f32 = tf.lite.TFLiteConverter.from_keras_model(final_model._model).convert()
    (out / "cnn_1d_model.tflite").write_bytes(tflite_f32)

    print("  Quantizing to INT8 ...")
    tflite_i8 = quantize(final_model._model, X_scaled, n_calib=min(200, len(X_scaled)))
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)

    # Quantization-fidelity check on a random slice of this participant's own
    # data. This is NOT a generalization test (there's no cross-participant
    # holdout for a personalized model) -- it only confirms INT8 didn't
    # change predictions relative to the float32 model.
    rng = np.random.RandomState(PILOT_B_RANDOM_STATE)
    check_idx = rng.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    y_true_check = np.asarray(y)[check_idx]
    y_pred_f32 = final_model.predict(X_scaled[check_idx])

    interpreter = tf.lite.Interpreter(model_content=tflite_i8)
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    inp_scale, inp_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    x_i8 = np.clip(np.round(X_scaled[check_idx] / inp_scale + inp_zp), -128, 127).astype(np.int8)

    y_pred_i8 = []
    alphabetical_classes = labels  # LabelEncoder used inside CNNClassifier is alphabetical too
    for i in range(len(x_i8)):
        interpreter.set_tensor(inp_d["index"], x_i8[i:i + 1])
        interpreter.invoke()
        probs = (interpreter.get_tensor(out_d["index"])[0].astype(np.float32) - out_zp) * out_scale
        y_pred_i8.append(alphabetical_classes[int(np.argmax(probs))])
    y_pred_i8 = np.array(y_pred_i8)

    agreement = float(np.mean(y_pred_f32 == y_pred_i8))
    f32_acc = accuracy_score(y_true_check, y_pred_f32)
    i8_acc = accuracy_score(y_true_check, y_pred_i8)

    mem = estimate_tensor_arena(tflite_i8)
    n_params = final_model._model.count_params()

    report_lines = [
        f"{name} personalized CNN -- INT8 quantization report\n",
        f"Input shape (timesteps, channels) : (112, {X.shape[2]})",
        f"Channel order : {channels}",
        f"Classes (model output order, alphabetical) : {labels}"
        + (f"  -- {missing} never observed for {name}, not predictable by this model" if missing else ""),
        f"Trainable params  : {n_params}",
        f"Float32 file size : {len(tflite_f32) / 1024:.1f} KB",
        f"INT8 file size    : {len(tflite_i8) / 1024:.1f} KB",
        "",
        f"General CV accuracy / F1 macro : " + (f"{cv_accuracy:.4f} / {cv_f1_macro:.4f}" if cv_accuracy is not None else "(skipped, insufficient data)"),
        "",
        f"Quantization fidelity check (n={len(check_idx)} windows from {name}'s own data -- not a generalization test):",
        f"  Float32 accuracy          : {f32_acc:.4f}",
        f"  INT8 accuracy             : {i8_acc:.4f}",
        f"  Float32 vs INT8 agreement : {agreement:.4f}",
        "",
        "Quantization parameters (hardcode in firmware):",
        f"  Input  scale={inp_scale:.8f}  zero_point={inp_zp}",
        f"  Output scale={out_scale:.8f}  zero_point={out_zp}",
        "",
        "Estimated RAM (tensor arena) -- best-effort, same method/caveats as pilot_b/results/cnn_generalized/int8_quantization_report.txt:",
        f"  Estimated peak arena       : {mem['estimated_arena_bytes'] / 1024:.2f} KB",
        f"  Largest single activation  : {mem['largest_single_activation_bytes'] / 1024:.2f} KB",
        f"  Total weight bytes (flash) : {mem['total_weight_bytes'] / 1024:.2f} KB",
    ]
    report_text = "\n".join(report_lines)
    (out / "int8_quantization_report.txt").write_text(report_text, encoding="utf-8")
    print("\n  " + report_text.replace("\n", "\n  "))

    return dict(
        participant_id=participant_id, name=name, n_windows=X.shape[0], n_channels=X.shape[2],
        classes="|".join(labels), missing_classes="|".join(missing), n_params=n_params,
        float32_kb=round(len(tflite_f32) / 1024, 1), int8_kb=round(len(tflite_i8) / 1024, 1),
        cv_accuracy=cv_accuracy, cv_f1_macro=cv_f1_macro,
        int8_vs_float32_agreement=agreement,
        estimated_arena_kb=round(mem["estimated_arena_bytes"] / 1024, 2),
    )


def main():
    parser = argparse.ArgumentParser(description="Train personalized CNNs, one per Pilot B participant.")
    parser.add_argument("--output_dir", default=str(PILOT_B_RESULTS_DIR / "personalized"))
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--participant_id", type=int, default=None,
                        help="Retrain only this participant (default: all four).")
    parser.add_argument("--detrend", action="store_true",
                        help="Subtract a rolling per-burst baseline from slow-drift channels "
                             "(LED1/2/3, Temp, Humidity) before windowing. Off by default.")
    parser.add_argument("--label_column", default=None,
                        help="One-off/experimental: label from this survey column's raw values "
                             "instead of the standard Opportunity/Pressure scheme (e.g. 'mental state'). "
                             "Off by default -- doesn't change standard labeling.")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = [p for p in PARTICIPANTS if args.participant_id is None or p[0] == args.participant_id]
    summaries = [
        train_one_participant(participant_id, name, out_root, args.n_folds, detrend=args.detrend, label_column=args.label_column)
        for participant_id, name in targets
    ]

    overview_path = out_root / "personalized_overview.csv"
    new_df = pd.DataFrame(summaries)
    if args.participant_id is not None and overview_path.exists():
        existing = pd.read_csv(overview_path)
        existing = existing[existing["participant_id"] != args.participant_id]
        new_df = pd.concat([existing, new_df], ignore_index=True).sort_values("participant_id")
    new_df.to_csv(overview_path, index=False)
    print(f"\nAll personalized models written to {out_root}/")


if __name__ == "__main__":
    main()
