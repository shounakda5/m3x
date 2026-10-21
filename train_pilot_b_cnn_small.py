"""
train_pilot_b_cnn_small.py -- Train a smaller CNN variant on the Pilot B
cohort, for teams whose MCU can't fit the full 3-block model.

Architecture (per CNN_ARCH_TUNING_GUIDE.md's "smallest possible" recipe,
one notch below the "2-block" variant already explored for the original
dataset in results/cnn_test1_2block/ -- fewer filters AND no Dense hidden
layer, so it is smaller on both axes):

    Conv1D(16, k=7) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.20)
    Conv1D(32, k=5) -> BN -> ReLU -> GlobalAvgPool -> Dropout(0.20)
    Dense(4, softmax)                                    [no hidden Dense]

Uses the exact same Pilot B data pipeline as train_pilot_b_cnn.py (raw
loader, Aisha holdout split, per-participant balancing) so results are
directly comparable to the full-size model -- only the architecture and
output directory differ.

Usage
-----
    python train_pilot_b_cnn_small.py
    python train_pilot_b_cnn_small.py --output_dir results/pilot_b_cnn_small

Outputs (in --output_dir)
-------------------------
    fold_results.csv                general-validation per-fold accuracy/F1
    general_validation_report.txt   pooled classification report + confusion matrix
    aisha_holdout_report.txt         Aisha-only held-out classification report
    cnn_1d_model.pkl                 final Keras model (full balanced pool)
    cnn_1d_scaler.npz                per-channel mean/std
    cnn_1d_model.tflite              float32 export
    cnn_1d_model_int8.tflite         INT8 export
    int8_quantization_report.txt     size/RAM/accuracy report
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import CLASSES, N_FOLDS, RESULTS_DIR, PILOT_B_RANDOM_STATE
from src.data.pilot_b_raw import load_pilot_b_dataset
from train_pilot_b_cnn import split_aisha_holdout, uniform_sample_by_participant_total
from train_cnn import _fit_channel_scaler, _apply_channel_scaler
from quantize_pilot_b_cnn import quantize, evaluate_int8, estimate_tensor_arena


# ---------------------------------------------------------------------------
# Small architecture: 2 blocks, 16/32 filters, no Dense hidden layer
# ---------------------------------------------------------------------------

def build_small_model(n_timesteps: int, n_channels: int, n_classes: int, random_state: int):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)

    inp = keras.Input(shape=(n_timesteps, n_channels))

    x = keras.layers.Conv1D(16, kernel_size=7, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.20)(x)

    x = keras.layers.Conv1D(32, kernel_size=5, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.20)(x)

    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit(model_builder, X, y, n_timesteps, n_channels, n_classes, random_state):
    import tensorflow as tf
    from tensorflow import keras

    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    class_weight_dict = dict(enumerate(
        compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    ))

    model = model_builder(n_timesteps, n_channels, n_classes, random_state)
    model.fit(
        X, y,
        epochs=200, batch_size=32, validation_split=0.15,
        class_weight=class_weight_dict, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0),
        ],
    )
    return model


def run_small_cross_val(X, y, groups, n_folds=N_FOLDS):
    le = LabelEncoder().fit(y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        mean, std = _fit_channel_scaler(X_tr)
        X_tr = _apply_channel_scaler(X_tr, mean, std)
        X_te = _apply_channel_scaler(X_te, mean, std)

        y_tr_enc = le.transform(y_tr).astype(np.int32)
        model = _fit(build_small_model, X_tr, y_tr_enc, X_tr.shape[1], X_tr.shape[2], len(le.classes_), PILOT_B_RANDOM_STATE)

        y_pred_enc = np.argmax(model.predict(X_te, verbose=0), axis=1)
        y_pred = le.inverse_transform(y_pred_enc)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro", labels=CLASSES, zero_division=0)
        f1w = f1_score(y_te, y_pred, average="weighted", labels=CLASSES, zero_division=0)
        fold_results.append(dict(fold=fold, accuracy=acc, f1_macro=f1m, f1_weighted=f1w, y_test=y_te, y_pred=y_pred))
        print(f"    Fold {fold}/{n_folds}  |  acc={acc:.4f}  f1_macro={f1m:.4f}")

    return fold_results


def _classification_report_str(y_true, y_pred) -> str:
    lines = []
    lines.append(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    lines.append(f"F1 Macro    : {f1_score(y_true, y_pred, average='macro', labels=CLASSES, zero_division=0):.4f}")
    lines.append(f"F1 Weighted : {f1_score(y_true, y_pred, average='weighted', labels=CLASSES, zero_division=0):.4f}")
    lines.append("")
    lines.append(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    lines.append(f"Confusion matrix (rows=true, cols=predicted)  classes={CLASSES}")
    lines.append(str(confusion_matrix(y_true, y_pred, labels=CLASSES)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train the small 2-block/no-dense CNN on Pilot B data.")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "pilot_b_cnn_small"))
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(PILOT_B_RANDOM_STATE)

    print("Loading Pilot B raw sensor data ...")
    X, y, groups, participants, _ = load_pilot_b_dataset()

    is_holdout, holdout_groups = split_aisha_holdout(y, groups, participants, rng)
    X_aisha_holdout, y_aisha_holdout = X[is_holdout], y[is_holdout]

    pool_mask = ~is_holdout
    X_pool, y_pool = X[pool_mask], y[pool_mask]
    groups_pool, participants_pool = groups[pool_mask], participants[pool_mask]

    balance_mask, sampling_report = uniform_sample_by_participant_total(y_pool, groups_pool, participants_pool, rng)
    X_train_pool = X_pool[balance_mask]
    y_train_pool = y_pool[balance_mask]
    groups_train_pool = groups_pool[balance_mask]
    sampling_report.to_csv(out / "sampling_report.csv", index=False)

    print(f"Training pool: {X_train_pool.shape[0]} windows, Aisha holdout: {X_aisha_holdout.shape[0]} windows")

    print(f"\n{'=' * 62}\nSmall model -- general cross-participant validation ({args.n_folds}-fold)\n{'=' * 62}")
    fold_results = run_small_cross_val(X_train_pool, y_train_pool, groups_train_pool, n_folds=args.n_folds)

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
    (out / "general_validation_report.txt").write_text(_classification_report_str(all_y_test, all_y_pred), encoding="utf-8")

    print("\nRetraining final small model on the full balanced pool ...")
    mean, std = _fit_channel_scaler(X_train_pool)
    X_train_scaled = _apply_channel_scaler(X_train_pool, mean, std)
    le = LabelEncoder().fit(y_train_pool)
    y_train_enc = le.transform(y_train_pool).astype(np.int32)

    final_model = _fit(build_small_model, X_train_scaled, y_train_enc,
                        X_train_scaled.shape[1], X_train_scaled.shape[2], len(le.classes_), PILOT_B_RANDOM_STATE)

    joblib.dump(final_model, out / "cnn_1d_model.pkl")
    np.savez(out / "cnn_1d_scaler.npz", mean=mean, std=std)

    import tensorflow as tf
    tflite_f32 = tf.lite.TFLiteConverter.from_keras_model(final_model).convert()
    (out / "cnn_1d_model.tflite").write_bytes(tflite_f32)
    f32_size = len(tflite_f32)

    print("\nQuantizing to INT8 ...")
    tflite_i8 = quantize(final_model, X_train_scaled, n_calib=200)
    (out / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)
    i8_size = len(tflite_i8)

    print("\nEvaluating on Aisha's held-out slice (float32 + INT8) ...")
    X_aisha_scaled = _apply_channel_scaler(X_aisha_holdout, mean, std)
    y_aisha_pred_f32 = le.inverse_transform(np.argmax(final_model.predict(X_aisha_scaled, verbose=0), axis=1))
    aisha_f32_report = _classification_report_str(y_aisha_holdout, y_aisha_pred_f32)
    (out / "aisha_holdout_report.txt").write_text(aisha_f32_report, encoding="utf-8")
    print(aisha_f32_report)

    y_pred_idx, quant_params = evaluate_int8(tflite_i8, X_aisha_scaled, y_aisha_holdout)
    alphabetical_classes = sorted(CLASSES)
    y_aisha_pred_i8 = np.array([alphabetical_classes[i] for i in y_pred_idx])
    i8_acc = accuracy_score(y_aisha_holdout, y_aisha_pred_i8)
    i8_f1m = f1_score(y_aisha_holdout, y_aisha_pred_i8, average="macro", labels=CLASSES, zero_division=0)

    print("\nEstimating tensor arena (RAM) ...")
    mem = estimate_tensor_arena(tflite_i8)

    n_params = final_model.count_params()

    report_lines = [
        "Pilot B SMALL CNN (2-block, 16/32 filters, no Dense hidden layer) -- quantization report\n",
        f"Trainable params  : {n_params}",
        f"Float32 file size : {f32_size/1024:.1f} KB",
        f"INT8 file size    : {i8_size/1024:.1f} KB  <- what gets flashed to the chip",
        "",
        f"General CV accuracy    : {np.mean(accs):.4f} +/- {np.std(accs):.4f}",
        f"General CV F1 macro    : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        "",
        f"Aisha holdout accuracy (float32) : {accuracy_score(y_aisha_holdout, y_aisha_pred_f32):.4f}",
        f"Aisha holdout accuracy (INT8)    : {i8_acc:.4f}",
        f"Aisha holdout F1 macro (INT8)    : {i8_f1m:.4f}",
        "",
        "Quantization parameters (hardcode in firmware):",
        f"  Input  scale={quant_params['inp_scale']:.8f}  zero_point={quant_params['inp_zp']}",
        f"  Output scale={quant_params['out_scale']:.8f}  zero_point={quant_params['out_zp']}",
        "",
        "Estimated RAM (tensor arena) -- best-effort, same method/caveats as the full-size model's report:",
        f"  Estimated peak arena       : {mem['estimated_arena_bytes']/1024:.2f} KB",
        f"  Largest single activation  : {mem['largest_single_activation_bytes']/1024:.2f} KB",
        f"  Total weight bytes (flash) : {mem['total_weight_bytes']/1024:.2f} KB",
        f"  ({mem['n_tensors']} tensors, {mem['n_ops']} ops)",
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out / "int8_quantization_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\nAll results written to {out}/")


if __name__ == "__main__":
    main()
