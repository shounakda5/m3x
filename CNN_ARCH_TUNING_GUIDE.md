# CNN Architecture Tuning Guide

**Audience:** Farhad (MCU deployment)  
**Purpose:** Run your own architecture experiments to find the best size/accuracy tradeoff for your target hardware.

---

## Where things stand

Three trained CNN variants are already in the repo. All use the same input format and output the same 4 classes.

| Model | Params | Float32 | INT8 | F1 Macro | Notes |
|-------|-------:|--------:|-----:|:--------:|-------|
| Baseline (3-block) | 46,468 | 186 KB | 61 KB | 0.801 | Original model |
| **2-block** | **17,156** | **71 KB** | **27 KB** | **0.821** | Better + smaller |
| Half-filters (16/32/64) | ~12,000 | — | — | 0.739 | Too much loss |

The 2-block model in `results/cnn_test1_2block/` is currently the best. If you need to go even smaller, the experiments below show you how to test any architecture in about 20 minutes.

---

## What the model does (briefly)

Input: a 112-sample window of raw sensor data, 9 channels, pre-normalized.  
Output: 4 softmax probabilities → argmax → stress class.

Internally, Conv1D layers scan across the time axis looking for physiological patterns (PPG pulse shape, HRV features). Each filter sees a fixed-width slice of the time axis; the width is called the kernel size.

At 25 Hz, the timing math is:
```
kernel size 7  =  7 / 25 Hz  =  280 ms  (one full PPG cycle at 70 bpm)
kernel size 5  =  200 ms
kernel size 3  =  120 ms
```

MaxPooling halves the temporal resolution after each block. GlobalAveragePooling collapses the entire remaining sequence into a single vector before the Dense output layer.

---

## The experiment pattern

All experiments follow the same four steps:

1. Define a `build_fn(T, C, n_classes)` — returns a compiled Keras model
2. Run 5-fold CV using `run_cv()`
3. Read the F1 macro result
4. If the result is good, export TFLite and INT8

Copy the script below into a new file (e.g. `my_cnn_test.py`) and edit the architecture inside `build_my_model()`. Everything else is boilerplate that you do not need to change.

---

## Ready-to-run experiment template

```python
"""
my_cnn_test.py -- Architecture experiment script.
Run: python my_cnn_test.py
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------------ #
#  EDIT THIS FUNCTION -- everything else is boilerplate              #
# ------------------------------------------------------------------ #
def build_my_model(T, C, n_classes):
    """
    T         = number of timesteps (112 for 4.5s windows)
    C         = number of channels (always 9)
    n_classes = number of output classes (always 4)

    Return a compiled keras.Model.

    Tips
    ----
    - Start from the 2-block template below and adjust filter counts,
      kernel sizes, or dropout.
    - Kernel size k means the filter sees k/25 seconds of data.
    - Smaller filter counts (16, 32) = fewer parameters = smaller model.
    - Adding MaxPooling1D(2) halves the sequence before the next block.
    - GlobalAveragePooling1D() must appear exactly once, before the Dense head.
    - Dropout between 0.1 and 0.4 helps prevent overfitting.
    """
    inp = keras.Input(shape=(T, C))

    # ----- Conv block 1 -----
    x = keras.layers.Conv1D(filters=32, kernel_size=7, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    # ----- Conv block 2 -----
    x = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)   # collapses time axis
    x = keras.layers.Dropout(0.25)(x)

    # ----- Dense head -----
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.30)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------ #
#  Boilerplate below -- no edits needed                              #
# ------------------------------------------------------------------ #

EXPERIMENT_NAME = "my_test"   # output saved to results/cnn_<EXPERIMENT_NAME>/
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

from src.data.raw_preprocess import load_raw_cnn_dataset

def _fit_scaler(X_tr):
    m = X_tr.mean(axis=(0, 1), keepdims=True)
    s = X_tr.std(axis=(0, 1), keepdims=True)
    return m, np.where(s == 0, 1.0, s)

def run_cv(build_fn, X, y, groups, n_folds=5):
    le   = LabelEncoder().fit(y)
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    accs, f1s = [], []
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        m, s = _fit_scaler(Xtr)
        Xtr  = ((Xtr - m) / s).astype(np.float32)
        Xte  = ((Xte - m) / s).astype(np.float32)
        ytr_enc = le.transform(ytr).astype(np.int32)
        n_cls   = len(le.classes_)
        cw = dict(enumerate(compute_class_weight(
            "balanced", classes=np.arange(n_cls), y=ytr_enc
        )))
        model = build_fn(Xtr.shape[1], Xtr.shape[2], n_cls)
        idx   = np.random.permutation(len(Xtr))
        model.fit(
            Xtr[idx], ytr_enc[idx],
            epochs=200, batch_size=32,
            validation_split=0.15,
            class_weight=cw,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=20,
                    restore_best_weights=True, verbose=0,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=8, min_lr=1e-6, verbose=0,
                ),
            ],
        )
        ypred = le.inverse_transform(
            np.argmax(model.predict(Xte, verbose=0), axis=1)
        )
        acc = accuracy_score(yte, ypred)
        f1  = f1_score(yte, ypred, average="macro",
                       labels=CLASSES, zero_division=0)
        accs.append(acc); f1s.append(f1)
        print(f"  Fold {fold}/5  acc={acc:.4f}  f1_macro={f1:.4f}", flush=True)
    return np.array(accs), np.array(f1s)


def save_model_and_export(build_fn, X, y, out_dir: Path):
    """Retrain on full dataset and export float32 + INT8 TFLite."""
    import joblib

    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = np.where(X.std(axis=(0, 1), keepdims=True) == 0, 1.0,
                    X.std(axis=(0, 1), keepdims=True))
    X_scaled = ((X - mean) / std).astype(np.float32)

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y).astype(np.int32)
    n_cls = len(le.classes_)
    cw = dict(enumerate(compute_class_weight(
        "balanced", classes=np.arange(n_cls), y=y_enc
    )))

    model = build_fn(X_scaled.shape[1], X_scaled.shape[2], n_cls)
    idx   = np.random.permutation(len(X_scaled))
    model.fit(
        X_scaled[idx], y_enc[idx],
        epochs=200, batch_size=32,
        validation_split=0.15, class_weight=cw, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20,
                restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=8, min_lr=1e-6, verbose=0),
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "cnn_1d_model.pkl")
    np.savez(out_dir / "cnn_1d_scaler.npz", mean=mean, std=std)

    # Float32 TFLite
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = conv.convert()
    (out_dir / "cnn_1d_model.tflite").write_bytes(tflite_f32)
    print(f"  float32 TFLite : {len(tflite_f32)/1024:.1f} KB")

    # INT8 TFLite
    calib = X_scaled[np.random.default_rng(42).choice(
        len(X_scaled), 200, replace=False
    )]
    def rep_ds():
        for i in range(len(calib)): yield [calib[i:i+1]]
    conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv2.optimizations = [tf.lite.Optimize.DEFAULT]
    conv2.representative_dataset = rep_ds
    conv2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv2.inference_input_type  = tf.int8
    conv2.inference_output_type = tf.int8
    tflite_i8 = conv2.convert()
    (out_dir / "cnn_1d_model_int8.tflite").write_bytes(tflite_i8)

    # Print quantization params needed for MCU
    interp = tf.lite.Interpreter(
        model_content=tflite_i8
    )
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  INT8 TFLite    : {len(tflite_i8)/1024:.1f} KB")
    print(f"  Input  scale={inp_d['quantization'][0]:.8f}  zero_point={inp_d['quantization'][1]}")
    print(f"  Output scale={out_d['quantization'][0]:.8f}  zero_point={out_d['quantization'][1]}")
    print(f"  Saved to {out_dir}")

    return model


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading data ...")
    X, y, groups = load_raw_cnn_dataset()
    print(f"Dataset: {X.shape[0]} windows  ({X.shape[1]} timesteps x {X.shape[2]} channels)\n")

    print("--- 5-fold CV ---")
    accs, f1s = run_cv(build_my_model, X, y, groups)
    print(f"\nResult: acc={accs.mean():.4f}+/-{accs.std():.4f}  "
          f"f1_macro={f1s.mean():.4f}+/-{f1s.std():.4f}")
    print(f"vs baseline:  delta acc={accs.mean()-0.806:+.4f}  "
          f"delta f1={f1s.mean()-0.801:+.4f}")

    # Uncomment the block below once you are happy with the CV result
    # and want to save the trained model + TFLite exports.
    #
    # print("\n--- Saving model + TFLite exports ---")
    # out = Path(f"results/cnn_{EXPERIMENT_NAME}")
    # save_model_and_export(build_my_model, X, y, out)
```

---

## Architecture knobs explained

### Filters per block

Controls how many patterns each block can detect. More filters = more expressive = larger model.

```python
Conv1D(filters=32, ...)   # 32 pattern detectors
Conv1D(filters=64, ...)   # 64 pattern detectors
```

Halving the filters approximately quarters the parameters in that block (parameters scale with filters²). Good range to explore: 8, 16, 32, 64.

### Kernel size

How many time steps one filter sees at once. At 25 Hz:

| Kernel | Window | What it captures |
|-------:|-------:|-----------------|
| 3 | 120 ms | Fine-grained texture |
| 5 | 200 ms | Sub-cycle features |
| 7 | 280 ms | One full PPG cycle at 70 bpm |
| 11 | 440 ms | Multi-cycle patterns |

Larger kernels see more context but add more parameters. Keep block 1's kernel the largest (it sees raw signal), and reduce in later blocks (which see downsampled, abstracted signal).

### Number of blocks

Each block adds representational depth. The key constraint: after each MaxPooling1D(2), the sequence length halves. Starting from 112 timesteps:

```
Block 1 → MaxPool → 56 timesteps
Block 2 → MaxPool → 28 timesteps
Block 3 → GlobalAvgPool → 1 vector (no sequence left)
```

If you add a 3rd MaxPool you'd get 14 timesteps before GlobalAvgPool, which is fine. Removing a MaxPool leaves more temporal resolution for the next block.

### Dense head size

The Dense layer after GlobalAvgPool combines all learned features into the final prediction. The baseline uses 64 units. You can skip it entirely (go straight to Dense(4, softmax)) — worth testing if the model is overfitting.

### Dropout

Applied after pooling and the Dense layer. Higher dropout = more regularization. Useful range: 0.1–0.4. If the model is overfitting (val loss rises while train loss drops), increase dropout.

---

## Tuning recipes

**I want the smallest possible model that still works:**
Start from 2 blocks with filters=16/32, no Dense hidden layer, dropout=0.20. If F1 drops below 0.75, add back the Dense(32) layer. Then try filters=8/16 as the next step down.

**I want to go deeper without adding many parameters:**
Keep filters small (16/32) but add a third block. Use smaller kernels (3/3/3) since the sequence is already downsampled by the time block 3 sees it.

**Validation loss spikes mid-training (unstable):**
Reduce the learning rate default from 1e-3 to 5e-4, or increase dropout. The EarlyStopping and ReduceLROnPlateau callbacks handle most of this automatically.

**All folds have high variance (e.g. ±0.06 or more):**
This usually means the model is too large for the dataset. Reduce filters or remove a block.

---

## Reading the results

The key number is **F1 Macro** (not accuracy). It averages F1 across all four classes equally, so a model can't cheat by just getting good at the majority class (relax, 36%).

```
Baseline F1 Macro  : 0.801
Target floor (−5%) : 0.751
Current best       : 0.821  (2-block model)
```

If your experiment's F1 is above 0.751 and the model is smaller than what you need, it passes. If it's above 0.801, it's a new best.

---

## Exporting once you're happy

Uncomment the `save_model_and_export(...)` block at the bottom of the template and re-run. It will:

1. Retrain on the full dataset
2. Save `cnn_1d_model.pkl` and `cnn_1d_scaler.npz`
3. Export `cnn_1d_model.tflite` (float32)
4. Export `cnn_1d_model_int8.tflite` (INT8, smallest for MCU)
5. Print the `scale` and `zero_point` values you need for on-device quantization

The INT8 quantization parameters from the two existing models for reference:

| Model | Input scale | Input ZP | Output scale | Output ZP |
|-------|------------:|---------:|-------------:|----------:|
| Baseline (3-block) | 0.07942456 | 13 | 0.00390625 | −128 |
| 2-block | varies* | varies* | 0.00390625 | −128 |

*Run the export script and read the printed values — they depend on the calibration data and the specific trained weights.

---

## Quick reference: existing model sizes

```
results/
  cnn_raw_baseline/           <- 3-block baseline
    cnn_1d_model.tflite       186 KB (float32)
    cnn_1d_model_int8.tflite   61 KB (INT8)    F1=0.801

  cnn_test1_2block/           <- 2-block, current best
    cnn_1d_model.tflite        71 KB (float32)
    cnn_1d_model_int8.tflite   27 KB (INT8)    F1=0.821
```

Both use the same input/output format: input `(1, 112, 9)` int8 or float32, output `(1, 4)` probabilities.
