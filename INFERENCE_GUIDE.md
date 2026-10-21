# Inference Guide

This document covers everything needed to run the trained models on new data. It is primarily written for Farhad (MCU deployment) but applies equally to anyone integrating the models in a Python environment.

Two dataset variants exist for each model: unfiltered (all data) and motion-filtered (MF, motion artifacts removed). The unfiltered CNN is the recommended deployment model.

---

## Model overview

| Model | Dataset | Best for | Input | Output file |
|-------|---------|----------|-------|-------------|
| **CNN (1D, raw)** | Unfiltered | **Deployment -- best F1** | 112x9 raw windows | `cnn_raw_baseline/cnn_1d_model.tflite` |
| CNN (1D, raw MF) | Motion-filtered | Comparative study | 112x9 raw windows | `cnn_raw_mf_baseline/cnn_1d_model.tflite` |
| **Pilot B CNN (generalized)** | 5-participant cohort (Shuail/Lehar/Emma/Aisha/Sophia) | **Deployment for the new cohort** | 112x9 raw windows | `pilot_b_cnn/cnn_1d_model_int8.tflite` |
| XGBoost | Unfiltered | Feature-based fallback | 38 features | `xgb_baseline/xgboost_model.onnx` |
| XGBoost MF | Motion-filtered | Feature-based MF | 38 features | `xgb_mf_baseline/xgboost_model.onnx` |
| Random Forest | Unfiltered | Feature-based fallback | 38 features | `baseline/randomforest_model.onnx` |
| SVM | Unfiltered | Feature-based | 38 features | `baseline/svm_model.onnx` |

All models output one of four class labels: `relax`, `eustress`, `distress`, `eustress+distress`.

---

## Class index mapping

All models use `sklearn.LabelEncoder` (alphabetical ordering) internally. When interpreting raw probability or class index outputs, use this mapping:

| Index | Label |
|------:|-------|
| 0 | `distress` |
| 1 | `eustress` |
| 2 | `eustress+distress` |
| 3 | `relax` |

---

## 1. CNN (1D) on raw sensor data

This is the recommended model for deployment. It operates directly on raw M3X ring sensor output with no MATLAB feature extraction required.

### Input specification

- **Window size:** 112 samples
- **Sampling rate:** ~25 Hz (nominal)
- **Window duration:** ~4.5 seconds
- **Channels:** 9 (order is critical — see below)

### Channel order

```
Index  Name         Source
  0    LED1         Raw PPG channel 1 (ADC counts)
  1    LED2         Raw PPG channel 2 (ADC counts)
  2    LED3         Raw PPG channel 3 (ADC counts)
  3    Temp         Skin temperature
  4    AccX         Accelerometer X axis
  5    AccY         Accelerometer Y axis
  6    AccZ         Accelerometer Z axis
  7    LED3_LED1    Derived: LED3 - LED1
  8    LED3_LED2    Derived: LED3 - LED2
```

Channels 7 and 8 are computed on the fly from channels 0-2. No other preprocessing is applied to the raw values before normalization.

### Preprocessing (required before TFLite inference)

Load the saved scaler stats and apply per-channel z-normalization:

```python
import numpy as np

# Load scaler (saved as (1, 1, 9) arrays)
scaler = np.load("results/cnn_raw_baseline/cnn_1d_scaler.npz")
mean = scaler["mean"]   # shape (1, 1, 9)
std  = scaler["std"]    # shape (1, 1, 9)

# window shape: (112, 9)
window_normalized = (window - mean) / std  # broadcasts correctly
```

### TFLite inference (Python)

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="results/cnn_raw_baseline/cnn_1d_model.tflite")
interpreter.allocate_tensors()

inp_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]

# window_normalized: (112, 9) float32 — after z-normalization above
inp = window_normalized.reshape(1, 112, 9).astype(np.float32)
interpreter.set_tensor(inp_detail["index"], inp)
interpreter.invoke()
probs = interpreter.get_tensor(out_detail["index"])[0]  # (4,) softmax probs

CLASS_NAMES = ["distress", "eustress", "eustress+distress", "relax"]
predicted_label = CLASS_NAMES[np.argmax(probs)]
```

### TFLite model details

Two files are available in `results/cnn_raw_baseline/`:

| Property | Float32 | INT8 |
|----------|---------|------|
| File | `cnn_1d_model.tflite` | `cnn_1d_model_int8.tflite` |
| Input shape | `(1, 112, 9)` | `(1, 112, 9)` |
| Input dtype | `float32` | `int8` |
| Output shape | `(1, 4)` | `(1, 4)` |
| Output dtype | `float32` | `int8` |
| File size | 186 KB | **61 KB** |
| CV F1 Macro | 0.801 | 0.801 (no measurable loss) |

The INT8 model is ~3x smaller with no accuracy loss. It is the recommended model for MCU deployment.

---

## 2a. INT8 TFLite inference (Python)

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="results/cnn_raw_baseline/cnn_1d_model_int8.tflite"
)
interpreter.allocate_tensors()

inp_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]

# Quantization parameters (hardcoded from the model -- do not re-derive)
INP_SCALE      = 0.07942456007003784
INP_ZERO_POINT = 13
OUT_SCALE      = 0.00390625
OUT_ZERO_POINT = -128

# window_normalized: (112, 9) float32 -- after z-normalization
inp_f32 = window_normalized.reshape(1, 112, 9).astype(np.float32)
inp_i8  = np.clip(np.round(inp_f32 / INP_SCALE + INP_ZERO_POINT), -128, 127).astype(np.int8)

interpreter.set_tensor(inp_detail["index"], inp_i8)
interpreter.invoke()
out_i8 = interpreter.get_tensor(out_detail["index"])[0]  # (4,) int8

probs = (out_i8.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE  # (4,) float32
CLASS_NAMES = ["distress", "eustress", "eustress+distress", "relax"]
predicted_label = CLASS_NAMES[np.argmax(probs)]
```

---

### MCU deployment notes (for Farhad)

Use `cnn_1d_model_int8.tflite` — it is 61 KB vs 186 KB for the float32 version, with no measurable accuracy loss (0.801 F1 on both). This is the recommended file for TFLite Micro.

**INT8 inference on-device (pseudocode):**
```
INP_SCALE      = 0.07942456
INP_ZERO_POINT = 13
OUT_SCALE      = 0.00390625
OUT_ZERO_POINT = -128

// After z-normalization (float):
for each channel c in 0..8:
    x_norm[c] = (x_raw[c] - mean[c]) / std[c]

// Quantize input to int8:
for each value v in window (112 x 9 values):
    x_int8[v] = clamp(round(x_norm[v] / INP_SCALE) + INP_ZERO_POINT, -128, 127)

// Run TFLite Micro inference on x_int8 -> output int8[4]

// Dequantize output to float probabilities:
for i in 0..3:
    prob[i] = (output_int8[i] - OUT_ZERO_POINT) * OUT_SCALE

predicted_class = argmax(prob)
// 0=distress  1=eustress  2=eustress+distress  3=relax
```

The scaler stats (`mean`, `std`) are small arrays that can be hardcoded in firmware:
- Each is shape `(1, 1, 9)` — 9 float values per array
- Apply element-wise: `x_norm[channel] = (x_raw[channel] - mean[channel]) / std[channel]`

The scaler values can be extracted with:
```python
import numpy as np
s = np.load("results/cnn_raw_baseline/cnn_1d_scaler.npz")
print("mean:", s["mean"].flatten().tolist())
print("std:", s["std"].flatten().tolist())
```

Windowing on-device: collect 112 consecutive samples from one duty-cycle ON burst, compute the 2 derived channels, normalize, and run inference. Two windows fit in a single ~10 s burst — run inference twice per burst if latency allows.

---

## 1b. Pilot B generalized CNN (5-participant cohort)

Same architecture, same 112x9 raw-window input format, and the same channel order/preprocessing as the CNN above -- trained on `data/pliot_b/` (Shuail/501, Lehar/502, Emma/503, Aisha/504, Sophia/505) instead of the original two sessions. Use this model for the Pilot B cohort; use `cnn_raw_baseline` for the original two-session data. Aisha and Sophia's devices have a humidity sensor, but this model uses the standard 9-channel schema (no humidity) for a uniform input across all five participants -- humidity is only used in their personalized models (section 1c below).

Files are in `results/pilot_b_cnn/`:

| Property | Float32 | INT8 (recommended for MCU) |
|----------|---------|------|
| File | `cnn_1d_model.tflite` | `cnn_1d_model_int8.tflite` |
| Input shape | `(1, 112, 9)` | `(1, 112, 9)` |
| Input dtype | `float32` | `int8` |
| Output shape | `(1, 4)` | `(1, 4)` |
| Output dtype | `float32` | `int8` |
| File size (flash) | 185.9 KB | **61.0 KB** |
| Aisha-holdout accuracy | 0.8765 | 0.8824 (no meaningful loss) |

**Estimated RAM (tensor arena):** ~7.0 KB peak, largest single activation tensor 3.5 KB -- see `results/pilot_b_cnn/int8_quantization_report.txt` for the method and its caveats (it's a best-effort simulation of a greedy memory planner over the model's own execution graph, not a measurement on real hardware). Confirm with `interpreter.arena_used_bytes()` on the actual target before final sign-off.

**INT8 inference (Python)** -- identical pattern to the section above, with these values:

```python
import numpy as np
import tensorflow as tf

scaler = np.load("results/pilot_b_cnn/cnn_1d_scaler.npz")
mean, std = scaler["mean"], scaler["std"]

interpreter = tf.lite.Interpreter(model_path="results/pilot_b_cnn/cnn_1d_model_int8.tflite")
interpreter.allocate_tensors()
inp_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]

INP_SCALE      = 0.04210300
INP_ZERO_POINT = 10
OUT_SCALE      = 0.00390625
OUT_ZERO_POINT = -128

# window: (112, 9) float32 raw sensor values, channel order per "Channel order" above
window_normalized = (window - mean) / std
inp_i8 = np.clip(np.round(window_normalized.reshape(1, 112, 9) / INP_SCALE + INP_ZERO_POINT), -128, 127).astype(np.int8)

interpreter.set_tensor(inp_detail["index"], inp_i8)
interpreter.invoke()
out_i8 = interpreter.get_tensor(out_detail["index"])[0]

probs = (out_i8.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE
CLASS_NAMES = ["distress", "eustress", "eustress+distress", "relax"]
predicted_label = CLASS_NAMES[np.argmax(probs)]
```

To regenerate the INT8 export (e.g. after retraining on more Pilot B data): `python quantize_pilot_b_cnn.py`. Full training/validation results are in `results/pilot_b_cnn/PILOT_B_SUMMARY.md`.

### Smaller variant, if flash/RAM is tighter

`results/pilot_b_cnn_small/` has a second Pilot B model for MCUs that can't fit the one above: 2 conv blocks (16/32 filters) with no Dense hidden layer, 3,940 params vs 46,468.

| | Full (`pilot_b_cnn/`) | Small (`pilot_b_cnn_small/`) |
|---|---:|---:|
| INT8 flash size | 61.0 KB | **11.0 KB** |
| Estimated RAM (tensor arena) | ~7.0 KB | **~3.5 KB** |
| General CV accuracy / F1 macro | 0.8106 / 0.7853 | 0.7996 / 0.7730 (essentially tied) |
| Aisha-holdout accuracy / F1 macro (INT8) | 0.8824 / 0.5141 | 0.8647 / 0.3287 |

Same input format, same channel order, same preprocessing, same class mapping -- only the file paths and quantization params differ:

```
Input  scale=0.04210300  zero_point=10
Output scale=0.00390625  zero_point=-128
```

(coincidentally identical to the full model's, since both were quantized from the same calibration pool and z-normalized inputs land in a similar range -- verify against your own `int8_quantization_report.txt` if you retrain either model). This round, the full model pulls meaningfully ahead on Aisha-holdout F1 macro (0.51 vs 0.33) -- the small model's limited capacity leans harder on the `eustress` majority class and recovers less of her `eustress+distress` responses. Pick the full model if Aisha-like generalization matters more than footprint; pick the small model if flash/RAM is the binding constraint -- it still costs little on general cross-participant accuracy. Full comparison in `results/pilot_b_cnn/PILOT_B_SUMMARY.md`.

To retrain/re-export the small model: `python train_pilot_b_cnn_small.py`.

### Personalized models (one per participant)

`results/pilot_b_personalized/<name>/` has a separate model trained only on each person's own data, for cases where a per-person model outperforms the generalized one for that person (it does, substantially -- see `results/pilot_b_personalized/PERSONALIZED_SUMMARY.md`).

**Read this before wiring up firmware for a specific person -- these models are NOT interchangeable:**

- Aisha's and Sophia's models take a **10-channel** input (`LED1, LED2, LED3, Temp, Humidity, AccX, AccY, AccZ, LED3_LED1, LED3_LED2`) because their devices have a humidity sensor the others' don't. Shuail, Lehar, and Emma's models take the standard 9 channels. Don't feed a 10-channel person's windows into a 9-channel person's model or vice versa.
- Lehar, Aisha, and Sophia never reported "distress" in their data, so their models only output **3 classes** (`eustress, eustress+distress, relax`) instead of 4, in that alphabetical order. Shuail and Emma's models output the full 4 classes (`distress, eustress, eustress+distress, relax`). Each participant's exact class order and quantization params are in their own `int8_quantization_report.txt` -- always read the specific person's report rather than assuming one mapping fits all five.

Everything else (windowing, preprocessing, z-normalization, INT8 inference pattern) is identical to the sections above -- just point at `results/pilot_b_personalized/<name>/cnn_1d_model_int8.tflite` and that person's own scaler and class list.

To retrain: `python train_personalized_cnn.py`.

---

## 2. XGBoost on engineered features

Use this if you have access to Farhad's MATLAB feature extraction pipeline and need a feature-based model.

### Input specification

38 features in this exact order (from `config.py FEATURE_COLS`):

```
PA1, PA_STD_LED1, PA_RMSSD_LED1, PA_CV_LED1, mean_LED1,
HR_LED1, SDNN_LED1, RMSSD_LED1,
PA2, PA_STD_LED2, PA_RMSSD_LED2, PA_CV_LED2, mean_LED2,
PA3, PA_STD_LED3, PA_RMSSD_LED3, PA_CV_LED3, LED3_mean,
LED1_LED2_Raw_Mean_Diff, LED1_LED2_Raw_Mean_AbsDiff,
LED3_LED1_Raw_Mean_Diff, LED3_LED1_Raw_Mean_AbsDiff,
LED3_LED2_Raw_Mean_Diff, LED3_LED2_Raw_Mean_AbsDiff,
PA1_PA2_mean_diff, PA1_PA2_AbsDiff,
PA1_PA3_mean_diff, PA1_PA3_AbsDiff,
PA2_PA3_mean_diff, PA2_PA3_AbsDiff,
PA1_PA3_Ratio, PA2_PA3_Ratio, PA1_PA2_Ratio,
mean_temperature,
AccX_STD, AccY_STD, AccZ_STD, Acc_Total_STD
```

No scaling required.

### Python inference

```python
import joblib
import numpy as np

model = joblib.load("results/xgb_baseline/xgboost_model.pkl")

# features: numpy array or DataFrame of shape (N, 38) in the column order above
predictions = model.predict(features)           # string labels
probabilities = model.predict_proba(features)   # (N, 4) — class order: alphabetical
```

Note: `model.predict()` returns string labels directly via the internal `LabelEncoder`. `predict_proba()` column order follows the alphabetical class index mapping above.

---

## 3. Random Forest on engineered features

Same 38-feature input as XGBoost. No scaling.

```python
import joblib

model = joblib.load("results/baseline/randomforest_model.pkl")
predictions    = model.predict(features)
probabilities  = model.predict_proba(features)
```

---

## 4. SVM on engineered features

Same 38-feature input. **Requires scaling** with the saved StandardScaler before inference.

```python
import joblib

scaler = joblib.load("results/baseline/svm_scaler.pkl")
model  = joblib.load("results/baseline/svm_model.pkl")

features_scaled = scaler.transform(features)
predictions     = model.predict(features_scaled)
probabilities   = model.predict_proba(features_scaled)
```

---

## Motion-filtered CNN inference

Use the same TFLite inference code as above, but load the MF scaler and MF model:

```python
scaler = np.load("results/cnn_raw_mf_baseline/cnn_1d_scaler.npz")
interpreter = tf.lite.Interpreter(model_path="results/cnn_raw_mf_baseline/cnn_1d_model.tflite")
```

Everything else (channel order, z-norm formula, output mapping) is identical.

---

## ONNX inference for feature-based models

All feature-based models (XGBoost, RF, SVM) are exported to ONNX alongside their `.pkl` files. ONNX inference is useful for deployment outside Python or for cross-platform serving.

```python
import onnxruntime as rt
import numpy as np

CLASS_NAMES = ["distress", "eustress", "eustress+distress", "relax"]  # alphabetical

# Example for XGBoost (same pattern for RF and SVM)
sess = rt.InferenceSession("results/xgb_baseline/xgboost_model.onnx")
input_name  = sess.get_inputs()[0].name
label_name  = sess.get_outputs()[0].name
prob_name   = sess.get_outputs()[1].name

# features: numpy array shape (N, 38) in FEATURE_COLS order, dtype float32
features = features.astype(np.float32)
labels, probs = sess.run([label_name, prob_name], {input_name: features})
# labels: (N,) int64 class indices; probs: dict with key 'probabilities', shape (N, 4)
```

For the SVM ONNX model, apply the StandardScaler first:
```python
import joblib
scaler = joblib.load("results/baseline/svm_scaler.pkl")
features_scaled = scaler.transform(features).astype(np.float32)
```

---

## Handling NaN / Inf in feature inputs

Before passing feature vectors to any feature-based model, clean them the same way the training pipeline did:

```python
import numpy as np
import pandas as pd

X = pd.DataFrame(features, columns=FEATURE_COLS)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
```

This applies to SVM, RF, and XGBoost. The raw CNN does not use engineered features, so this step does not apply.
