# Inference Guide

This document covers everything needed to run the trained models on new data. It is primarily written for Farhad (MCU deployment) but applies equally to anyone integrating the models in a Python environment.

Two dataset variants exist for each model: unfiltered (all data) and motion-filtered (MF, motion artifacts removed). The unfiltered CNN is the recommended deployment model.

---

## Model overview

| Model | Dataset | Best for | Input | Output file |
|-------|---------|----------|-------|-------------|
| **CNN (1D, raw)** | Unfiltered | **Deployment -- best F1** | 112x9 raw windows | `cnn_raw_baseline/cnn_1d_model.tflite` |
| CNN (1D, raw MF) | Motion-filtered | Comparative study | 112x9 raw windows | `cnn_raw_mf_baseline/cnn_1d_model.tflite` |
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

| Property | Value |
|----------|-------|
| Input shape | `(1, 112, 9)` |
| Input dtype | `float32` |
| Output shape | `(1, 4)` |
| Output dtype | `float32` (softmax probabilities) |
| File size | ~186 KB |
| Quantization | None (float32) |

### MCU deployment notes (for Farhad)

The model is float32 — if the target MCU has limited RAM, consider post-training INT8 quantization. INT8 reduces model size by ~4x and is supported by TFLite Micro; it requires a representative calibration dataset (~100 windows) and typically costs 1-2 F1 points.

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
