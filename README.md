# M3X Stress Classification Pipeline

A machine-learning pipeline that classifies momentary stress states from M3X ring sensor physiological data using four model types evaluated with 5-fold Stratified Group Cross-Validation:

- **SVM, Random Forest, XGBoost** — trained on MATLAB-extracted engineered features
- **1D CNN** — trained directly on raw sensor time-series windows (LED1/2/3, Temp, Acc)

Part of a workplace lighting EMA (Ecological Momentary Assessment) study at USC.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Data description](#3-data-description)
4. [Label taxonomy](#4-label-taxonomy)
5. [Installation](#5-installation)
6. [End-to-end workflow](#6-end-to-end-workflow)
7. [Baseline results](#7-baseline-results)
8. [Design decisions](#8-design-decisions)
9. [Configuration](#9-configuration)
10. [Output artefacts](#10-output-artefacts)

---

## 1. Project overview

Participants wear an M3X ring sensor that records PPG (three LED channels), accelerometer, and temperature data in a 10 s ON / ~21 s OFF duty cycle at ~25 Hz. Periodically they receive a short EMA survey rating their current activity on **Opportunity** and **Pressure** (1-6 Likert scale). The two ratings map to one of four stress-state labels.

**Two parallel training pipelines:**

| Pipeline | Input | Models |
|----------|-------|--------|
| Feature-based (`train.py`) | 38 MATLAB-extracted features per ~5 s sub-window | SVM, Random Forest, XGBoost |
| Raw time-series (`train_cnn.py`) | 112 samples x 9 channels (~4.5 s window at 25 Hz) | 1D CNN |

**Evaluation:** 5-fold Stratified Group CV — all sub-windows from the same duty-cycle burst share the same fold, preventing correlated-sample leakage.

---

## 2. Repository structure

```
Project/
|
+-- config.py                    # All tuneable constants (paths, hyperparams, class names)
+-- train.py                     # Feature-based models: SVM, RF, XGBoost
+-- train_cnn.py                 # Raw time-series CNN
+-- tune.py                      # Hyperparameter search (RandomizedSearchCV)
+-- requirements.txt
|
+-- src/
|   +-- data/
|   |   +-- classify_survey.py   # Step 1: Qualtrics CSV -> labelled survey CSV
|   |   +-- merge_labels.py      # Step 2: attach survey labels to feature rows by timestamp
|   |   +-- preprocess.py        # Feature pipeline: load, clean, extract X/y/groups
|   |   \-- raw_preprocess.py    # Raw pipeline: burst detection, windowing, label matching
|   |
|   +-- models/
|   |   +-- base.py              # Abstract BaseClassifier interface
|   |   +-- svm_model.py         # SVC (kernel=rbf, class_weight=balanced)
|   |   +-- rf_model.py          # RandomForestClassifier + feature_importances()
|   |   +-- xgb_model.py         # XGBClassifier (sample_weight=balanced)
|   |   \-- cnn_model.py         # 1D CNN (3 Conv blocks -> GlobalAvgPool -> Dense)
|   |
|   \-- evaluation/
|       +-- cross_val.py         # StratifiedGroupKFold runner (feature models)
|       \-- metrics.py           # Reporting: summary, comparison table, CSV export
|
+-- data/
|   +-- raw/
|   |   +-- session1_raw.csv     # Raw M3X sensor data -- session 1 (June 1)
|   |   \-- session2_raw.csv     # Raw M3X sensor data -- session 2 (June 2)
|   |
|   +-- features/
|   |   +-- 1_suhail_M3X_Features_AllBlocks_output.xlsx  # MATLAB features -- session 1
|   |   +-- 2_suhail_M3X_Features_AllBlocks_output.xlsx  # MATLAB features -- session 2
|   |   \-- features_labeled_new.csv                     # Final labeled dataset (pipeline output)
|   |
|   \-- survey/
|       +-- M3X_EMA_session1_Jun1_labelled.csv
|       \-- M3X_EMA_session2_Jun2_labelled.csv
|
+-- results/
|   +-- baseline/          # SVM + RF baseline (pre-tuning)
|   +-- xgb_baseline/      # XGBoost baseline
|   \-- cnn_raw_baseline/  # 1D CNN on raw data (current best model)
|       +-- fold_results.csv
|       +-- cnn_1d_model.pkl  +  cnn_1d_scaler.npz
|       \-- cnn_1d_model.tflite   <- deployment artefact for Farhad
|
\-- tests/
    +-- test_classify_survey.py  # 26 tests
    \-- test_merge_labels.py     # 26 tests
```

---

## 3. Data description

### Raw sensor data -- `data/raw/`

Two CSV files, one per recording session. Each file contains ~75 000-87 000 rows.

| Column | Description |
|--------|-------------|
| `Time_PPG` | Elapsed time in **milliseconds** since device boot |
| `LED1` | PPG amplitude, LED channel 1 (raw ADC counts) |
| `LED2` | PPG amplitude, LED channel 2 |
| `LED3` | PPG amplitude, LED channel 3 |
| `Temp` | Skin temperature (milli-degrees or raw ADC, session-dependent) |
| `AccX/Y/Z` | 3-axis accelerometer |

Duty cycle: ~10 s ON / ~21 s OFF. Each ON burst contains ~250 samples at ~25 Hz.
The CNN uses 9 channels: the 7 above plus `LED3-LED1` and `LED3-LED2` computed on the fly.

### Labeled feature dataset -- `data/features/features_labeled_new.csv`

1,288 rows x 49 columns (38 engineered features + index + label). Each row is one ~5 s sub-window. Two sub-windows per duty-cycle block (`SubWindow_ID` = 1 or 2).

**Class distribution:**

| Label | Count | % |
|-------|------:|--:|
| relax | 468 | 36.3% |
| eustress+distress | 358 | 27.8% |
| distress | 286 | 22.2% |
| eustress | 176 | 13.7% |

---

## 4. Label taxonomy

Labels are derived from EMA survey Opportunity and Pressure scores (1-6). Threshold: <= 3 = Low, >= 4 = High.

| Label | Opportunity | Pressure | Interpretation |
|-------|:-----------:|:--------:|----------------|
| `relax` | Low | Low | Low demand, low activation |
| `eustress` | High | Low | Engaging without pressure |
| `distress` | Low | High | Pressured without engagement |
| `eustress+distress` | High | High | High challenge and high pressure |

---

## 5. Installation

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=2.0
tensorflow>=2.12
```

---

## 6. End-to-end workflow

The labeled datasets are already built and committed. Steps 1-3 are documented for reference if re-running from raw data.

### Feature-based pipeline (SVM / RF / XGBoost)

**Step 1 -- Classify EMA surveys**
```bash
python -m src.data.classify_survey --input_dir data/survey/ --output_dir data/survey/
```

**Step 2 -- Merge labels onto feature rows (per session)**
```bash
python -m src.data.merge_labels \
    --features data/features/1_suhail_M3X_Features_AllBlocks_output.xlsx \
    --labels   data/survey/M3X_EMA_session1_Jun1_labelled.csv \
    --output   data/features/features_labeled_session1.csv

python -m src.data.merge_labels \
    --features data/features/2_suhail_M3X_Features_AllBlocks_output.xlsx \
    --labels   data/survey/M3X_EMA_session2_Jun2_labelled.csv \
    --output   data/features/features_labeled_session2.csv
```

**Step 3 -- Combine sessions**
```python
import pandas as pd
s1 = pd.read_csv("data/features/features_labeled_session1.csv")
s2 = pd.read_csv("data/features/features_labeled_session2.csv")
s2["Block_ID"] += s1["Block_ID"].max() + 1
combined = pd.concat([s1, s2], ignore_index=True)
combined[combined["Label"] != "Unknown"].to_csv("data/features/features_labeled_new.csv", index=False)
```

**Step 4 -- Train**
```bash
python train.py --data data/features/features_labeled_new.csv
python train.py --data data/features/features_labeled_new.csv --models svm rf --output_dir results/my_run
```

**Tune hyperparameters (optional)**
```bash
python tune.py --data data/features/features_labeled_new.csv --models svm --n-iter 40
```

### Raw time-series pipeline (1D CNN)

The raw pipeline reads `data/raw/session1_raw.csv` and `data/raw/session2_raw.csv` directly.
Label assignment is done automatically by matching duty-cycle burst timestamps to the labeled feature CSV.

```bash
python train_cnn.py
python train_cnn.py --output_dir results/cnn_raw_v2
```

**What the raw pipeline does internally:**
1. Loads each session's raw CSV, drops trailer rows
2. Detects duty-cycle ON bursts (gap > 1 s between samples)
3. Discards bursts shorter than 8 s (startup/partial bursts)
4. Splits each burst into 2 non-overlapping windows of 112 samples (~ 4.5 s each)
5. Matches each burst to the nearest Block_Start_s in features_labeled_new.csv (tolerance 2 s) to get its label
6. Runs 5-fold StratifiedGroupKFold CV, grouped by Block_ID

### Run tests
```bash
pytest tests/ -v
```

---

## 7. Baseline results

All models evaluated with 5-fold Stratified Group CV (group = Block_ID).

| Model | Input | Accuracy | F1 Macro | Notes |
|-------|-------|:--------:|:--------:|-------|
| **CNN (1D, raw)** | Raw 112x9 windows | **80.6%** +/- 2.4% | **0.801** +/- 0.023 | **Best overall** |
| XGBoost | 38 features | 78.6% +/- 1.8% | 0.766 +/- 0.018 | |
| Random Forest | 38 features | 74.5% +/- 2.8% | 0.700 +/- 0.030 | Pre-tuning |
| SVM | 38 features | 59.1% +/- 2.7% | 0.510 +/- 0.020 | Pre-tuning |

**Per-class F1 for CNN_1D (best model):**

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| eustress+distress | 0.90 | 0.86 | 0.88 |
| distress | 0.71 | 0.90 | 0.80 |
| relax | 0.85 | 0.71 | 0.78 |
| eustress | 0.72 | 0.79 | 0.75 |

The CNN operating on raw time-series outperforms all feature-based models by 3.5+ F1 points.
The previous feature-based CNN (treating engineered feature vectors as 1D sequences) scored only 44.9% F1 macro -- applying Conv1D to a sequence of unordered features is architecturally unsound.

---

## 8. Design decisions

### StratifiedGroupKFold

Each duty-cycle burst produces two sub-windows. Sub-windows from the same burst share physiological signal and are not independent. All sub-windows from the same burst (Block_ID) are always in the same fold, preventing train/test leakage.

### Scaler discipline

- **Feature models (SVM):** StandardScaler fitted on training fold only, applied to test fold.
- **CNN (raw):** per-channel z-normalisation (mean/std over all training windows and timesteps), fitted on training fold only. The final scaler is saved as `cnn_1d_scaler.npz`.

### Class-balanced weights

SVM, RF, and CNN all apply class weighting proportional to inverse class frequency. XGBoost uses `compute_sample_weight("balanced")`. This prevents the majority class (`relax`, 36%) from dominating.

### CNN architecture for raw time-series

The 1D CNN uses temporal kernels tuned to the 25 Hz sampling rate:

```
Input (112, 9)
  -> Conv1D(32, k=7) -> BN -> ReLU -> MaxPool(2) -> Dropout   [k=7 ~ 280 ms]
  -> Conv1D(64, k=5) -> BN -> ReLU -> MaxPool(2) -> Dropout   [k=5 ~ 200 ms]
  -> Conv1D(128, k=3) -> BN -> ReLU -> GlobalAvgPool -> Dropout
  -> Dense(64, relu) -> Dropout
  -> Dense(4, softmax)
```

MaxPooling halves the temporal dimension twice (112 -> 56 -> 28) before GlobalAvgPool collapses it. This gives the network receptive fields that span one full PPG cycle (~280 ms at 70 bpm) down to fine-grained texture at 120 ms.

---

## 9. Configuration

All tuneable constants live in `config.py`.

```python
N_FOLDS = 5
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

# Raw CNN windowing
RAW_SAMPLING_HZ      = 25
RAW_WINDOW_S         = 4.5
RAW_WINDOW_SAMPLES   = 112   # = floor(4.5 * 25)
RAW_CHANNELS         = ["LED1", "LED2", "LED3", "Temp",
                         "AccX", "AccY", "AccZ",
                         "LED3_LED1", "LED3_LED2"]

# Hyperparameters
SVM_PARAMS  = {"C": 6.552, "kernel": "rbf", "gamma": 0.3003, ...}
RF_PARAMS   = {"n_estimators": 406, "max_depth": 15, ...}
XGB_PARAMS  = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, ...}
CNN_PARAMS  = {"filters_1": 32, "kernel_1": 7, "filters_2": 64, "kernel_2": 5,
               "filters_3": 128, "kernel_3": 3, "dense_1": 64, ...}
```

---

## 10. Output artefacts

| File | Description |
|------|-------------|
| `results/<dir>/fold_results.csv` | Per-fold accuracy, F1 macro, F1 weighted |
| `results/baseline/randomforest_model.pkl` | Serialized RF model |
| `results/baseline/svm_model.pkl` + `svm_scaler.pkl` | Serialized SVM + StandardScaler |
| `results/xgb_baseline/xgboost_model.pkl` | Serialized XGBoost model |
| `results/xgb_baseline/xgboost_model.onnx` | XGBoost in ONNX format |
| `results/cnn_raw_baseline/cnn_1d_model.pkl` | Serialized Keras model |
| `results/cnn_raw_baseline/cnn_1d_scaler.npz` | Per-channel mean/std for inference |
| `results/cnn_raw_baseline/cnn_1d_model.tflite` | **TFLite float32 -- input (1, 112, 9), output (1, 4)** |
| `results/<dir>/RandomForest_feature_importances.csv` | RF feature importances |
