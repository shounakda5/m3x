# M3X Stress Classification Pipeline

A machine-learning pipeline that classifies momentary stress states from M3X ring sensor physiological data using four model types evaluated with 5-fold Stratified Group Cross-Validation. Each model is trained and evaluated on **two dataset variants** for a comparative study: the full unfiltered dataset and a motion-filtered subset.

- **SVM, Random Forest, XGBoost** -- trained on MATLAB-extracted engineered features
- **1D CNN** -- trained directly on raw sensor time-series windows (LED1/2/3, Temp, Acc)

Part of a workplace lighting EMA (Ecological Momentary Assessment) study at USC.

> **AI assistants**: read [`AI_AGENT_GUIDE.md`](AI_AGENT_GUIDE.md) first — it maps the whole repo (original pipeline, `pilot_b/`, `PilotPhaseC/`) and covers implementation detail not written down elsewhere.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Data description](#3-data-description)
4. [Label taxonomy](#4-label-taxonomy)
5. [Installation](#5-installation)
6. [End-to-end workflow](#6-end-to-end-workflow)
7. [Results](#7-results)
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

**Two dataset variants (each pipeline runs on both):**

| Variant | Description | Feature rows | CNN windows |
|---------|-------------|:------------:|:-----------:|
| Unfiltered | All blocks from both sessions | 1,288 | 1,540 |
| Motion-filtered (`--mf`) | Blocks with high-motion artifacts removed by MATLAB | 931 | 896 |

**Evaluation:** 5-fold Stratified Group CV -- all sub-windows from the same duty-cycle burst share the same fold, preventing correlated-sample leakage.

---

## 2. Repository structure

```
Project/
|
+-- config.py                    # All tuneable constants (paths, hyperparams, class names)
+-- train.py                     # Feature-based models: SVM, RF, XGBoost
+-- train_cnn.py                 # Raw time-series CNN (supports --mf for motion-filtered)
+-- tune.py                      # Hyperparameter search (RandomizedSearchCV)
+-- requirements.txt
|
+-- src/
|   +-- data/
|   |   +-- classify_survey.py   # Step 1: Qualtrics CSV -> labelled survey CSV
|   |   +-- merge_labels.py      # Step 2: attach survey labels to feature rows by timestamp
|   |   +-- preprocess.py        # Feature pipeline: load, clean, extract X/y/groups
|   |   \-- raw_preprocess.py    # Raw pipeline: burst detection, windowing, label matching
|   |                            #   also: load_raw_mf_cnn_dataset() for motion-filtered xlsx
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
|   |   +-- session1_raw.csv          # Raw M3X sensor data -- session 1 (June 1)
|   |   +-- session2_raw.csv          # Raw M3X sensor data -- session 2 (June 2)
|   |   +-- session1_raw_mf.xlsx      # Motion-filtered raw -- session 1 (MATLAB output)
|   |   \-- session2_raw_mf.xlsx      # Motion-filtered raw -- session 2 (MATLAB output)
|   |
|   +-- features/
|   |   +-- session1_features.xlsx    # MATLAB features -- session 1, unfiltered
|   |   +-- session2_features.xlsx    # MATLAB features -- session 2, unfiltered
|   |   +-- session1_features_mf.xlsx # MATLAB features -- session 1, motion-filtered
|   |   +-- session2_features_mf.xlsx # MATLAB features -- session 2, motion-filtered
|   |   +-- features_labeled.csv      # Final labeled unfiltered dataset (pipeline output)
|   |   \-- features_labeled_mf.csv   # Final labeled motion-filtered dataset
|   |
|   \-- survey/
|       +-- M3X_EMA_session1_Jun1_labelled.csv
|       \-- M3X_EMA_session2_Jun2_labelled.csv
|
+-- results/
|   +-- baseline/              # SVM + RF on unfiltered data
|   |   +-- randomforest_model.pkl + .onnx
|   |   \-- svm_model.pkl + svm_scaler.pkl + .onnx
|   +-- xgb_baseline/          # XGBoost on unfiltered data
|   |   \-- xgboost_model.pkl + .onnx
|   +-- xgb_mf_baseline/       # XGBoost on motion-filtered data
|   |   \-- xgboost_model.pkl + .onnx
|   +-- cnn_raw_baseline/      # 1D CNN on unfiltered raw data (best overall model)
|   |   +-- cnn_1d_model.pkl + cnn_1d_scaler.npz
|   |   \-- cnn_1d_model.tflite   <- deployment artefact for Farhad
|   \-- cnn_raw_mf_baseline/   # 1D CNN on motion-filtered raw data
|       +-- cnn_1d_model.pkl + cnn_1d_scaler.npz
|       \-- cnn_1d_model.tflite
|
\-- tests/
    +-- test_classify_survey.py  # 26 tests
    \-- test_merge_labels.py     # 26 tests
```

---

## 3. Data description

### Unfiltered raw sensor data -- `data/raw/session{1,2}_raw.csv`

Two CSV files, one per recording session. Each file contains ~75,000-87,000 rows.

| Column | Description |
|--------|-------------|
| `Time_PPG` | Elapsed time in **milliseconds** since device boot |
| `LED1` | PPG amplitude, LED channel 1 (raw ADC counts) |
| `LED2` | PPG amplitude, LED channel 2 |
| `LED3` | PPG amplitude, LED channel 3 |
| `Temp` | Skin temperature |
| `AccX/Y/Z` | 3-axis accelerometer |

Duty cycle: ~10 s ON / ~21 s OFF. Each ON burst contains ~250 samples at ~25 Hz.
The CNN uses 9 channels: the 7 above plus `LED3-LED1` and `LED3-LED2` computed on the fly.

### Motion-filtered raw data -- `data/raw/session{1,2}_raw_mf.xlsx`

Same sensor channels, but MATLAB has already removed bursts where motion artifacts were detected and annotated each row with block and sub-window metadata (`Block_ID`, `SubWindow_ID`, `Block_Start_s`, etc.). Time_PPG is in **seconds** (converted from ms). These files contain ~48,000-57,000 rows.

### Labeled feature datasets

| File | Rows | Blocks | Description |
|------|-----:|------:|-------------|
| `features_labeled.csv` | 1,288 | 644 | All blocks, unfiltered |
| `features_labeled_mf.csv` | 931 | 561 | Motion-filtered subset |

Each row is one ~5 s sub-window (2 per block). 38 engineered features + metadata + label.

**Class distribution (unfiltered vs motion-filtered):**

| Label | Unfiltered | MF |
|-------|:----------:|:--:|
| relax | 468 (36.3%) | 343 (36.8%) |
| eustress+distress | 358 (27.8%) | 277 (29.8%) |
| distress | 286 (22.2%) | 40 (4.3%) |
| eustress | 176 (13.7%) | 271 (29.1%) |

Motion filtering disproportionately removed distress blocks (likely higher-motion events), reducing distress from 22% to 4% and increasing eustress proportionally.

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
skl2onnx>=1.16
onnxmltools>=1.11
openpyxl>=3.1
```

---

## 6. End-to-end workflow

The labeled datasets are already built and committed. Steps 1-3 are documented for reference if re-running from raw data.

### Feature-based pipeline (SVM / RF / XGBoost)

**Step 1 -- Classify EMA surveys**
```bash
python -m src.data.classify_survey --input_dir data/survey/ --output_dir data/survey/
```

**Step 2 -- Merge labels onto feature rows (per session, run for both variants)**
```bash
# Unfiltered
python -m src.data.merge_labels \
    --features data/features/session1_features.xlsx \
    --labels   data/survey/M3X_EMA_session1_Jun1_labelled.csv \
    --output   data/features/features_labeled_s1.csv

python -m src.data.merge_labels \
    --features data/features/session2_features.xlsx \
    --labels   data/survey/M3X_EMA_session2_Jun2_labelled.csv \
    --output   data/features/features_labeled_s2.csv

# Motion-filtered (same command -- merge_labels auto-detects the sheet name)
python -m src.data.merge_labels \
    --features data/features/session1_features_mf.xlsx \
    --labels   data/survey/M3X_EMA_session1_Jun1_labelled.csv \
    --output   data/features/features_labeled_mf_s1.csv

python -m src.data.merge_labels \
    --features data/features/session2_features_mf.xlsx \
    --labels   data/survey/M3X_EMA_session2_Jun2_labelled.csv \
    --output   data/features/features_labeled_mf_s2.csv
```

**Step 3 -- Combine sessions**
```python
import pandas as pd

def combine(s1_path, s2_path, out_path):
    s1 = pd.read_csv(s1_path)
    s2 = pd.read_csv(s2_path)
    s1.insert(0, "Session_ID", 1); s1.insert(1, "Participant_ID", 501)
    s2.insert(0, "Session_ID", 2); s2.insert(1, "Participant_ID", 501)
    s2["Block_ID"] += s1["Block_ID"].max() + 1
    combined = pd.concat([s1, s2], ignore_index=True)
    combined[combined["Label"] != "Unknown"].to_csv(out_path, index=False)

combine("data/features/features_labeled_s1.csv",
        "data/features/features_labeled_s2.csv",
        "data/features/features_labeled.csv")

combine("data/features/features_labeled_mf_s1.csv",
        "data/features/features_labeled_mf_s2.csv",
        "data/features/features_labeled_mf.csv")
```

**Step 4 -- Train feature-based models**
```bash
# Unfiltered
python train.py --data data/features/features_labeled.csv --output_dir results/baseline
python train.py --data data/features/features_labeled.csv --models xgb --output_dir results/xgb_baseline

# Motion-filtered
python train.py --data data/features/features_labeled_mf.csv --models xgb --output_dir results/xgb_mf_baseline
```

**Tune hyperparameters (optional)**
```bash
python tune.py --data data/features/features_labeled.csv --models svm --n-iter 40
```

### Raw time-series pipeline (1D CNN)

```bash
# Unfiltered (default)
python train_cnn.py

# Motion-filtered
python train_cnn.py --mf
```

**What each mode does internally:**

*Unfiltered:* loads raw CSVs, detects duty-cycle ON bursts from 1 s timestamp gaps, discards bursts < 8 s, splits each burst into 2 non-overlapping 112-sample windows, matches burst start times to the labeled feature CSV (tolerance 2 s).

*Motion-filtered (--mf):* loads the pre-segmented MATLAB Excel files where Block_ID and SubWindow_ID are already annotated; groups rows by (Block_ID, SubWindow_ID) and takes the first 112 rows from each group; groups with fewer than 112 rows are dropped (~4% loss).

### Run tests
```bash
pytest tests/ -v
```

---

## 7. Results

All models evaluated with 5-fold Stratified Group CV (group = Block_ID).

### Unfiltered dataset

| Model | Input | Accuracy | F1 Macro | Export |
|-------|-------|:--------:|:--------:|--------|
| **CNN (1D, raw)** | Raw 112x9 windows | **80.6%** +/- 2.4% | **0.801** +/- 0.023 | TFLite |
| XGBoost | 38 features | 78.6% +/- 1.8% | 0.766 +/- 0.018 | ONNX |
| Random Forest | 38 features | 74.5% +/- 2.8% | 0.700 +/- 0.033 | ONNX |
| SVM | 38 features | 59.1% +/- 2.7% | 0.512 +/- 0.023 | ONNX |

### Motion-filtered dataset

| Model | Input | Accuracy | F1 Macro | Export |
|-------|-------|:--------:|:--------:|--------|
| **CNN (1D, raw MF)** | Raw 112x9 windows | **81.0%** +/- 3.6% | **0.758** +/- 0.060 | TFLite |
| XGBoost MF | 38 features | 79.4% +/- 4.0% | 0.705 +/- 0.028 | ONNX |

Note: motion filtering removes most distress blocks (4.3% vs 22.2% in unfiltered), which lowers macro F1 for the MF runs despite similar or slightly better accuracy. The low distress support in MF data makes that class harder to learn; distress F1 drops from 0.80 (CNN unfiltered) to 0.56 (CNN MF).

**Per-class F1 -- CNN_1D unfiltered (best deployment model):**

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| eustress+distress | 0.90 | 0.86 | 0.88 |
| distress | 0.71 | 0.90 | 0.80 |
| relax | 0.85 | 0.71 | 0.78 |
| eustress | 0.72 | 0.79 | 0.75 |

**Per-class F1 -- CNN_1D motion-filtered:**

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| eustress+distress | 0.89 | 0.87 | 0.88 |
| eustress | 0.81 | 0.79 | 0.80 |
| relax | 0.81 | 0.80 | 0.80 |
| distress | 0.47 | 0.69 | 0.56 |

---

## 8. Design decisions

### StratifiedGroupKFold

Each duty-cycle burst produces two sub-windows. Sub-windows from the same burst share physiological signal and are not independent. All sub-windows from the same burst (Block_ID) are always in the same fold, preventing train/test leakage.

### Scaler discipline

- **Feature models (SVM):** StandardScaler fitted on training fold only, applied to test fold.
- **CNN (raw):** per-channel z-normalisation (mean/std over all training windows and timesteps), fitted on training fold only. The final scaler is saved as `cnn_1d_scaler.npz`.

### Class-balanced weights

SVM, RF, and CNN all apply class weighting proportional to inverse class frequency. XGBoost uses `compute_sample_weight("balanced")`. This prevents the majority class (`relax`, ~36%) from dominating.

### Motion-filtered CNN data loader

The unfiltered raw CSVs require burst detection from timestamp gaps. The motion-filtered MATLAB Excel files already have Block_ID and SubWindow_ID embedded. The MF loader (`load_raw_mf_cnn_dataset`) groups rows by (Block_ID, SubWindow_ID) and takes the first 112 rows. Sub-windows with fewer than 112 rows (~3-4%) are dropped. Labels are matched by (Session_ID, Block_Start_s) with a 2 s tolerance to the combined labeled CSV.

### ONNX / TFLite exports

All trained models are exported to a portable format alongside the `.pkl`:
- **CNN** -> TFLite float32 (input `(1, 112, 9)`, output `(1, 4)`)
- **XGBoost** -> ONNX (via onnxmltools)
- **Random Forest, SVM** -> ONNX (via skl2onnx)

### CNN architecture for raw time-series

```
Input (112, 9)
  -> Conv1D(32, k=7) -> BN -> ReLU -> MaxPool(2) -> Dropout   [k=7 ~ 280 ms]
  -> Conv1D(64, k=5) -> BN -> ReLU -> MaxPool(2) -> Dropout   [k=5 ~ 200 ms]
  -> Conv1D(128, k=3) -> BN -> ReLU -> GlobalAvgPool -> Dropout
  -> Dense(64, relu) -> Dropout
  -> Dense(4, softmax)
```

MaxPooling halves the temporal dimension twice (112 -> 56 -> 28) before GlobalAvgPool collapses it.

---

## 9. Configuration

All tuneable constants live in `config.py`.

```python
N_FOLDS = 5
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

# Dataset paths
LABELED_FEATURES_FILE    = "data/features/features_labeled.csv"
LABELED_FEATURES_MF_FILE = "data/features/features_labeled_mf.csv"
RAW_SESSION1_FILE        = "data/raw/session1_raw.csv"
RAW_MF_SESSION1_FILE     = "data/raw/session1_raw_mf.xlsx"

# Raw CNN windowing
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
| `results/baseline/randomforest_model.pkl` + `.onnx` | RF model (unfiltered) |
| `results/baseline/svm_model.pkl` + `svm_scaler.pkl` + `.onnx` | SVM + scaler (unfiltered) |
| `results/xgb_baseline/xgboost_model.pkl` + `.onnx` | XGBoost (unfiltered) |
| `results/xgb_mf_baseline/xgboost_model.pkl` + `.onnx` | XGBoost (motion-filtered) |
| `results/cnn_raw_baseline/cnn_1d_model.tflite` | **CNN TFLite -- unfiltered, best deployment model** |
| `results/cnn_raw_baseline/cnn_1d_scaler.npz` | Per-channel mean/std (unfiltered) |
| `results/cnn_raw_mf_baseline/cnn_1d_model.tflite` | CNN TFLite -- motion-filtered |
| `results/cnn_raw_mf_baseline/cnn_1d_scaler.npz` | Per-channel mean/std (motion-filtered) |
| `results/baseline/RandomForest_feature_importances.csv` | RF feature importances |
