# M3X Stress Classification Pipeline

A machine-learning pipeline that classifies momentary stress states from M3X ring sensor physiological features using four model types — SVM, Random Forest, XGBoost, and 1D CNN — evaluated with 5-fold Stratified Group Cross-Validation.

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

Participants wear an M3X ring sensor that continuously records PPG (three LED channels), accelerometer, and temperature data. Periodically, they receive a short EMA survey rating their current activity on two dimensions — **Opportunity** and **Pressure** — on a 1–6 Likert scale. The combination of these two ratings maps to one of four stress-state labels.

The pipeline trains a supervised classifier that predicts the momentary stress state from wearable sensor features, enabling passive stress monitoring without self-report.

**Models:** SVM · Random Forest · XGBoost · 1D CNN  
**Evaluation:** 5-fold Stratified Group Cross-Validation (`Block_ID` as group)  
**Dataset:** 1,288 rows · 38 features · 4 classes · 2 recording sessions (June 1 + June 2)

---

## 2. Repository structure

```
Project/
│
├── config.py                    # All tuneable constants (paths, hyperparams, class names)
├── train.py                     # Main entry point — trains models, runs CV, serializes outputs
├── tune.py                      # Hyperparameter search (RandomizedSearchCV)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── classify_survey.py   # Step 1: converts Qualtrics CSV → labelled survey CSV
│   │   ├── merge_labels.py      # Step 2: attaches survey labels to feature rows by timestamp
│   │   └── preprocess.py        # Load, clean, extract X / y / groups
│   │
│   ├── models/
│   │   ├── base.py              # Abstract BaseClassifier interface
│   │   ├── svm_model.py         # SVC (kernel=rbf, class_weight=balanced)
│   │   ├── rf_model.py          # RandomForestClassifier + feature_importances()
│   │   ├── xgb_model.py         # XGBClassifier (sample_weight=balanced)
│   │   └── cnn_model.py         # 1D CNN (Conv → GlobalAvgPool → Dense, early stopping)
│   │
│   └── evaluation/
│       ├── cross_val.py         # StratifiedGroupKFold runner
│       └── metrics.py           # Reporting: per-model summary, comparison table, CSV export
│
├── data/
│   ├── survey/
│   │   ├── M3X_EMA_session1_Jun1_labelled.csv   # Labelled survey — session 1
│   │   └── M3X_EMA_session2_Jun2_labelled.csv   # Labelled survey — session 2
│   │
│   └── features/
│       ├── 1_suhail_M3X_Features_AllBlocks_output.xlsx   # Raw MATLAB features — session 1
│       ├── 2_suhail_M3X_Features_AllBlocks_output.xlsx   # Raw MATLAB features — session 2
│       └── features_labeled_new.csv                      # Final labeled dataset (pipeline output)
│
├── results/
│   ├── baseline/                # SVM + RF baseline (pre-tuning)
│   │   ├── fold_results.csv
│   │   ├── svm_model.pkl · svm_scaler.pkl
│   │   ├── randomforest_model.pkl
│   │   └── RandomForest_feature_importances.csv
│   ├── xgb_baseline/            # XGBoost baseline
│   │   ├── fold_results.csv
│   │   ├── xgboost_model.pkl
│   │   └── xgboost_model.onnx
│   └── cnn_baseline/            # CNN baseline
│       ├── fold_results.csv
│       ├── cnn_1d_model.pkl · cnn_1d_scaler.pkl
│       └── cnn_1d_model.tflite  ← deployment artefact for Farhad
│
└── tests/
    ├── test_classify_survey.py  # 26 tests for EMA classification logic
    └── test_merge_labels.py     # 26 tests for label merging logic
```

---

## 3. Data description

### Labeled feature dataset — `data/features/features_labeled_new.csv`

1,288 rows · 49 columns (38 physiological features + index + label columns)

Each row is one ~5 s sub-window extracted by MATLAB from a 10 s recording block. Two sub-windows come from each block (`SubWindow_ID` = 1 or 2).

**Key columns:**

| Column | Role | Description |
|--------|------|-------------|
| `Block_ID` | Group | Globally unique 10 s block ID across both sessions |
| `SubWindow_ID` | Index | Sub-window index within the block (1 or 2) |
| `Block_Start_s` / `Block_End_s` | Index | Elapsed time of the block in seconds since session start |
| `Label` | Target | One of: `relax`, `eustress`, `distress`, `eustress+distress` |
| `PA1`, `PA2`, `PA3`, … | Feature | PPG amplitude features per LED channel |
| `HR_LED1`, `SDNN_LED1`, `RMSSD_LED1` | Feature | Heart rate and HRV features |
| `AccX_STD`, `AccY_STD`, `AccZ_STD`, `Acc_Total_STD` | Feature | Accelerometer standard deviation per axis |
| `mean_temperature` | Feature | Average skin temperature |

**Class distribution:**

| Label | Count | % |
|-------|------:|--:|
| relax | 468 | 36.3% |
| eustress+distress | 358 | 27.8% |
| distress | 286 | 22.2% |
| eustress | 176 | 13.7% |

**Sessions:**
- `Session_ID = 1` — June 1 recording
- `Session_ID = 2` — June 2 recording
- Session 2 Block IDs are offset to avoid collision with session 1

### Survey files — `data/survey/`

Two labelled survey CSVs produced by `classify_survey.py`. Each row is one EMA response with columns: `StartDate`, `EndDate`, `Opportunity`, `Pressure`, `Opportunity_score`, `Pressure_score`, `Label`.

---

## 4. Label taxonomy

Labels are assigned by `classify_survey.py` from the participant's Opportunity (opp) and Pressure (pres) scores (1–6 Likert scale). Threshold: ≤ 3 = Low, ≥ 4 = High.

| Label | Opportunity | Pressure | Interpretation |
|-------|:-----------:|:--------:|----------------|
| `relax` | Low | Low | Low demand, low activation |
| `eustress` | High | Low | Engaging without feeling pressured |
| `distress` | Low | High | Pressured without meaningful engagement |
| `eustress+distress` | High | High | High challenge and high pressure simultaneously |

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

The labeled dataset (`features_labeled_new.csv`) is already built and committed. Steps 1–3 are documented here for reference if re-running from raw data.

### Step 1 — Classify EMA surveys

Convert a raw Qualtrics CSV export into a labelled survey CSV:

```bash
python -m src.data.classify_survey \
    --input_dir  data/survey/ \
    --output_dir data/survey/
```

Handles both old text-format and new numeric-format Qualtrics exports.

### Step 2 — Merge labels onto feature rows (per session)

Attach a label to every feature row based on its block timestamp:

```bash
python -m src.data.merge_labels \
    --features  data/features/1_suhail_M3X_Features_AllBlocks_output.xlsx \
    --labels    data/survey/M3X_EMA_session1_Jun1_labelled.csv \
    --output    data/features/features_labeled_session1.csv

python -m src.data.merge_labels \
    --features  data/features/2_suhail_M3X_Features_AllBlocks_output.xlsx \
    --labels    data/survey/M3X_EMA_session2_Jun2_labelled.csv \
    --output    data/features/features_labeled_session2.csv
```

`Block_Start_s` / `Block_End_s` are real elapsed timestamps from MATLAB — no duty-cycle correction is applied.

### Step 3 — Combine sessions

Offset session 2 Block IDs to avoid collision, then concatenate:

```python
import pandas as pd

s1 = pd.read_csv("data/features/features_labeled_session1.csv")
s2 = pd.read_csv("data/features/features_labeled_session2.csv")
s2["Block_ID"] += s1["Block_ID"].max() + 1
combined = pd.concat([s1, s2], ignore_index=True)
combined = combined[combined["Label"] != "Unknown"]
combined.to_csv("data/features/features_labeled_new.csv", index=False)
```

### Step 4 — Train and evaluate

```bash
# All four models
python train.py --data data/features/features_labeled_new.csv

# Specific models only
python train.py --data data/features/features_labeled_new.csv --models svm rf

# Custom output directory
python train.py --data data/features/features_labeled_new.csv --models xgb --output_dir results/xgb_tuned
```

### Step 5 — Tune hyperparameters (optional)

```bash
python tune.py \
    --data    data/features/features_labeled_new.csv \
    --models  svm \
    --n-iter  40
```

Best params are written to `results/best_params.txt`. Update `config.py` with the new values and re-run `train.py`.

### Run tests

```bash
pytest tests/ -v
```

52 tests covering EMA classification logic and label merging logic.

---

## 7. Baseline results

All models trained on `data/features/features_labeled_new.csv` with 5-fold Stratified Group Cross-Validation (`Block_ID` as group — prevents sub-window leakage).

| Model | Accuracy | F1 Macro | Notes |
|-------|:--------:|:--------:|-------|
| **XGBoost** | **78.6%** ± 1.8% | **0.766** ± 0.018 | Best overall |
| Random Forest | 74.5% ± 2.8% | 0.700 ± 0.030 | Pre-tuning |
| SVM | 59.1% ± 2.7% | 0.510 ± 0.020 | Pre-tuning — params from old 3-class data |
| CNN (1D) | 49.9% ± 2.3% | 0.449 ± 0.029 | Underperforms on hand-engineered features |

**Per-class F1 for XGBoost (best model):**

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| relax | 0.81 | 0.82 | 0.82 |
| eustress+distress | 0.81 | 0.84 | 0.83 |
| distress | 0.75 | 0.73 | 0.74 |
| eustress | 0.71 | 0.65 | 0.68 |

Serialized models and fold results are in `results/`.

---

## 8. Design decisions

### StratifiedGroupKFold

Each 10 s recording block (`Block_ID`) produces two ~5 s sub-windows. Those sub-windows share physiological signal and are not independent. If a random split placed sub-windows from the same block in both the training and test folds, the model would partially memorise test windows during training.

`StratifiedGroupKFold(groups=Block_ID)` prevents this: all sub-windows from the same block are always in the same fold. Class distribution is preserved proportionally across folds.

### Scaler fitted per fold

For models requiring scaling (SVM, CNN), `StandardScaler` is fitted exclusively on the training portion of each fold and applied to the test portion without refitting. Fitting on the full dataset before splitting would leak test-set statistics into normalisation.

### Class-balanced weights

SVM, RF, and CNN use `class_weight="balanced"` or equivalent sample weighting. XGBoost uses `compute_sample_weight("balanced")`. This prevents the majority class (`relax`, 36%) from dominating the decision boundary.

---

## 9. Configuration

All tunable constants live in `config.py`.

```python
N_FOLDS = 5

SVM_PARAMS  = {"C": 6.552, "kernel": "rbf", "gamma": 0.3003, "class_weight": "balanced", ...}
RF_PARAMS   = {"n_estimators": 406, "max_depth": 15, "min_samples_leaf": 3, ...}
XGB_PARAMS  = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, ...}
CNN_PARAMS  = {"filters_1": 32, "filters_2": 64, "dense_1": 128, "epochs": 200, "patience": 25, ...}

CLASSES = ["relax", "eustress", "distress", "eustress+distress"]
```

---

## 10. Output artefacts

| File | Description |
|------|-------------|
| `results/<dir>/fold_results.csv` | Per-fold accuracy, F1 macro, F1 weighted for each model |
| `results/<dir>/randomforest_model.pkl` | Serialized RF model |
| `results/<dir>/svm_model.pkl` + `svm_scaler.pkl` | Serialized SVM + fitted scaler |
| `results/<dir>/xgboost_model.pkl` | Serialized XGBoost model |
| `results/<dir>/xgboost_model.onnx` | XGBoost in ONNX format |
| `results/<dir>/cnn_1d_model.pkl` + `cnn_1d_scaler.pkl` | Serialized CNN + fitted scaler |
| `results/cnn_baseline/cnn_1d_model.tflite` | CNN in TFLite format (float32, input: `[1, 38, 1]`, output: `[1, 4]`) |
| `results/<dir>/RandomForest_feature_importances.csv` | Mean decrease in impurity per feature |
