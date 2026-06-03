# Stress Classification from Physiological EMA Data

A machine-learning pipeline that classifies momentary stress states — **Eustress**, **Distress**, **Relaxed**, **Mixed** — from wearable physiological time-series features using SVM and Random Forest classifiers, evaluated with stratified group k-fold cross-validation.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Data description](#3-data-description)
4. [Label taxonomy](#4-label-taxonomy)
5. [Installation](#5-installation)
6. [End-to-end workflow](#6-end-to-end-workflow)
7. [Design decisions — bias prevention](#7-design-decisions--bias-prevention)
8. [Configuration](#8-configuration)
9. [Output artefacts](#9-output-artefacts)
10. [Remaining steps to complete the experiment](#10-remaining-steps-to-complete-the-experiment)

---

## 1. Project overview

This project is part of a workplace lighting EMA (Ecological Momentary Assessment) study conducted at USC. Participants wear a physiological sensor and periodically receive short surveys asking them to rate their current activity as a source of **opportunity** and/or **pressure**. The combination of these two ratings maps to one of four stress-state labels.

The goal is to train a supervised classifier that predicts the momentary stress state from wearable sensor features, enabling passive stress monitoring without requiring self-report.

**Models compared:** Support Vector Machine (SVM) and Random Forest (RF)  
**Evaluation strategy:** 5-fold Stratified Group Cross-Validation (see [§7](#7-design-decisions--bias-prevention))

---

## 2. Repository structure

```
Project/
│
├── config.py                   # All tuneable constants (paths, params, class names)
├── train.py                    # Main entry point — runs CV and prints comparison
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── classify_survey.py  # Step 1: converts Qualtrics CSV → labelled CSV
│   │   ├── merge_labels.py     # Step 2: attach survey labels to feature rows by timestamp
│   │   └── preprocess.py       # Load, clean, extract X / y / groups
│   │
│   ├── models/
│   │   ├── base.py             # Abstract BaseClassifier interface
│   │   ├── svm_model.py        # SVC(kernel=rbf, class_weight=balanced)
│   │   └── rf_model.py         # RandomForestClassifier + feature_importances()
│   │
│   └── evaluation/
│       ├── cross_val.py        # StratifiedGroupKFold runner
│       └── metrics.py          # Reporting: per-model summary, comparison table, CSV export
│
├── data/
│   ├── survey/
│   │   ├── Lighting_EMA_Main_April_6_2026_14.52.csv          # Raw Qualtrics survey export
│   │   └── Lighting_EMA_Main_April_6_2026_14.52_labelled.csv # Survey labels (8 rows)
│   │
│   └── features/
│       ├── features_output_24hz.csv   # Physiological feature matrix (860 rows × 19 cols)
│       └── features_labeled.csv       # Merged feature + label dataset (created by merge_labels.py)
│
└── results/                    # Created at runtime by train.py
    ├── fold_results.csv
    └── RandomForest_feature_importances.csv
```

---

## 3. Data description

### Physiological feature file — `data/features/features_output_24hz.csv`

860 rows · 19 columns

| Column | Role | Description |
|--------|------|-------------|
| `Window_10s` | **Index** | 10-second epoch the row belongs to (e.g. `"10.0-20.0"`) |
| `SubWindow_3s` | **Index** | 3-second sub-window within that epoch (e.g. `"10.5-13.5"`) |
| `Time_Mid` | **Index** | Midpoint of the sub-window in **seconds elapsed since recording start** |
| `HR_LED1` | Feature | Heart rate from LED1 (raw) |
| `HRV_LED1` | Feature | Heart rate variability from LED1 (raw) |
| `PA_LED1` / `PA_LED2` | Feature | Photoplethysmogram amplitude, both channels |
| `PAV_LED1` / `PAV_LED2` | Feature | Amplitude velocity, both channels |
| `PA_LED1_minus_LED2` | Feature | Channel difference |
| `PA_LED1_div_LED2` | Feature | Channel ratio |
| `AccX_std` / `AccY_std` / `AccZ_std` | Feature | Accelerometer standard deviation per axis |
| `Temp_Avg` | Feature | Average skin temperature |
| `HR_LED1_Clean` / `HRV_LED1_Clean` | Feature | Cleaned/artifact-filtered HR and HRV |
| `PA_LED1_Clean` / `PA_LED2_Clean` | Feature | Cleaned amplitude signals |

**Recording span:** `Time_Mid` ranges from **12 s** (first sub-window: `10.5–13.5 s`) to **2878 s** (last sub-window: `2876.5–2879.5 s`), giving a total recording length of **≈ 47 minutes 58 seconds**.

**Sub-window structure:** Each 10-second epoch contains three 3-second sub-windows (slight overlap). All three sub-windows of the same epoch share the same Window_10s label — this matters for cross-validation grouping (see §7).

### Survey label file — `data/survey/Lighting_EMA_Main_April_6_2026_14.52_labelled.csv`

8 labelled survey responses, produced by `src/data/classify_survey.py` from the raw Qualtrics export.

| Column | Description |
|--------|-------------|
| `StartDate` | Wall-clock time the participant opened the survey |
| `EndDate` | Wall-clock time the participant submitted the survey |
| `Opportunity` / `Pressure` | Raw free-text responses |
| `Opportunity_score` / `Pressure_score` | Numeric scale 1–6 |
| `Label` | One of: `Eustress`, `Distress`, `Relaxed`, `Mixed` |

Survey windows from the dataset:

| # | StartDate | EndDate | Label |
|---|-----------|---------|-------|
| 1 | 2026-03-31 14:33:12 | 2026-03-31 14:33:40 | Relaxed |
| 2 | 2026-03-31 14:47:56 | 2026-03-31 14:48:24 | Relaxed |
| 3 | 2026-03-31 15:20:32 | 2026-03-31 15:21:02 | Distress |
| 4 | 2026-03-31 15:29:48 | 2026-03-31 15:30:08 | Relaxed |
| 5 | 2026-03-31 17:16:00 | 2026-03-31 17:16:33 | Mixed |
| 6 | 2026-03-31 17:21:41 | 2026-03-31 17:22:09 | Mixed |
| 7 | 2026-03-31 17:29:28 | 2026-03-31 17:29:50 | Eustress |
| 8 | 2026-03-31 17:34:50 | 2026-03-31 17:35:08 | Mixed |

---

## 4. Label taxonomy

Labels are assigned by `classify_survey.py` based on the participant's Opportunity and Pressure ratings (6-point Likert scale, 1 = "very definitely NOT a source", 6 = "very definitely IS a source"):

| Label | Opportunity score | Pressure score | Interpretation |
|-------|:-----------------:|:--------------:|----------------|
| **Eustress** | 4–6 (High) | 1–3 (Low) | Engaging without feeling pressured |
| **Distress** | 1–3 (Low) | 4–6 (High) | Pressured without meaningful engagement |
| **Relaxed** | 1–3 (Low) | 1–3 (Low) | Low demand, low activation |
| **Mixed** | 4–6 (High) | 4–6 (High) | High challenge and high pressure simultaneously |

---

## 5. Installation

```bash
# Python 3.10+ required (uses X | Y union type hints)
pip install -r requirements.txt
```

`requirements.txt` pins the minimum versions:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
```

---

## 6. End-to-end workflow

Once labels can be merged (see §10 for the current blocker), the full pipeline is three commands:

### Step 1 — Generate survey labels

Convert a raw Qualtrics CSV export into a labelled CSV:

```bash
python -m src.data.classify_survey \
    --input_dir  data/survey/ \
    --output_dir data/survey/
```

Output: `data/survey/<survey_filename>_labelled.csv`  
This step is already complete for the current dataset.

### Step 2 — Merge labels onto feature rows

Attach a label to every physiological feature row whose timestamp falls within a survey response window:

```bash
python -m src.data.merge_labels \
    --features  data/features/features_output_24hz.csv \
    --labels    data/survey/Lighting_EMA_Main_April_6_2026_14.52_labelled.csv \
    --output    data/features/features_labeled.csv
```

The script corrects the feature timestamps for the sensor's 30 s duty cycle (10 s ON, 20 s OFF) and assigns each feature row the label of the survey window that governs it. The first survey's StartDate is treated as `t = 0`; gaps between surveys are labelled with the next survey's label; rows after the last survey are marked `"Unknown"`.

Optional flags for different sensor configurations:
```
--duty-cycle  30   # total duty-cycle in seconds
--on-period   10   # sensor ON period per cycle in seconds
```

The script prints a coverage summary:
```
Rows total : 860
Labeled    : XXX  (XX.X%)
Unknown    : YYY  (XX.X%)

Label distribution (labeled rows only):
  Relaxed       ...
  Distress      ...
  ...
```

### Step 3 — Train and evaluate

```bash
python train.py --data data/features/features_labeled.csv
```

Optional flags:
```
--n_folds     5        # number of CV folds (default from config.py)
--output_dir  results  # where to write result CSVs
```

The script prints per-fold accuracy, per-class precision/recall/F1 (pooled over folds), a confusion matrix, and a side-by-side model comparison table. Results CSVs are written to `results/`.

---

## 7. Design decisions — bias prevention

### StratifiedGroupKFold

Every `Window_10s` epoch contains three overlapping `SubWindow_3s` rows. Those sub-windows share physiological signal and are therefore **not independent observations**. If a random split placed some sub-windows of the same epoch in the training fold and others in the test fold, the model would effectively memorise part of a test window during training — a form of data leakage that inflates reported accuracy.

We prevent this using `StratifiedGroupKFold(groups=Window_10s)`:

- **Group constraint** — all sub-windows of the same 10-second epoch are always in the same fold.
- **Stratified** — the class distribution (Eustress / Distress / Relaxed / Mixed) is preserved proportionally in each fold, preventing a fold from being dominated by a single label.

### Scaler fitted per fold, not upfront

For the SVM (which is sensitive to feature scale), `StandardScaler` is fitted **exclusively on the training portion** of each fold, then applied to the test portion without refitting. Fitting the scaler on the full dataset before splitting would leak test-set statistics (mean, variance) into the normalisation step.

The Random Forest does not receive scaling at all — tree-based models are scale-invariant.

### Class-balanced model weights

Both models are initialised with `class_weight="balanced"`. This scales each class's loss contribution inversely to its frequency, preventing the majority class from dominating the decision boundary in an imbalanced dataset.

---

## 8. Configuration

All tunable constants live in `config.py`. Editing this file is the only change needed to adjust paths, hyperparameters, or the fold count.

```python
# Cross-validation
N_FOLDS = 5

# SVM
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale", ...}

# Random Forest
RF_PARAMS = {"n_estimators": 200, "max_depth": None, ...}
```

---

## 9. Output artefacts

| File | Description |
|------|-------------|
| `results/fold_results.csv` | Per-fold accuracy, F1 macro, F1 weighted for every model |
| `results/RandomForest_feature_importances.csv` | Mean decrease in impurity for each feature, sorted descending |
| `data/features/features_labeled.csv` | Merged feature + label dataset (created by merge_labels.py) |
| `data/survey/*_labelled.csv` | Survey label files (created by classify_survey.py) |

---

## 10. Remaining steps to complete the experiment

| # | Step | Command / action | Status |
|---|------|-----------------|--------|
| 1 | Generate survey labels | `python -m src.data.classify_survey ...` | ✅ Done |
| 2 | Merge labels onto feature rows | `python -m src.data.merge_labels ...` | ✅ Done |
| 3 | Inspect label coverage and class balance | Review merge output summary | ✅ Done — see note below |
| 4 | Train and evaluate models | `python train.py --data data/features/features_labeled.csv` | ⏳ Ready |
| 5 | Review results and tune hyperparameters | Edit `config.py`, re-run `train.py` | ⏳ Waiting on step 4 |
| 6 | Collect more survey data (recommended) | Run additional EMA sessions | ⬜ Optional but strongly advised |

### Label coverage — current dataset

After duty-cycle correction the feature timeline spans **t = 12 s – 8598 s** (≈ 2.4 hours). The survey span is **t = 0 s – 10 916 s** (3 h 1 m 56 s). Since the feature recording (8 610 s) is shorter than the survey span, **every feature row receives a label** via the forward-fill rule. The missing data is on the survey side: surveys 5–8 all fall after the recording ended and have no corresponding feature rows.

| Label | Rows | % | Time region |
|-------|-----:|--:|------------|
| Mixed | 519 | 60.3% | t = 3416 – 8598 s — gap before survey 5 (forward-filled with survey 5's label) |
| Distress | 197 | 22.9% | t = 912 – 2870 s — gap before survey 3 + survey 3 window |
| Relaxed | 145 | 16.8% | t = 0 – 912 s and t = 2870 – 3416 s — surveys 1 & 2 region and gap + survey 4 |
| Eustress | 0 | — | Survey 7 (Eustress) at t = 10 576 s is beyond the recording |

**Per-survey status:**

| Survey | Label | Time (relative) | Feature data |
|--------|-------|----------------|-------------|
| 1 | Relaxed | t = 0 – 28 s | covered |
| 2 | Relaxed | t = 884 – 912 s | covered |
| 3 | Distress | t = 2840 – 2870 s | covered |
| 4 | Relaxed | t = 3396 – 3416 s | covered |
| 5 | Mixed | t = 9768 – 9801 s | **no feature data** (recording ends 1170 s before this) |
| 6 | Mixed | t = 10109 – 10137 s | **no feature data** |
| 7 | Eustress | t = 10576 – 10598 s | **no feature data** |
| 8 | Mixed | t = 10898 – 10916 s | **no feature data** |

**Key observations:**
- **No Eustress rows** — the only Eustress survey (survey 7) falls 1 978 s after the recording ends. A 4-class classifier cannot be trained on this session alone.
- **Mixed is forward-filled from survey 5** — all 519 Mixed rows come from the ~5 180 s gap before survey 5. The actual survey 5 window starts 1 170 s after the recording ends; its label reaches back to cover the trailing portion of the recording.
- **Recommendation** — collect additional EMA sessions whose recording windows extend into the later survey windows, particularly sessions that yield Eustress labels.
