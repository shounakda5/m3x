# Implementation Report: Physiological Stress Classification Pipeline

**Project:** Workplace Lighting EMA Study — Stress State Classification from Wearable Sensor Data  
**Date:** May 4, 2026  
**Author:** Shounak Das

---

> **How to Read This Report**
> This document describes the full design and implementation of a machine learning pipeline for classifying momentary stress states from wrist-worn physiological sensors. It is written for a technical reader who has not seen the source code. Sections progress from data collection and preprocessing through model design, evaluation, and known limitations. Each section stands on its own, but reading in order gives the clearest picture.

---

## 1. Project Overview

This pipeline was built for a study conducted at USC in which participants wore a wrist sensor while periodically answering short surveys (Ecological Momentary Assessments, or EMAs) about their psychological state. The goal is to train a classifier that can infer a participant's stress state from physiological signals alone — without needing them to fill out a survey.

Stress is modeled along two independent dimensions: **Opportunity** (perceived challenge or potential upside) and **Pressure** (perceived demand or threat). Combining these two dimensions yields four theoretically distinct states:

| | Pressure LOW | Pressure HIGH |
|---|---|---|
| **Opportunity HIGH** | Eustress | Mixed |
| **Opportunity LOW** | Relaxed | Distress |

In this dataset, only three states are represented: **Relaxed**, **Distress**, and **Mixed**. No participant reported a pure Eustress state during the recording session, so the classifier is trained on three classes only.

---

## 2. Data Sources

The pipeline draws on two independent data streams that must be carefully aligned:

**Wrist Sensor Data:** A wrist-worn optical sensor collected photoplethysmography (PPG), heart rate, heart rate variability, and accelerometer signals at 24 Hz. Crucially, the sensor operated on a duty cycle — 10 seconds on, 20 seconds off — to conserve battery. Signals were pre-processed externally into a feature matrix of 861 rows and 16 physiological features per row.

**EMA Survey Data:** Participants answered 8 surveys via Qualtrics over the course of the session. Each survey asked them to rate Opportunity and Pressure on a 6-point Likert scale (text responses such as "Slightly agree"). Survey timestamps are wall-clock times; sensor timestamps are seconds elapsed since the sensor powered on.

Bridging these two timelines is the central data engineering challenge of the pipeline.

---

## 3. Pipeline Architecture

The full pipeline executes in four sequential stages:

```
Raw Qualtrics CSV
      │
      ▼
[Stage 1] classify_survey.py  →  Labelled survey CSV
                                          │
                       Feature CSV ───────▼
                                   [Stage 2] merge_labels.py  →  features_labeled.csv
                                                                          │
                                                                          ▼
                                                               [Stage 3] tune.py  (optional)
                                                                          │
                                                                          ▼
                                                               [Stage 4] train.py  →  Results
```

---

## 4. Stage 1 — Survey Label Classification

**Input:** Raw Qualtrics CSV export. The first two rows of this file are Qualtrics metadata headers; the pipeline skips them and reads actual responses from the third row onward.

**Label Assignment:** Each response contains free-text answers for Opportunity and Pressure. These are mapped to a numeric score from 1 to 6. A simple threshold rule then assigns the stress label:

- Score 4–6 is considered **HIGH**; score 1–3 is **LOW**
- Opportunity HIGH + Pressure LOW → **Eustress**
- Opportunity LOW + Pressure HIGH → **Distress**
- Both LOW → **Relaxed**
- Both HIGH → **Mixed**

**Output:** A CSV with one row per survey response, containing start/end timestamps, the raw text responses, numeric scores, and the assigned label. This file feeds directly into Stage 2.

**Session Summary:** 8 surveys were collected over approximately 3 hours (14:33 to 17:35). Labels were: 3 × Relaxed, 1 × Distress, 3 × Mixed, 1 × Eustress.

---

## 5. Stage 2 — Label Merging and Time Alignment

This is the most technically involved stage. It resolves the mismatch between two incompatible time systems and propagates survey labels onto every sensor feature row.

### 5a. Time Synchronization

Survey timestamps are absolute wall-clock times (e.g., `14:33:12`). Sensor timestamps are seconds elapsed since power-on. To align them, the pipeline designates the first survey's start time as `t = 0` and converts all survey timestamps to seconds relative to that origin. Only the time-of-day component is used (not the date), making the script safe for multi-day sessions.

### 5b. Duty-Cycle Correction

This correction is critical and easy to overlook. Because the sensor was off for 20 seconds of every 30-second cycle, the raw feature file's timestamps do not reflect real elapsed time. The raw file increments timestamps by 10 seconds per block (the on-period), but each block actually represents 30 seconds of real time. Without correction, every sensor measurement would be placed far too early in the session timeline and would be matched to the wrong survey label.

The fix adds `block_index × 20 seconds` to every timestamp, where block index is assigned by sorting the unique window values. For example:

| Raw Window | Corrected Window |
|---|---|
| 20.0 – 30.0 s | 40.0 – 50.0 s |
| 30.0 – 40.0 s | 70.0 – 80.0 s |
| 60.0 – 70.0 s | 160.0 – 170.0 s |

After correction, the sensor timeline spans `t = 12s` to `t = 8,598s`, consistent with 287 recording blocks × 30 seconds per cycle = 8,610 seconds of real elapsed time.

### 5c. Label Propagation

Once the timelines are aligned, labels are assigned using a **forward-fill** strategy:

- Any feature window that falls within a survey's [start, end] interval receives that survey's label directly.
- Any feature window in the **gap between two consecutive surveys** is labeled with the **next** survey's label (forward-fill). The rationale is that a participant's current state at the time of answering most closely reflects the preceding period.
- Feature windows **after the last survey** are labeled "Unknown" and excluded from training.

### 5d. Survey Coverage and Label Distribution

The sensor recording (8,610s) ended well before the survey session concluded (10,916s). Surveys 5–8 fall entirely outside the recording window, but their labels reach backward through the gaps to cover earlier sensor data.

| Survey | Label | Session Time | Feature Data Available |
|---|---|---|---|
| 1 | Relaxed | t = 0 – 28s | Yes |
| 2 | Relaxed | t = 884 – 912s | Yes |
| 3 | Distress | t = 2,840 – 2,870s | Yes |
| 4 | Relaxed | t = 3,396 – 3,416s | Yes |
| 5 | Mixed | t = 9,768 – 9,801s | No |
| 6 | Mixed | t = 10,109 – 10,137s | No |
| 7 | Eustress | t = 10,576 – 10,598s | No |
| 8 | Mixed | t = 10,898 – 10,916s | No |

Because Survey 5 (Mixed) is the next survey after the recording ends, its label forward-fills the entire 5,180-second gap between Survey 4 and the end of the recording. This results in a substantially imbalanced label distribution:

| Label | Rows | Percentage |
|---|---|---|
| Mixed | 519 | 60.3% |
| Distress | 197 | 22.9% |
| Relaxed | 145 | 16.8% |

This imbalance has direct implications for model design and evaluation, discussed in Sections 8 and 9.

---

## 6. Feature Set

Each row in the final labeled dataset represents a 3-second physiological sub-window nested within a 10-second recording block. The 16 input features span four physiological domains:

| Domain | Features |
|---|---|
| Heart Rate | HR_LED1 (raw), HR_LED1_Clean (cleaned) |
| Heart Rate Variability | HRV_LED1 (raw), HRV_LED1_Clean (cleaned) |
| PPG Amplitude — LED1 | PA_LED1, PA_LED1_Clean |
| PPG Amplitude — LED2 | PA_LED2, PA_LED2_Clean |
| PPG Amplitude Velocity | PAV_LED1, PAV_LED2 |
| Cross-Channel Derived | PA_LED1_minus_LED2, PA_LED1_div_LED2 |
| Motion (Accelerometer) | AccX_std, AccY_std, AccZ_std |
| Skin Temperature | Temp_Avg |

Both raw and cleaned variants are retained for HR, HRV, and PA features. The cleaned variants use signal processing to remove artifacts; retaining both gives the model the option to learn from either or use their difference implicitly.

---

## 7. Preprocessing

Before any model sees the data, three preprocessing steps are applied:

1. **Artifact Handling:** Infinite values (from division by zero in ratio features, for example) are replaced with NaN. All remaining NaN values are imputed with the column median — a robust choice that is unaffected by extreme outliers from sensor noise.

2. **Feature Scaling:** A `StandardScaler` (zero mean, unit variance) is applied to normalize features for the SVM. Critically, the scaler is fit **only on the training portion of each cross-validation fold** and then applied to the test portion. Fitting on the full dataset before splitting would leak test-set statistics (mean and variance) into training — a subtle but consequential form of data leakage. The Random Forest model skips scaling entirely, as tree-based models are scale-invariant.

3. **Group and Label Extraction:** Index columns (window identifiers and timestamps) are separated from model inputs. The `Window_10s` column is retained as a grouping key for cross-validation.

---

## 8. Cross-Validation Strategy

The pipeline uses **Stratified Group K-Fold** with 5 folds. This design simultaneously satisfies two requirements that are in tension:

- **Group constraint:** Each 10-second window contains 3 sub-windows (3-second rows). These rows share overlapping signals and are highly correlated. If they were split across train and test folds, the model would effectively see parts of the same epoch in training and be tested on trivially similar data — an optimistic and misleading evaluation. Grouping ensures all rows from the same 10-second window are always in the same fold.

- **Stratification:** With 287 unique windows spread across 5 folds (~57 windows per fold), stratification ensures each fold preserves the 60/23/17 class ratio. Without it, a fold could end up dominated by a single class, making per-fold metrics unstable and incomparable.

---

## 9. Models and Hyperparameter Tuning

Four classifiers were evaluated. All use class balancing strategies to counteract the 60/23/17 label imbalance.

**Support Vector Machine (SVM):** Uses an RBF (radial basis function) kernel. Probability outputs are enabled via Platt scaling. The key hyperparameters are `C` (regularization strength) and `gamma` (kernel bandwidth). Requires feature scaling.

**Random Forest (RF):** An ensemble of independently grown decision trees (bagging). Each tree votes on the final prediction. Key hyperparameters govern tree depth, minimum samples per split/leaf, number of trees, and feature sampling strategy. Scale-invariant; no feature scaling required.

**XGBoost:** A gradient-boosted tree ensemble. Unlike RF (which trains trees in parallel), XGBoost trains trees sequentially — each tree corrects the residual errors of the previous ensemble. This sequential correction makes it more sample-efficient than RF on small datasets. Key hyperparameters govern tree depth, learning rate (step-size shrinkage), row/column subsampling, and L1/L2 regularization. Scale-invariant; no feature scaling required. Per-sample class weights are computed inside `fit()` via `compute_sample_weight("balanced")` because XGBoost's multiclass objective has no native `class_weight` parameter.

**1D Convolutional Neural Network (CNN_1D):** A deep learning model that treats the 16-feature vector as a 1-D sequence and learns local feature interaction patterns via convolutional filters. Architecture:

```
Input       (16 features, 1 channel)
Conv1D(32, kernel=3, padding='same') → BatchNorm → ReLU
MaxPooling1D(2)  → Dropout(0.30)
Conv1D(64, kernel=3, padding='same') → BatchNorm → ReLU
GlobalAveragePooling1D → Dropout(0.30)
Dense(128) + ReLU + Dropout(0.40)          ← FC layer 1
Dense(64)  + ReLU + Dropout(0.30)          ← FC layer 2
Dense(32)  + ReLU + Dropout(0.20)          ← FC layer 3
Dense(n_classes) + Softmax                  ← FC layer 4 (output)
```

Training uses the Adam optimizer (lr = 1 × 10⁻³), sparse categorical cross-entropy loss, and per-class weights computed via `sklearn.utils.class_weight.compute_class_weight("balanced")`. EarlyStopping monitors validation loss with patience = 25 epochs and restores the best weights; ReduceLROnPlateau halves the learning rate after 8 stagnant epochs (minimum lr = 1 × 10⁻⁶). Up to 200 epochs are allowed, but early stopping typically terminates training well before that. Feature scaling is required.

Hyperparameters for SVM, RF, and XGBoost were tuned using `RandomizedSearchCV` with 60 iterations per model, using the same Stratified Group K-Fold strategy, scored by **macro F1**. CNN hyperparameters were tuned via a dedicated random search function in `tune.py` that manually iterates over 15 random draws from the CNN search space with the same CV strategy.

| Parameter | SVM Tuned | RF Tuned | XGB Tuned | CNN Tuned |
|---|---|---|---|---|
| C | 6.55 | — | — | — |
| gamma | 0.30 | — | — | — |
| n_estimators | — | 406 | 300 | — |
| max_depth | — | 15 | 4 | — |
| learning_rate | — | — | 0.05 | 1 × 10⁻³ |
| subsample | — | — | 0.80 | — |
| colsample_bytree | — | — | 0.80 | — |
| min_child_weight | — | — | 3 | — |
| gamma (XGB) | — | — | 0.10 | — |
| reg_alpha / reg_lambda | — | — | 0.10 / 1.0 | — |
| filters (conv 1/2) | — | — | — | 32 / 64 |
| dense units (FC 1/2/3) | — | — | — | 128 / 64 / 32 |
| conv / dense dropout | — | — | — | 0.30 / 0.40 |
| batch size | — | — | — | 32 |

---

## 10. Results

### 10a. Overall Performance

The Random Forest leads across all metrics, with XGBoost a close second. The SVM trails both tree ensembles, and the 1D CNN underperforms on this small tabular dataset as expected.

| Model | Accuracy | F1 Macro | F1 Weighted |
|---|---|---|---|
| SVM | 0.663 ± 0.035 | 0.555 ± 0.052 | 0.655 ± 0.038 |
| **Random Forest** | **0.759 ± 0.032** | **0.701 ± 0.042** | **0.760 ± 0.033** |
| XGBoost | 0.745 ± 0.042 | 0.676 ± 0.060 | 0.744 ± 0.042 |
| CNN_1D | 0.523 ± 0.025 | 0.497 ± 0.024 | 0.556 ± 0.022 |

F1 Macro is the primary metric. RF edges out XGBoost by 1.4 pp accuracy and 2.5 pp F1 Macro — a modest gap attributable to XGBoost's higher variance across folds (±0.060 vs ±0.042), likely due to the small dataset limiting boosting's error-correction advantage. The CNN achieves 0.497 macro F1, below the SVM, as expected for a convolutional model on unordered tabular features with only 861 samples.

### 10b. Per-Class Performance

| Class | SVM F1 | RF F1 | XGB F1 | CNN F1 | Support |
|---|---|---|---|---|---|
| Mixed | 0.80 | 0.85 | 0.85 | 0.65 | 519 |
| Distress | 0.44 | 0.59 | 0.56 | 0.41 | 197 |
| Relaxed | 0.43 | 0.67 | 0.63 | 0.44 | 145 |

All three tree-based models perform best on Mixed (the majority class). XGBoost matches RF on Mixed F1 (both 0.85) and comes within 0.04 F1 of RF on both minority classes, making it the strongest alternative to RF. The CNN over-predicts minority classes (high recall, low precision on Distress/Relaxed) at the cost of Mixed recall (0.48), yielding the lowest overall F1 despite Relaxed recall that rivals RF's (0.60 vs. 0.68).

### 10c. Feature Importances (Random Forest)

The RF's feature importance scores (mean decrease in impurity) identify skin temperature as the single most informative feature, followed by PPG amplitude from both LED channels:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Temp_Avg | 0.223 |
| 2 | PA_LED1_Clean | 0.101 |
| 3 | PA_LED2_Clean | 0.079 |
| 4 | AccX_std | 0.079 |
| 5 | PA_LED1 | 0.068 |

Skin temperature dominates at roughly twice the importance of any other feature. This is physiologically plausible — skin temperature changes with stress-induced vasoconstriction and is a relatively stable, low-noise signal compared to instantaneous heart rate.

---

## 11. Limitations

Several important caveats apply to interpreting these results:

1. **Missing Eustress class.** The four-quadrant stress model has no Eustress data in this session. The deployed classifier can only predict three states and will misclassify any true Eustress response.

2. **Mixed class label quality.** All 519 Mixed-labeled rows come from a single 5,180-second gap that was forward-filled with Survey 5's label. The participant was not actively reporting Mixed stress during this window — the label is inferred, not directly measured.

3. **Class imbalance.** Despite balanced class weighting, the 60/23/17 distribution limits how well the model can learn from Distress and Relaxed examples. Additional data collection would help substantially.

4. **Small dataset.** 861 rows across three classes is a limited training set for this type of signal. Cross-validation variance (±0.04 on macro F1) is nontrivial. This limitation is especially pronounced for the CNN: convolutional networks typically require thousands of samples to reliably learn spatial filters, and with only 861 rows the CNN underfits compared to tree-based ensembles.

5. **Feature ordering for CNN.** The 16 physiological features have no intrinsic spatial or temporal ordering within a single feature row. The 1D CNN treats adjacent features as "neighboring timesteps," but adjacency in the feature vector is an artifact of how features were engineered, not a physical relationship. Models that treat each feature independently (RF, SVM) are architecturally better suited to this tabular input.

6. **Single participant, single session.** No claim of generalizability can be made. A model trained on one person's physiology in one context may not transfer to other individuals or environments.

---

## 12. How to Run the Pipeline

```bash
# Stage 1: Classify survey responses into stress labels
python -m src.data.classify_survey \
  --input_dir data/survey/ \
  --output_dir data/survey/

# Stage 2: Align sensor timestamps and merge labels onto feature rows
python -m src.data.merge_labels \
  --features data/features/features_output_24hz.csv \
  --labels data/survey/Lighting_EMA_Main_April_6_2026_14.52_labelled.csv \
  --output data/features/features_labeled.csv

# Stage 3 (optional): Re-tune hyperparameters (--skip-cnn omits the slow CNN search)
python tune.py --data data/features/features_labeled.csv
python tune.py --data data/features/features_labeled.csv --skip-cnn    # SVM + RF only
python tune.py --data data/features/features_labeled.csv --cnn-iter 20 # more CNN trials

# Stage 4: Train all four models and evaluate with cross-validation
python train.py --data data/features/features_labeled.csv
```

Results are written to the `results/` directory: per-fold metrics in `fold_results.csv`, tuned parameters in `best_params.txt`, and RF feature importances in `RandomForest_feature_importances.csv`.

**Dependencies:** `pip install xgboost>=2.0 tensorflow>=2.12`. The CNN suppresses verbose TensorFlow logging automatically; only fold-level metrics are printed to stdout.

---

## 13. Summary

The pipeline classifies three momentary stress states (Relaxed, Distress, Mixed) from wrist-worn physiological sensor data using four models. The **Random Forest** is the best performer (75.9% accuracy, 0.701 macro F1). **XGBoost** is a close second (74.5% accuracy, 0.676 macro F1), with higher fold-to-fold variance (±0.060) reflecting the small training set limiting gradient boosting's error-correction advantage. The **SVM** reaches 66.3% accuracy and 0.555 macro F1. The **1D CNN** achieves 52.3% accuracy and 0.497 macro F1 — consistent with the known limitation of convolutional architectures on small, unordered tabular data.

The most technically sensitive parts of the implementation are the duty-cycle timestamp correction in Stage 2 (required for correct time alignment) and the grouped cross-validation strategy (which prevents data leakage from correlated sub-windows). Skin temperature is the dominant predictive feature, with PPG amplitude and accelerometer variability also contributing meaningfully.

The primary path to improving all models — and especially XGBoost and the CNN — is additional data: more participants, more sessions, and balanced representation across all four stress quadrants. With thousands of samples per class, XGBoost's sequential error-correction and the CNN's nonlinear feature-combination capacity could both surpass the Random Forest.
