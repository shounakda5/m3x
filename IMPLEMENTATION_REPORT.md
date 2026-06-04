# Implementation Report: Physiological Stress Classification Pipeline

**Project:** Workplace Lighting EMA Study -- Stress State Classification from Wearable Sensor Data
**Updated:** June 2026
**Author:** Shounak Das

---

> **How to Read This Report**
> This document describes the full design and implementation of a machine learning pipeline for classifying momentary stress states from M3X ring sensor physiological data. It is written for a technical reader who has not seen the source code. Sections progress from data collection and preprocessing through model design, evaluation, and known limitations. Each section stands on its own, but reading in order gives the clearest picture.

---

## 1. Project Overview

This pipeline was built for a study conducted at USC in which participants wore an M3X ring sensor while periodically answering short surveys (Ecological Momentary Assessments, or EMAs) about their psychological state. The goal is to train a classifier that can infer a participant's stress state from physiological signals alone -- without needing them to fill out a survey.

Stress is modeled along two independent dimensions: **Opportunity** (perceived challenge or engagement) and **Pressure** (perceived demand or threat). Scores are collected on a 6-point Likert scale; scores 1-3 are treated as Low and scores 4-6 as High. Combining the two dimensions yields four distinct states:

| | Pressure LOW | Pressure HIGH |
|---|---|---|
| **Opportunity HIGH** | eustress | eustress+distress |
| **Opportunity LOW** | relax | distress |

All four states are present in the dataset. Data was collected across two recording sessions: June 1 and June 2.

**Two parallel training pipelines operate on different input representations:**

| Pipeline | Input | Models |
|---|---|---|
| Feature-based | 38 MATLAB-extracted features per ~5 s sub-window | SVM, Random Forest, XGBoost |
| Raw time-series | 112 samples x 9 channels (~4.5 s window at 25 Hz) | 1D CNN |

---

## 2. Data Sources

The pipeline draws on three independent data streams that must be carefully aligned:

**Raw Sensor Data:** Two CSV files (`data/raw/session1_raw.csv`, `data/raw/session2_raw.csv`), one per session. Each file contains 75,000-87,000 rows of continuous M3X ring sensor output: three LED PPG channels, accelerometer (3 axes), and skin temperature at approximately 25 Hz. The sensor operates on a 10 s ON / ~21 s OFF duty cycle to conserve battery -- each ON burst contains roughly 250 samples.

**MATLAB Feature Files:** Two Excel files (`data/features/1_suhail_M3X_Features_AllBlocks_output.xlsx`, `data/features/2_suhail_M3X_Features_AllBlocks_output.xlsx`), one per session. Each row represents one ~5 s sub-window and contains 38 engineered physiological features extracted by Farhad's MATLAB pipeline from a single ON burst.

**EMA Survey Data:** Two CSV files in `data/survey/`, one per session. Each contains survey responses with wall-clock timestamps and Opportunity/Pressure Likert ratings.

Bridging the raw sensor timestamps, the MATLAB feature block timestamps, and the survey wall-clock timestamps is the central data engineering challenge.

---

## 3. Pipeline Architecture

Two independent training paths share the label assignment infrastructure (Stages 1-3) but diverge at training time.

```
[Stage 1] classify_survey.py  -->  Labelled survey CSVs (per session)

[Stage 2] merge_labels.py (x2)  -->  features_labeled_session1.csv
                                      features_labeled_session2.csv

[Stage 3] Combine sessions  -->  data/features/features_labeled_new.csv
                                             |
               +-----------------------------+-----------------------------+
               |                                                           |
               v                                                           v
  [Feature path] train.py                                 [Raw CNN path] train_cnn.py
   SVM / RF / XGBoost                                      1D CNN on raw windows
```

---

## 4. Stage 1 -- Survey Label Classification

**Input:** Raw Qualtrics CSV exports from both sessions. The first two rows of each file are Qualtrics metadata; the pipeline skips them and reads actual responses from the third row onward.

**Label Assignment:** Each response contains text answers for Opportunity and Pressure. These are mapped to numeric scores (1-6). A threshold rule assigns the stress label:
- Score >= 4 is HIGH; score <= 3 is LOW
- Opportunity HIGH + Pressure LOW --> `eustress`
- Opportunity LOW + Pressure HIGH --> `distress`
- Both LOW --> `relax`
- Both HIGH --> `eustress+distress`

**Output:** A labeled CSV per session with survey start/end timestamps, numeric scores, and assigned labels. These feed into Stage 2.

---

## 5. Stage 2 -- Label Merging and Time Alignment

This stage propagates survey labels onto every row of the MATLAB feature matrix.

### 5a. Time Synchronization

Survey timestamps are absolute wall-clock times. Feature block timestamps (`Block_Start_s`) are seconds elapsed since the device powered on. The pipeline designates the first survey's start time as `t = 0` and converts survey timestamps to seconds relative to that origin, then aligns them with the feature block timeline.

### 5b. Label Propagation

Once the timelines are aligned, labels are assigned using a forward-fill strategy:
- Feature windows that fall within a survey's [start, end] interval receive that survey's label.
- Feature windows in gaps between surveys receive the next survey's label (forward-fill).
- Feature windows after the last survey are labeled "Unknown" and excluded from training.

---

## 6. Stage 3 -- Dataset Assembly

Sessions 1 and 2 are merged into a single CSV. Session 2 Block_IDs are offset by `max(session_1_block_id) + 1` so they do not collide with session 1 IDs. Rows labeled "Unknown" are dropped. This produces `data/features/features_labeled_new.csv`: 1,288 rows with 38 features and 4 class labels.

**Class distribution:**

| Label | Count | % |
|---|---:|---:|
| relax | 468 | 36.3% |
| eustress+distress | 358 | 27.8% |
| distress | 286 | 22.2% |
| eustress | 176 | 13.7% |

---

## 7. Feature Set

### Feature-based pipeline (38 features)

Each feature row represents one ~5 s sub-window. Two sub-windows are extracted per ON burst (`SubWindow_ID` = 1 or 2). The 38 features span PPG amplitudes and their statistics across all three LED channels, derived cross-channel differences and ratios, heart rate and HRV from LED1, skin temperature, and accelerometer standard deviations.

Key feature groups:

| Group | Description |
|---|---|
| PA1 / PA2 / PA3 | Peak amplitude per LED channel with STD, RMSSD, CV, mean |
| HR / SDNN / RMSSD | Heart rate and HRV from LED1 |
| Cross-channel diffs | LED1-LED2, LED3-LED1, LED3-LED2 mean differences and absolute differences |
| PA ratios | PA1/PA3, PA2/PA3, PA1/PA2 |
| Temperature | Mean skin temperature |
| Accelerometer | AccX_STD, AccY_STD, AccZ_STD, Acc_Total_STD |

### Raw time-series pipeline (9 channels)

The CNN uses 112 raw samples per window (~4.5 s at 25 Hz) across 9 channels: LED1, LED2, LED3, Temp, AccX, AccY, AccZ, and two derived channels (LED3-LED1, LED3-LED2) computed on the fly. No MATLAB processing is required.

---

## 8. Preprocessing

**Feature-based models:** Infinite values are replaced with NaN, then imputed with the column median. A `StandardScaler` is applied for the SVM only; RF and XGBoost are scale-invariant and skip this step. The scaler is fit only on the training fold and applied to the test fold to prevent leakage.

**Raw CNN:** Per-channel z-normalization (subtract channel mean, divide by channel std) computed over all training windows and timesteps. The mean and std arrays are fit on the training fold and applied to the test fold. The final scaler (fit on the full dataset) is saved as `results/cnn_raw_baseline/cnn_1d_scaler.npz` for deployment.

---

## 9. Cross-Validation Strategy

Both pipelines use **5-fold Stratified Group K-Fold**. The grouping key is `Block_ID` (feature pipeline) or the matched Block_ID (CNN pipeline). All sub-windows from the same ON burst are always assigned to the same fold, preventing train/test leakage from correlated samples. Stratification ensures each fold preserves the 36/28/22/14 class ratio.

---

## 10. Models

### Feature-based models

**Support Vector Machine (SVM):** RBF kernel, class_weight="balanced". Requires feature scaling. Hyperparameters tuned via `RandomizedSearchCV` on the 4-class data: `C=6.552`, `gamma=0.3003`.

**Random Forest (RF):** Ensemble of independently grown decision trees. class_weight="balanced". Scale-invariant. Key hyperparameters: `n_estimators=406`, `max_depth=15`.

**XGBoost:** Gradient-boosted tree ensemble. Per-sample class weights via `compute_sample_weight("balanced")` (XGBoost has no native `class_weight` parameter). Key hyperparameters: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

### 1D CNN on raw time-series

The CNN operates directly on raw sensor windows -- no MATLAB feature extraction required. It uses temporal convolutional filters tuned to the 25 Hz sampling rate:

```
Input (112, 9)
  Conv1D(32, k=7, same) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)    [k=7 ~ 280ms]
  Conv1D(64, k=5, same) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)    [k=5 ~ 200ms]
  Conv1D(128, k=3, same) -> BN -> ReLU -> GlobalAvgPool -> Dropout(0.25) [k=3 ~ 120ms]
  Dense(64, relu) -> Dropout(0.30)
  Dense(4, softmax)
```

MaxPooling halves the temporal dimension twice (112 -> 56 -> 28) before GlobalAvgPool collapses it. Kernel sizes are chosen so that the first layer's receptive field spans one full PPG cycle (~280ms at 70 bpm). Class weights are computed via `compute_class_weight("balanced")`. Training: Adam (lr=1e-3), EarlyStopping (patience=20), up to 200 epochs.

**Why the raw CNN outperforms a feature-based CNN:** An earlier experiment treated the 38 engineered features as a 1D temporal sequence and applied Conv1D filters. This produced only 44.9% F1 macro because adjacent features have no spatial or temporal relationship -- Conv1D is architecturally unsound on unordered tabular vectors. The raw CNN operates on actual physiological time-series where temporal locality is meaningful, and achieves 80.1% F1 macro.

---

## 11. Results

### Unfiltered dataset (5-fold StratifiedGroupKFold, 1,288 rows)

| Model | Input | Accuracy | F1 Macro | Export |
|---|---|:---:|:---:|---|
| **CNN (1D, raw)** | Raw 112x9 windows | **80.6%** +/- 2.4% | **0.801** +/- 0.023 | TFLite |
| XGBoost | 38 features | 78.6% +/- 1.8% | 0.766 +/- 0.018 | ONNX |
| Random Forest | 38 features | 74.5% +/- 2.8% | 0.700 +/- 0.033 | ONNX |
| SVM | 38 features | 59.1% +/- 2.7% | 0.512 +/- 0.023 | ONNX |

### Motion-filtered dataset (5-fold StratifiedGroupKFold, 931 rows)

| Model | Input | Accuracy | F1 Macro | Export |
|---|---|:---:|:---:|---|
| **CNN (1D, raw MF)** | Raw 112x9 windows | **81.0%** +/- 3.6% | **0.758** +/- 0.060 | TFLite |
| XGBoost MF | 38 features | 79.4% +/- 4.0% | 0.705 +/- 0.028 | ONNX |

Motion filtering removes most distress blocks (4.3% vs 22.2% unfiltered), which lowers macro F1 despite similar accuracy. The CNN and XGBoost rankings hold on both datasets.

### Per-class performance -- CNN_1D unfiltered (best deployment model)

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| eustress+distress | 0.90 | 0.86 | 0.88 | 358 |
| distress | 0.71 | 0.90 | 0.80 | 286 |
| relax | 0.85 | 0.71 | 0.78 | 468 |
| eustress | 0.72 | 0.79 | 0.75 | 176 |

### Per-class performance -- CNN_1D motion-filtered

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| eustress+distress | 0.89 | 0.87 | 0.88 | 269 |
| eustress | 0.81 | 0.79 | 0.80 | 266 |
| relax | 0.81 | 0.80 | 0.80 | 322 |
| distress | 0.47 | 0.69 | 0.56 | 39 |

### RF feature importances (top 5)

| Rank | Feature | Importance |
|---|---|:---:|
| 1 | PA3 (LED3 peak amplitude) | highest |
| 2-3 | Cross-channel PA differences | high |
| 4-5 | AccX_STD / Acc_Total_STD | moderate |

Temperature and PPG amplitude remain dominant predictors, consistent with their physiological relevance to stress-induced vasoconstriction.

---

## 12. Limitations

1. **Small dataset.** 1,288 sub-windows across two sessions and one participant. Cross-validation variance (+/- 0.02-0.03 on F1 macro) is meaningful. No generalizability claim can be made to other individuals or environments.

2. **Single participant.** A model trained on one person's physiology may not transfer to others. More participants and sessions are needed before deployment in a general population.

3. **Eustress underrepresentation.** Eustress accounts for only 13.7% of the dataset. Despite balanced class weighting, this limits the model's ability to learn the eustress signal reliably.

4. **Label quality at survey boundaries.** Labels are forward-filled from survey timestamps -- a feature window labeled "eustress" was recorded before the survey that reported eustress, not during an objectively confirmed eustress state.

5. **CNN is float32 only (for now).** The TFLite deployment artifact is unquantized float32. INT8 post-training quantization is possible with a representative calibration dataset and would reduce model size by ~4x for MCU-constrained deployment.

---

## 13. How to Run

**Step 1 -- Classify EMA surveys:**
```bash
python -m src.data.classify_survey --input_dir data/survey/ --output_dir data/survey/
```

**Step 2 -- Merge labels onto feature rows (run once per session):**
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

**Step 3 -- Combine sessions (Python snippet):**
```python
import pandas as pd
s1 = pd.read_csv("data/features/features_labeled_session1.csv")
s2 = pd.read_csv("data/features/features_labeled_session2.csv")
s2["Block_ID"] += s1["Block_ID"].max() + 1
combined = pd.concat([s1, s2], ignore_index=True)
combined[combined["Label"] != "Unknown"].to_csv("data/features/features_labeled_new.csv", index=False)
```

**Step 4a -- Train feature-based models:**
```bash
# Unfiltered
python train.py --data data/features/features_labeled.csv
python train.py --data data/features/features_labeled.csv --models xgb --output_dir results/xgb_baseline

# Motion-filtered
python train.py --data data/features/features_labeled_mf.csv --models xgb --output_dir results/xgb_mf_baseline
```

**Step 4b -- Train raw CNN:**
```bash
python train_cnn.py          # unfiltered
python train_cnn.py --mf     # motion-filtered
```

**Tune hyperparameters (optional):**
```bash
python tune.py --data data/features/features_labeled.csv --models svm --n-iter 40
```

**Run tests:**
```bash
pytest tests/ -v
```

**Dependencies:** `pip install -r requirements.txt`

---

## 14. Summary

The pipeline classifies four momentary stress states (relax, eustress, distress, eustress+distress) from M3X ring sensor data across two recording sessions, on both unfiltered and motion-filtered data variants. The **1D CNN on raw time-series** is the best model on the unfiltered dataset (80.6% accuracy, 0.801 macro F1), achieving this by operating directly on 112-sample x 9-channel sensor windows rather than on engineered features. On the motion-filtered dataset, the CNN achieves 81.0% accuracy but 0.758 macro F1, reflecting the near-elimination of distress samples (4.3% of blocks). **XGBoost** is the best feature-based model on both datasets. **Random Forest** and **SVM** models are available for the unfiltered data; all models are exported to ONNX or TFLite for portable deployment.

The most important engineering decisions in this pipeline are: (1) treating raw sensor time-series as the CNN input instead of engineered features, which raised F1 from 44.9% to 80.1%; (2) grouped cross-validation by Block_ID to prevent leakage from correlated sub-windows; and (3) per-channel z-normalization fit on the training fold only to prevent scaler leakage.

The primary path to improving all models is more data: additional participants, more sessions, and balanced representation across all four stress quadrants.
