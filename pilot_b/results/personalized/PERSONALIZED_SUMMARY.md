# Pilot B -- Personalized CNNs (one model per person)

Each model is trained only on that participant's own data -- no cross-participant pooling or balancing. Same architecture and windowing as the generalized model (`results/pilot_b_cnn/`), just fit to one person. Validation is a per-participant 5-fold StratifiedGroupKFold; the shipped model is then retrained on 100% of that person's data (no held-out slice, since there's no cross-participant generalization question for a personalized model).

Aisha's and Sophia's raw files include an extra humidity sensor column the other three participants' devices don't have. Per request, their personalized models use it as a 10th input channel. Everyone else's model uses the standard 9 channels.

**Update (2026-08-03):** Aisha's data was replaced with a newer export that supersedes the original one entirely -- her day1 and day2 raw recordings are unchanged (same sensor content, just a corrected timestamp format), but a third day (day4, July 29) was added. Retrained her model on this updated 3-day dataset. One data-pipeline issue surfaced and was fixed along the way: the shared survey snapshot file (bundled in Emma's zip) predates Aisha's July 29 session, so it had no responses for that day at all -- her day4 windows were silently getting zero labels. Fixed by pulling day4's labels directly from Aisha's own `Aisha_survey_day4.xlsx` instead of the shared snapshot.

**Update (2026-08-06):** Sophia (505), the cohort's fifth and final participant, was added -- 4 days, humidity sensor like Aisha's. The shared survey snapshot predates her entire dataset (all 4 of her days postdate it), so every one of her days uses her own per-day survey xlsx directly, same fix pattern as Aisha's day4.

## Overview

| Participant | Windows | Channels | Classes present | CV accuracy | CV F1 macro | INT8 size | Est. RAM |
|---|---:|---:|---|---:|---:|---:|---:|
| Shuail | 1,482 | 9 | all 4 | 0.824 | 0.805 | 61.0 KB | ~7.0 KB |
| Lehar | 1,117 | 9 | eustress, eustress+distress, relax (no distress) | 0.851 | 0.825 | 61.0 KB | ~7.0 KB |
| Emma | 1,729 | 9 | all 4 | 0.924 | 0.699 | 61.1 KB | ~7.0 KB |
| Aisha | 852 | **10 (incl. Humidity)** | eustress, eustress+distress, relax (no distress) | 0.952 | 0.883 | 61.1 KB | ~7.0 KB |
| Sophia | 1,284 | **10 (incl. Humidity)** | eustress, eustress+distress, relax (no distress) | 0.942 | **0.918** | 61.1 KB | ~7.0 KB |

Full per-fold results, confusion matrices, and quantization reports are in each participant's own subfolder (`results/pilot_b_personalized/<name>/`).

## Important: models are not interchangeable

Two things differ per participant and will break firmware/inference code if assumed fixed:

1. **Aisha's and Sophia's input shape is (112, 10); Shuail, Lehar, and Emma's is (112, 9).** The 10-channel order is `LED1, LED2, LED3, Temp, Humidity, AccX, AccY, AccZ, LED3_LED1, LED3_LED2` -- Humidity is inserted right after Temp. Don't feed a 10-channel person's windows into a 9-channel person's model or vice versa.

2. **Lehar, Aisha, and Sophia never reported "distress" at all** across their whole dataset, so their models only have a 3-class output (`eustress`, `eustress+distress`, `relax`) instead of 4, and the class order (alphabetical, from sklearn's LabelEncoder) differs from the 4-class models:
   - Shuail / Emma (4 classes): `distress, eustress, eustress+distress, relax`
   - Lehar / Aisha / Sophia (3 classes): `eustress, eustress+distress, relax`

   Always read the target participant's `int8_quantization_report.txt` for their exact class order and quantization params before wiring up inference -- don't assume the class index mapping from one person's model applies to another's.

## Reading the numbers

- **INT8 quantization is lossless in every case** -- float32-vs-INT8 prediction agreement is 99%+ for all five models (100% for Aisha's and Sophia's), essentially identical to the pattern seen on the generalized model.
- **Personalized accuracy is substantially higher than the generalized model's cross-participant CV** (0.70-0.92 F1 macro here vs 0.79 for the pooled 5-person model) -- expected, since there's no cross-participant variability left to generalize across.
- **Emma's F1 macro (0.699) is still the outlier**, and the per-fold numbers swing hard (0.50 to 0.85) -- she has only 10 `eustress` windows total in her whole dataset, so whichever fold happens to land those few examples in the test set is highly sensitive to chance. Accuracy (0.924) looks fine because `eustress` is such a small share of her data that missing it barely dents the overall accuracy number; F1 macro is the more honest read here and it's flagging a real data-scarcity problem, not a model problem.
- **Sophia's model (0.918 F1 macro, 0.942 accuracy) is now the strongest of the five**, edging out Aisha's (0.883). Both have a humidity channel and both lack a distress class, so they're a fairly direct comparison -- Sophia's classes are somewhat less skewed than Aisha's (her `relax` class alone has 192 windows vs Aisha's 21), which likely explains the gap more than the humidity channel itself. As before, we haven't run a controlled with/without-humidity ablation for either of them, so we can't yet say how much of their strong performance (if any) comes specifically from the humidity signal versus just having cleaner class coverage.

## Files per participant (`results/pilot_b_personalized/<name>/`)

```
dataset_summary.csv              per participant-day parsing/labeling diagnostics
fold_results.csv                 per-fold CV accuracy/F1
cv_report.txt                    pooled classification report + confusion matrix
cnn_1d_model.pkl                 final Keras model (100% of their data)
cnn_1d_scaler.npz                per-channel mean/std
cnn_1d_model.tflite              float32 export
cnn_1d_model_int8.tflite         INT8 export
int8_quantization_report.txt     size, RAM estimate, quantization params, class order
```
