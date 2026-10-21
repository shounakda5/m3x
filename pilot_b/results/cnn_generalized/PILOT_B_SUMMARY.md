# Pilot B -- Generalized CNN: Consolidated Results

**Update (2026-08-06):** Sophia (505), the cohort's fifth and final participant, was added -- 4 days, her device has a humidity sensor (like Aisha's) but the generalized model still uses the standard 9-channel schema for uniformity across everyone, so her humidity data isn't used here (it is used in her personalized model, see `results/pilot_b_personalized/PERSONALIZED_SUMMARY.md`). Both this model and the small variant were retrained on all 5 participants, using Aisha's latest (3-day) data. This replaces the previous 4-participant version of both models.

## Dataset composition (per participant-day)

|   participant_id | name   | rel_path                            | survey_date   |   n_bursts_found |   n_bursts_cropped_ringoff |   n_bursts_kept |   n_windows_labeled |   n_windows_unknown_dropped | has_survey   |
|-----------------:|:-------|:------------------------------------|:--------------|-----------------:|---------------------------:|----------------:|--------------------:|----------------------------:|:-------------|
|              501 | Shuail | 2_Shuail_501/1_suhail_501.csv       | 2026-06-01    |              297 |                          0 |             297 |                 576 |                           6 | True         |
|              501 | Shuail | 2_Shuail_501/2_suhail_501_6_2.csv   | 2026-06-02    |              431 |                          0 |             431 |                 677 |                           0 | True         |
|              501 | Shuail | 2_Shuail_501/3_suhail_501_6_3.csv   | 2026-06-03    |              117 |                          0 |             117 |                 229 |                           2 | True         |
|              502 | Lehar  | 3_Lehar_502/Day1_june6/lehar1.csv   | 2026-06-17    |              190 |                          0 |             190 |                 310 |                          61 | True         |
|              502 | Lehar  | 3_Lehar_502/Day2_June24/lehar2.csv  | 2026-06-24    |              505 |                        324 |             181 |                 353 |                           0 | True         |
|              502 | Lehar  | 3_Lehar_502/Day3_July6th/lehar3.csv | 2026-07-06    |              235 |                          0 |             235 |                 454 |                           5 | True         |
|              503 | Emma   | 4_EMMA_503/EMMA/emma_d1.csv         | 2026-07-07    |              174 |                          0 |             174 |                 321 |                          24 | True         |
|              503 | Emma   | 4_EMMA_503/EMMA/emma_d2.csv         | 2026-07-08    |              515 |                        297 |             218 |                 430 |                           0 | True         |
|              503 | Emma   | 4_EMMA_503/EMMA/emma_d4.csv         | 2026-07-10    |              657 |                        385 |             272 |                 523 |                           0 | True         |
|              503 | Emma   | 4_EMMA_503/EMMA/emma_d5.csv         | 2026-07-13    |              670 |                        383 |             287 |                 455 |                           0 | True         |
|              504 | Aisha  | 5_Aisha_504/Aisha_day1.csv          | 2026-07-23    |              242 |                          0 |             242 |                 241 |                           0 | True         |
|              504 | Aisha  | 5_Aisha_504/Aisha_day2.csv          | 2026-07-24    |              356 |                          0 |             356 |                 285 |                          66 | True         |
|              504 | Aisha  | 5_Aisha_504/Aisha_day4.csv          | 2026-07-29    |              362 |                          2 |             360 |                 326 |                          29 | True         |
|              505 | Sophia | 6_Sophia_505/Sophia_day1.csv        | 2026-07-27    |              275 |                          0 |             275 |                 272 |                           2 | True         |
|              505 | Sophia | 6_Sophia_505/Sophia_day2.csv        | 2026-07-29    |              346 |                          0 |             346 |                 344 |                           0 | True         |
|              505 | Sophia | 6_Sophia_505/sophia_day3.csv        | 2026-08-05    |              331 |                          0 |             331 |                 328 |                           0 | True         |
|              505 | Sophia | 6_Sophia_505/sophia_day4.csv        | 2026-08-06    |              346 |                          0 |             346 |                 340 |                           1 | True         |

## Dataset funnel

- Full labeled dataset: **6464** windows
- After removing Aisha's held-out slice: **6294** windows
- After per-participant balancing, equal total budget (final training pool): **4477** windows
- Aisha held-out validation set: **170** windows

## Per-participant sampling detail (equal total budget, natural class mix preserved)

|   participant | participant_name   | label             |   available_groups |   sampled_groups |
|--------------:|:-------------------|:------------------|-------------------:|-----------------:|
|           501 | Shuail             | distress          |                144 |              114 |
|           501 | Shuail             | eustress          |                 90 |               61 |
|           501 | Shuail             | eustress+distress |                295 |              216 |
|           501 | Shuail             | relax             |                226 |              175 |
|           502 | Lehar              | eustress          |                 64 |               64 |
|           502 | Lehar              | eustress+distress |                208 |              208 |
|           502 | Lehar              | relax             |                294 |              294 |
|           503 | Emma               | distress          |                131 |               88 |
|           503 | Emma               | eustress          |                  5 |                5 |
|           503 | Emma               | eustress+distress |                716 |              456 |
|           503 | Emma               | relax             |                 26 |               17 |
|           504 | Aisha              | eustress          |                566 |              465 |
|           504 | Aisha              | eustress+distress |                 99 |               86 |
|           504 | Aisha              | relax             |                 17 |               15 |
|           505 | Sophia             | eustress          |                860 |              397 |
|           505 | Sophia             | eustress+distress |                232 |               94 |
|           505 | Sophia             | relax             |                192 |               75 |

## Thread 1 -- General cross-participant validation (5-fold grouped CV)

- Accuracy    : 0.8106 +/- 0.0168
- F1 Macro    : 0.7853 +/- 0.0165
- F1 Weighted : 0.8132 +/- 0.0154

See `general_validation_report.txt` for the full per-class report and confusion matrix.

## Thread 2 -- Aisha-specific held-out validation

- Accuracy : 0.8765
- F1 Macro : 0.4928

See `aisha_holdout_report.txt` for the full per-class report and confusion matrix.

## INT8 quantization (MCU deployment)

`cnn_1d_model_int8.tflite` -- full-integer INT8 quantization of the final model above, via `quantize_pilot_b_cnn.py`. Same recipe as the original `cnn_raw_baseline` INT8 export (200 calibration windows from the training pool).

| | Flash file size | Aisha-holdout accuracy | Aisha-holdout F1 macro |
|---|---:|---:|---:|
| Float32 (`cnn_1d_model.tflite`) | 185.9 KB | 0.8765 | 0.4928 |
| **INT8** (`cnn_1d_model_int8.tflite`) | **61.0 KB** | 0.8824 | 0.5141 |

No meaningful accuracy loss from quantization (small positive wobble is normal run-to-run noise, not a systematic gain).

**Quantization parameters** (hardcode in firmware):
```
Input  scale=0.04210300  zero_point=10
Output scale=0.00390625  zero_point=-128
```

**Estimated RAM (tensor arena):** ~7.0 KB peak, largest single activation tensor 3.5 KB (37 tensors, 19 ops). Best-effort estimate -- see `int8_quantization_report.txt` for the method/caveats.

## Two deployment sizes -- pick based on your chip's budget

A second, smaller architecture was also retrained on the 5-participant data: 2 conv blocks (16/32 filters) with no Dense hidden layer. Full report in `results/pilot_b_cnn_small/int8_quantization_report.txt`.

| | Full model (`pilot_b_cnn/`) | Small model (`pilot_b_cnn_small/`) |
|---|---:|---:|
| Trainable params | 46,468 | 3,940 |
| INT8 flash size | 61.0 KB | **11.0 KB** |
| Estimated RAM (tensor arena) | ~7.0 KB | **~3.5 KB** |
| General CV accuracy | 0.8106 +/- 0.0168 | 0.7996 +/- 0.0029 |
| General CV F1 macro | 0.7853 +/- 0.0165 | 0.7730 +/- 0.0054 |
| Aisha holdout accuracy (INT8) | 0.8824 | 0.8647 |
| Aisha holdout F1 macro (INT8) | 0.5141 | 0.3287 |

**Takeaway:** the small model still tracks the full model closely on general cross-participant accuracy (within ~1pp), at 1/12th the parameters, but this round the full model pulls meaningfully ahead of the small model specifically on Aisha's holdout F1 macro (0.51 vs 0.33) -- the small model's limited capacity is leaning on the `eustress` majority class more heavily and recovering less of her `eustress+distress` responses (24% recall vs the full model's stronger showing). If Aisha-like generalization matters more than footprint, the full model is the clearer choice this round; if flash/RAM is the binding constraint, the small model still costs little on general accuracy.

## Reference -- original 2-session CNN baseline

- Accuracy 80.6% +/- 2.4%, F1 Macro 0.801 +/- 0.023 (`results/cnn_raw_baseline`)
