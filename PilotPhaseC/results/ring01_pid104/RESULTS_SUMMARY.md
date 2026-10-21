# Pilot Phase C -- Ring_01_PID_104: Results Summary

One participant, two label lineages, each producing its own classification
(`Label`) and regression (`Stress_0to100`) model -- four models total. See
`../../README.md` for the data format and pipeline, and each run's own
`manifest.json` for full reproducibility detail (git commit, exact
hyperparameters, dataset composition).

All four models are **trained, tuned, 5-fold cross-validated, and exported**
(`.pkl` / float32 `.tflite` / INT8 `.tflite` + scaler), in the formats this
project has shared models in previously.

## Dataset

| Lineage | Windows | Label groups (Survey_ID) | Label counts |
|---|---:|---:|---|
| `all_continuous` | 4,528 | 43 | relax 2911, eustress+distress 1567, distress 25, eustress 25 |
| `labeled_last5min` | 583 | 36 | relax 300, eustress+distress 233, distress 25, eustress 25 |

Both lineages are heavily imbalanced: `distress` and `eustress` each come
from only **2 independent survey events** (groups), the smallest amount of
independent evidence a model can learn from for those classes. This is a
hard limit of the current data, not a modeling choice -- more data for
those two classes is the only real fix.

## Classification (predicts `Label`)

5-fold `StratifiedGroupKFold`, grouped by `Survey_ID`:

| Lineage | Accuracy | F1 Macro | F1 Weighted |
|---|---:|---:|---:|
| `all_continuous` | 0.588 ± 0.036 | 0.294 ± 0.022 | 0.599 ± 0.029 |
| `labeled_last5min` | 0.579 ± 0.181 | 0.327 ± 0.095 | 0.564 ± 0.177 |

Both models learn `relax` and `eustress+distress` (the two classes with
enough independent groups to generalize) reasonably, but get essentially
**zero recall on `distress` and `eustress`** in the pooled cross-validation
report -- with only 2 groups per class, most folds see that class only in
train or only in test, never both. This is visible directly in each run's
`cv_report.txt` confusion matrix. Treat per-class numbers for those two
classes as not yet meaningful; the macro F1 above is dragged down by them
accordingly.

## Regression (predicts `Stress_0to100`)

5-fold `GroupKFold` (no stratification -- continuous target), grouped by
`Survey_ID`, metrics in real 0-100 stress units (target is z-scored for
training, inverted before scoring):

| Lineage | MAE | RMSE | R² |
|---|---:|---:|---:|
| `all_continuous` | 9.705 ± 2.040 | 12.209 ± 2.231 | -0.527 ± 0.495 |
| `labeled_last5min` | 10.581 ± 3.392 | 13.234 ± 2.886 | -0.053 ± 0.511 |

Mean per-fold R² is negative for both -- on held-out survey groups, the
model is often no better (sometimes worse) than just predicting the
training mean. This is the expected consequence of the regression target
effectively only taking ~20-25 distinct real values across the whole
dataset (one scalar per survey, replicated across every window in that
survey's group) split across just 36-43 groups: there isn't yet enough
independent signal for a CNN to learn a reliable continuous mapping. The
MAE/RMSE are a more stable read than R² here, but still reflect a model
that has not yet generalized well. More surveys (more distinct stress
values, more groups) is the most direct way to improve this.

## Model artifacts (deployment-ready)

| Lineage / task | Params | Float32 | INT8 | Est. RAM (arena) |
|---|---:|---:|---:|---:|
| `all_continuous` classification | 41,988 | 168.4 KB | 55.8 KB | ~7.0 KB |
| `all_continuous` regression | 28,041 | 114.6 KB | 41.5 KB | ~5.25 KB |
| `labeled_last5min` classification | 28,236 | 115.5 KB | 42.0 KB | ~5.25 KB |
| `labeled_last5min` regression | 46,049 | 184.0 KB | 60.3 KB | ~7.0 KB |

Float32-vs-INT8 agreement/error (own training data, not a generalization
check) was high across all four (classification agreement 0.955-1.000;
regression mean abs. difference 0.14-0.28 stress points) -- quantization
itself is not where these models are losing accuracy.

## Bottom line

The pipeline (preprocessing, grouping, tuning, CV, loss/metric choices,
export) is sound and was verified end-to-end before these numbers were
produced -- see `../../README.md` "Design notes". The limiting factor
right now is **data volume for the minority classes and for regression's
continuous target**, specific to this first single-participant drop:
`distress`/`eustress` each have only 2 independent survey observations,
and the whole dataset spans only 36-43 independent groups total. These
four models are real, exported, and usable, but should be understood as a
**first baseline to retrain against as more of this participant's (or
other participants') data arrives** -- not as a finished, production-grade
accuracy bar. Re-running `train_classifier.py` / `train_regressor.py`
against a future, larger drop is the expected next step.
