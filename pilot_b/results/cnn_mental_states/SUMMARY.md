# Pilot B -- Mental-States Labeling Experiment (one-off)

**This is an experiment and does not change how labeling works for any other model in this project.**

Participants in scope: Shuail (501), Lehar (502), Emma (503), Aisha (504) -- Sophia and Jiang excluded.

Labels come from the survey's `Mental states` column directly (one class per distinct value), **not** from the Opportunity/Pressure scheme used everywhere else -- Opportunity and Pressure were not used at all.

## Known limitation, checked before training

Shuail has zero non-blank `Mental states` responses across all of his survey data, so he contributes **no windows** despite being nominally in scope. This run is effectively Lehar + Emma + Aisha. There are only 2 distinct non-blank values in the data, so this is a **binary** classification task, not multi-class.

## Dataset

- Full labeled dataset: **3698** windows
- Classes: ['Cognitive strain, including cognitive overload or distraction', 'Mental capacity, including recovery, attention, and cognitive processing']
- Class counts (full dataset): {'Cognitive strain, including cognitive overload or distraction': 2290, 'Mental capacity, including recovery, attention, and cognitive processing': 1408}
- Participants contributing data: ['Lehar', 'Emma', 'Aisha']
- After per-participant balancing (training pool): **2796** windows, class counts {'Cognitive strain, including cognitive overload or distraction': 1718, 'Mental capacity, including recovery, attention, and cognitive processing': 1078}

## Results (5-fold grouped cross-validation)

- Accuracy    : 0.8916 +/- 0.0202
- F1 Macro    : 0.8850 +/- 0.0215
- F1 Weighted : 0.8913 +/- 0.0203

See `general_validation_report.txt` for the full per-class report and confusion matrix.

## Model files

- `cnn_1d_model.tflite` -- float32 (185.4 KB)
- `cnn_1d_model_int8.tflite` -- INT8 (60.8 KB), estimated RAM ~7.0 KB
- `cnn_1d_scaler.npz` -- per-channel mean/std used by the final model