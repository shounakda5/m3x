# Pilot Phase C

A new, ongoing phase of the project: unlike the original pipeline (`data/` +
root-level `train.py`/`train_cnn.py`) and the Pilot B cohort (`pilot_b/`),
this phase's incoming data arrives **already labeled**. There is no
survey-matching or burst-labeling step -- the pipeline here is just:

```
per-participant raw Excel files  -->  windowed dataset (cached)  -->  tune  -->  cross-validate  -->  final model  -->  export
```

Everything in this directory is self-contained -- its own data, its own
code (no imports from the root project's `config.py`, `src/`, or
`train_cnn.py`), its own results. New participants/data drops are expected
to keep arriving over the coming weeks, each producing their own model(s).

## Layout

```
PilotPhaseC/
├── config.py                  # paths, participant/lineage registry, column layout,
│                               #   label taxonomy, CNN hyperparameters, tuning grid
├── data/
│   ├── Ring_01_PID_104/        # one participant's raw drop, exactly as received
│   │   ├── 1_All_continous/*.xlsx        # lineage 1 (see "Data format" below)
│   │   └── 2_labeled_last5min/*.xlsx     # lineage 2
│   └── dataset/
│       └── <participant>/<lineage>.npz   # windowed-dataset cache (see src/loader.py)
├── src/
│   ├── loader.py                # parse a lineage's Excel files into (X, y_label, y_stress, groups)
│   ├── models.py                 # CNNClassifier / CNNRegressor (self-contained)
│   ├── common.py                 # scalers, TFLite export, INT8 quantization
│   ├── tuning.py                 # lightweight hyperparameter search
│   ├── train_classifier.py       # tune -> CV -> final model -> export, for one lineage
│   └── train_regressor.py        # same, for the regression target
└── results/
    └── <participant>/<lineage>/{classification,regression}/
        ├── cnn_1d_model.pkl / .tflite / _int8.tflite
        ├── cnn_1d_scaler.npz      # input channel scaler (+ target scaler for regression)
        ├── fold_results.csv, cv_report.txt, int8_quantization_report.txt
        └── manifest.json          # full lineage: data, git commit, hyperparams, metrics
```

## Data format (Ring_01_PID_104, first drop, 2026-10-21)

One participant (PID 104), raw ring-sensor Excel exports, pre-labeled, in
**two lineages** -- each lineage is a separate dataset and produces its own
bespoke model (the labels mean different things in each):

- **`1_All_continous`** -- every window across the full continuous
  recording is labeled, mostly by carrying a survey's answer forward across
  the whole gap until the next survey. Larger (4,528 windows / 43 label
  groups across the first drop) but noisier -- long `relax` stretches
  dominate.
- **`2_labeled_last5min`** -- only the ~5 minutes immediately preceding
  each survey are labeled. Smaller (583 windows / 36 groups) but more
  precisely tied to the actual survey response.

Each `.xlsx` file has one row per sensor sample, 112 rows per window (one
row = one sample within a window; columns beyond what's listed below are
ignored):

| Column | Name | Role |
|---|---|---|
| G | `Row_In_Window` | 0..111, cycles -- sorts rows within a window |
| D | `Global_Window` | segments the row stream into 112-row windows (used instead of re-deriving window boundaries from Row_In_Window's reset, since it's already provided and verified 1:1 with it) |
| I-P | `LED1, LED2, LED3, TempC_x100, HumidityRH_x100, AccX_mg, AccY_mg, AccZ_mg` | the 8 feature channels, fed to the model as-is (no derived cross-LED channels added -- see "Design notes" below) |
| R | `Survey_ID` | **not in the original column spec, but required**: groups every window from the same labeling interval together. Used as the cross-validation group key so temporally-adjacent, highly-correlated windows can't split across train/test. Verified globally unique across every file in a lineage. |
| W | `Stress_0to100` | regression target |
| X | `Label` | classification target (`relax`, `eustress`, `distress`, `eustress+distress`) |

`src/loader.py` validates every window has exactly 112 rows and an
unambiguous single label/stress value, dropping (with a loud warning,
never silently) any that don't.

## Adding a new participant / data drop

Drop the raw files under `data/<Ring_ID>_<PID>/<lineage>/*.xlsx`, matching
the layout above, and register the participant in `config.py`'s
`PARTICIPANTS` dict (directory + lineage subfolder names). If a new
lineage's column layout differs from what's documented above, update
`config.py`'s column constants and `REQUIRED_COLUMNS` in `src/loader.py`
accordingly -- don't assume it's identical without checking.

## Training a model

```bash
# Classification (predicts Label)
python -m PilotPhaseC.src.train_classifier --lineage all_continuous
python -m PilotPhaseC.src.train_classifier --lineage labeled_last5min

# Regression (predicts Stress_0to100)
python -m PilotPhaseC.src.train_regressor --lineage all_continuous
python -m PilotPhaseC.src.train_regressor --lineage labeled_last5min
```

Each run: loads the lineage (cached after the first parse), runs a small
hyperparameter search (`src/tuning.py`, one group-aware validation split,
reduced epoch budget -- not the full CV, to keep the search cheap), then
runs the winning config through 5-fold cross-validation (`StratifiedGroupKFold`
for classification, `GroupKFold` for regression; both grouped by
`Survey_ID`), then retrains on 100% of the data and exports `.pkl` +
`.tflite` (float32) + `.tflite` (INT8, with a quantization/tensor-arena
report) + the fitted scaler(s). `--no-tune` skips the search and uses
`config.CNN_PARAMS` defaults directly. A `manifest.json` ties every result
back to its exact dataset, git commit, hyperparameters, and metrics.

## Design notes / things worth knowing before extending this

- **Why `Survey_ID` and not `Global_Window`**: `Global_Window` only
  numbers windows within one file and resets per file; it can't safely
  group across a whole lineage's multiple day-files. `Survey_ID` already
  encodes the real survey event and is verified unique across files, so
  it's the correct grouping key for leakage-safe CV.
- **Derived channels not added**: the rest of this project (`src/models/cnn_model.py`,
  `pilot_b/`) adds two derived cross-LED channels (`LED3-LED1`, `LED3-LED2`)
  on top of the raw LED channels. This phase's column spec was explicit
  (columns I-P, 8 raw channels) and that was honored literally rather than
  assumed -- adding the same derived channels here would be a quick,
  likely-beneficial follow-up if wanted.
- **Regression target scaling**: the model trains against a z-scored
  `Stress_0to100` (fit on the train fold only, exactly like the input
  channel scaler) and predicts in that scaled space. Every reported metric
  and the `.pkl`/float32 `.tflite` model's `predict()` wrapper inverts back
  to real 0-100 units; the raw INT8 TFLite model's output is still in
  z-scored space -- invert with the `target_mean`/`target_std` stored in
  that run's `cnn_1d_scaler.npz` / `manifest.json`.
- **Single-participant data**: unlike the root pipeline (multi-session) or
  Pilot B (multi-participant with a held-out generalization thread), this
  is one person's data -- there's no cross-participant generalization
  question here, only within-person temporal generalization (hence the
  `Survey_ID` grouping, which is still essential even for one person).
