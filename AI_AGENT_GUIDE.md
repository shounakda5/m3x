# AI Agent Guide — M3X Stress Classification Project

**Read this first if you are an AI assistant picking up this repo.** It is
written so you can orient yourself and start making correct changes without
re-deriving the project's structure, conventions, or history from scratch.
It is kept in sync with the actual code — if something here looks stale
relative to the files on disk, trust the files and update this doc.

This doc assumes no prior context. Human-facing docs (`README.md`,
`INFERENCE_GUIDE.md`, etc.) are still the place for deployment detail and
narrative write-ups; this doc is the map that tells you which of those to
read for what, plus everything that isn't written down anywhere else.

---

## 1. What this project is

A machine-learning pipeline that classifies momentary stress states (and,
as of Pilot Phase C, also regresses a continuous stress score) from M3X
ring sensor physiological data (PPG/LED channels, temperature, humidity on
some devices, 3-axis accelerometer), collected as part of a workplace
lighting EMA (Ecological Momentary Assessment) study at USC. Participants
wear the ring continuously; periodically they answer a short survey about
their current state, which becomes the ground-truth label for the sensor
data around that time.

The 4-class stress taxonomy used throughout (derived from Opportunity ×
Pressure, both 1-6 Likert, thresholded at ≤3=Low / ≥4=High) — **this exact
list and order is reused in every phase of the project**:

```
["relax", "eustress", "distress", "eustress+distress"]
```

| Label | Opportunity | Pressure |
|---|---|---|
| `relax` | Low | Low |
| `eustress` | High | Low |
| `distress` | Low | High |
| `eustress+distress` | High | High |

## 2. Repo-wide map: three independent phases

The repo has grown in three successive phases, each with its own data and
code, **not meant to be merged**. Each is independently runnable.

| Phase | Directory | Data source | Status |
|---|---|---|---|
| **Original pipeline** | repo root (`config.py`, `train.py`, `train_cnn.py`, `tune.py`, `src/`) | 2 recording sessions, 1 participant, MATLAB-extracted features + raw sensor CSVs, needs full survey-labeling | Done, baseline models trained |
| **Pilot B** | `pilot_b/` | 5 participants, raw sensor CSVs + Qualtrics survey exports, needs full survey-labeling (burst detection, survey-interval matching) | Done, generalized + personalized models trained |
| **Pilot Phase C** | `PilotPhaseC/` | New participants arriving week over week, **pre-labeled** Excel exports (no survey-matching needed) | Ongoing — 1 participant (`ring01_pid104`) done, structure built to take more |

**Critical rule for Pilot Phase C**: it is explicitly self-contained —
nothing under `PilotPhaseC/` imports from the root `config.py`, root
`src/`, or `train_cnn.py`, and nothing outside `PilotPhaseC/` should import
from it. If you're tempted to reuse a root-level helper from inside
`PilotPhaseC/`, don't — re-implement it locally there instead (this is
intentional duplication, not an oversight). The original pipeline and
`pilot_b/` *do* share code with each other (`pilot_b/` imports
`src/models/cnn_model.py` and helpers from root `train_cnn.py`) — that
reuse is fine and expected; it's only `PilotPhaseC/` that's walled off.

## 3. Conventions shared across all three phases

These habits recur everywhere in this repo. If you're adding a new model
or dataset anywhere in this project, match them:

- **Scaler discipline**: any scaler (per-channel input z-score, target
  z-score, StandardScaler for tabular features) is fit on the **training
  fold only**, then applied to validation/test. Fitting on the full
  dataset before splitting leaks test statistics and inflates reported
  metrics. Every training script does this explicitly inside the CV loop.
- **Leakage-safe grouping**: cross-validation is always `StratifiedGroupKFold`
  (classification) or `GroupKFold` (regression), grouped by whatever key
  identifies "things that are temporally/physiologically correlated and
  must not be split across train/test" — `Block_ID` (original pipeline),
  burst-derived group IDs (Pilot B), `Survey_ID` (Pilot Phase C). Plain
  k-fold on raw windows is never used for raw time-series data in this
  project, because adjacent windows of the same recording are highly
  correlated.
- **CNN architecture family**: every raw-time-series model in this repo
  (original pipeline's `CNN_1D`, Pilot B's, Pilot Phase C's) is the same
  3-block-conv → global-average-pool → dense-head family, just with
  different hyperparameters/head. See §5 for the exact layer spec.
- **Export convention for CNN models** ("the formats we've been sharing
  models in"): `.pkl` (joblib-dumped Keras model) + `.tflite` (float32) +
  `.tflite` INT8-quantized (with a text quantization report including a
  best-effort tensor-arena/RAM estimate) + a `.npz` scaler file. Tabular
  models (SVM/RF/XGBoost, original pipeline only) export to ONNX instead.
- **INT8 tensor-arena estimation method**: a greedy/linear memory planner
  simulated by parsing the `.tflite` flatbuffer directly (same idea
  TFLite Micro's allocator uses) — tracks each tensor's first/last use
  across the op execution order and finds the peak simultaneously-live
  byte count. This is a **lower-bound estimate**, not a hardware
  measurement — every quantization report says so; don't present it as
  exact.
- **Manifest/traceability habit**: results directories carry either a
  written report (older phases) or a `manifest.json` (Pilot Phase C) that
  ties a trained model back to its exact data, git commit, hyperparameters,
  and metrics.
- **Commit style**: short imperative present-tense subject line ("Add X",
  "Restructure Y"), longer body explaining why, `Co-Authored-By:` trailer
  for AI-assisted commits. Destructive git operations (`--force`,
  `rebase -i`, etc.) are avoided; renames use `git mv` to preserve history.

## 4. Original pipeline (repo root)

**Read `README.md` and `IMPLEMENTATION_REPORT.md` for the full narrative.**
Summary for orientation:

- **Data**: 2 sessions (June 1-2), one participant, `data/raw/*.csv`
  (raw sensor) + `data/features/*.xlsx` (MATLAB-extracted 38 features) +
  `data/survey/*.csv` (Qualtrics exports). Two dataset variants: unfiltered
  and motion-filtered (`_mf` suffix, MATLAB removed high-motion bursts).
- **Labeling pipeline** (already run, outputs committed): `src/data/classify_survey.py`
  (Qualtrics → labeled survey CSV) → `src/data/merge_labels.py` (attach
  labels to feature rows by timestamp) → manual session-combine step
  (see `README.md` §6) → `data/features/features_labeled[_mf].csv`.
- **Two parallel training pipelines**:
  - `train.py` — SVM/RF/XGBoost on the 38 MATLAB features (`src/data/preprocess.py`
    loads/cleans, `src/evaluation/cross_val.py` runs `StratifiedGroupKFold`
    grouped by `Block_ID`, `src/models/{svm,rf,xgb}_model.py` implement
    `BaseClassifier`). Exports ONNX.
  - `train_cnn.py` — 1D CNN (`src/models/cnn_model.py`) directly on raw
    112-sample × 9-channel windows (7 raw channels + `LED3-LED1`/`LED3-LED2`
    derived). `src/data/raw_preprocess.py` does burst detection (gaps in
    `Time_PPG` > 1s) and windowing. Exports TFLite.
- `tune.py` — `RandomizedSearchCV` hyperparameter tuning for the tabular
  models only (the CNN's hyperparameters were tuned manually; see
  `CNN_ARCH_TUNING_GUIDE.md`).
- `config.py` (root) holds constants **shared with `pilot_b/`**: paths,
  `CLASSES`, `N_FOLDS`, `FEATURE_COLS`, all `RAW_*` raw-windowing constants
  (`RAW_SAMPLING_HZ=25`, `RAW_WINDOW_SAMPLES=112`, `RAW_CHANNELS` = the 9
  channels, `RAW_GAP_THRESHOLD_MS=1000`, `RAW_MIN_BURST_S=8.0`,
  `RAW_N_WINDOWS_PER_BURST=2`), `SVM_PARAMS`/`RF_PARAMS`/`XGB_PARAMS`,
  `CNN_PARAMS`. **Do not add Pilot-B-only or Pilot-Phase-C-only constants
  here** — they each have their own `config.py` now (see §5, §6).
- Best model: CNN on raw unfiltered data, 80.6% accuracy / 0.801 F1 macro
  (5-fold CV). Full results table in `README.md` §7.

## 5. Pilot B (`pilot_b/`)

**Read `pilot_b/results/*/`'s `.md` summaries for numbers.** This cohort's
data and code were restructured into a self-contained-looking subtree on
2026-10-21 (originally scattered across repo root + `data/pliot_b/` with a
typo) — if you ever see a reference to `data/pliot_b/`, `train_pilot_b_cnn.py`,
or similar at repo root in old context/history, that's stale; the current
layout is below.

```
pilot_b/
├── config.py            # PILOT_B_* constants only (flatline/holdout/
│                         #   random-state knobs). Shared RAW_* constants
│                         #   are imported from the ROOT config.py — this
│                         #   is the one place in the repo that
│                         #   deliberately crosses between a root config
│                         #   and a subphase config; it's fine here because
│                         #   pilot_b is NOT held to PilotPhaseC's
│                         #   self-containment rule.
├── data/
│   ├── raw/              # 5 participants' zips + extracted/ CSVs/xlsx +
│   │                     #   review/ (Sophia's reviewed day5 data)
├── src/
│   ├── loader.py          # raw CSV parsing (boot-log/garbled-byte
│   │                      #   tolerant), burst detection, ring-wear
│   │                      #   flatline cropping, optional slow-channel
│   │                      #   detrending, windowing, survey-interval
│   │                      #   labeling. Public API:
│   │                      #   load_pilot_b_dataset() (pooled) and
│   │                      #   load_pilot_b_participant_dataset() (one person)
│   ├── train_generalized.py        # cross-participant model + Aisha holdout thread
│   ├── train_generalized_small.py  # smaller 2-block arch for tight MCUs
│   ├── train_personalized.py       # one model per participant, no pooling
│   ├── quantize.py                 # INT8 quantization + tensor-arena estimate
│   └── experiments/
│       └── train_mental_states.py  # ONE-OFF: labels from a different survey
│                                    #   column, not the standard scheme
└── results/
    ├── cnn_generalized/, cnn_small/, cnn_mental_states/
    └── personalized/, personalized_detrend/, personalized_mental_states/,
        personalized_small*/        # per-participant subdirs inside each
```

Key facts: 5 participants (Shuail/501, Lehar/502, Emma/503, Aisha/504,
Sophia/505), raw-only (no MATLAB features). Aisha and Sophia's devices have
a humidity sensor (10-channel `HUMIDITY_CHANNELS` input for their
personalized models; everyone else/the generalized model uses the standard
9-channel `RAW_CHANNELS`). The generalized model holds out a slice of
Aisha's data entirely from training as a held-out generalization check
(`split_aisha_holdout` in `train_generalized.py`). Some personalized models
output fewer than 4 classes because that participant never reported every
class — always check that model's own `int8_quantization_report.txt` for
its actual class list/order before using it.

Run commands all follow `python -m pilot_b.src.<module>` from repo root
(each script inserts repo root onto `sys.path` itself, so this also works
as a plain `python pilot_b/src/<module>.py` invocation).

## 6. Pilot Phase C (`PilotPhaseC/`) — full detail

This is the newest, actively-growing phase. **Everything below is accurate
as of the `ring01_pid104` first data drop (2026-10-21/22).**

### 6.1 Why this phase is different

Unlike the other two phases, incoming data here arrives **already
labeled** — there's no survey-matching/burst-labeling step. The pipeline
is: parse per-participant Excel files → window → tune → cross-validate →
train final model → export. New participants are expected to keep
arriving over the coming weeks; each is registered in `config.py` and gets
its own model(s).

### 6.2 Directory layout (current)

```
PilotPhaseC/
├── __init__.py
├── config.py                  # see §6.4
├── README.md                  # human-facing version of this section
├── data/
│   ├── Ring_01_PID_104/        # participant "ring01_pid104"'s raw drop
│   │   ├── 1_All_continous/*.xlsx       # lineage "all_continuous" (7 files:
│   │   │                                #   baseline + day1,3,4,7,8,9)
│   │   └── 2_labeled_last5min/*.xlsx    # lineage "labeled_last5min" (same 7 days)
│   ├── incoming/.gitkeep       # reserved, unused by this first drop (see §6.9)
│   └── dataset/
│       └── ring01_pid104/
│           ├── all_continuous.npz       # windowed-dataset cache (GITIGNORED —
│           └── labeled_last5min.npz     #   regenerate via src/loader.py, seconds)
├── src/
│   ├── __init__.py
│   ├── loader.py               # Excel -> windows (see §6.5)
│   ├── models.py                # CNNClassifier / CNNRegressor (see §6.6)
│   ├── common.py                # scalers, TFLite export, INT8 quant (see §6.7)
│   ├── tuning.py                # hyperparameter search (see §6.8)
│   ├── train_classifier.py      # full pipeline, classification
│   └── train_regressor.py       # full pipeline, regression
├── tests/__init__.py            # empty, reserved
└── results/
    └── ring01_pid104/
        ├── RESULTS_SUMMARY.md   # human-readable consolidated results + caveats
        ├── all_continuous/{classification,regression}/
        └── labeled_last5min/{classification,regression}/
            ├── cnn_1d_model.pkl, .tflite, _int8.tflite
            ├── cnn_1d_scaler.npz
            ├── fold_results.csv, cv_report.txt, int8_quantization_report.txt
            └── manifest.json
```

A `PilotPhaseC/results.zip` also exists at the `PilotPhaseC/` root — a
manual zip of the `results/` tree someone made for sharing/download. It's
redundant with the tracked `results/` directory; harmless to keep, safe to
delete, not auto-regenerated by anything.

### 6.3 Data format (per lineage, per participant)

Each `.xlsx` has one row per sensor sample. One **window** = 112
consecutive rows (matches the rest of the project's `RAW_WINDOW_SAMPLES`
convention). Columns beyond those below exist in the files but are
unused/ignored.

| Excel col | Column name | Role |
|---|---|---|
| D | `Global_Window` | segments the row stream into 112-row windows — verified 1:1 with `Row_In_Window`'s 0→111 cycle, used instead of re-deriving boundaries from that cycle since it's already provided |
| G | `Row_In_Window` | 0..111, cycles — sorts rows within a window into correct temporal order |
| I | `LED1` | feature channel |
| J | `LED2` | feature channel |
| K | `LED3` | feature channel |
| L | `TempC_x100` | feature channel (temperature × 100, integer-scaled) |
| M | `HumidityRH_x100` | feature channel (humidity × 100) |
| N | `AccX_mg` | feature channel |
| O | `AccY_mg` | feature channel |
| P | `AccZ_mg` | feature channel |
| R | `Survey_ID` | **cross-validation group key** — not part of the originally-specified column set, but required: groups every window from the same labeling interval together so temporally-adjacent correlated windows can't split across train/test. Verified globally unique across every file within a lineage (checked by `loader.py` at load time — raises if violated). |
| W | `Stress_0to100` | regression target, continuous 0-100 |
| X | `Label` | classification target, one of the 4-class taxonomy (§1) |

**The 8 feature channels are used exactly as given — no derived
cross-LED channels (`LED3-LED1`, `LED3-LED2`) are added**, unlike the
other two phases. This was a deliberate choice to honor an explicit
column spec rather than assume the same feature engineering applies; it's
flagged in `README.md` as an easy, likely-beneficial follow-up if wanted.

**Lineage semantics** (why there are two datasets per participant, each
producing its own model):
- `all_continuous`: every window across the full continuous recording is
  labeled — mostly by carrying a survey's answer forward across the whole
  gap until the next survey (see `Period` column values
  `before_30to60min` / `past_5to30min` / `current_5min` inside one
  `Survey_ID` group — only `current_5min` is the "real" response window;
  the earlier periods default toward the prior confirmed state). Larger,
  noisier.
- `labeled_last5min`: only the `current_5min` period is kept. Smaller,
  more precisely tied to the actual survey response.

`ring01_pid104` first-drop sizes: `all_continuous` = 4,528 windows / 43
`Survey_ID` groups; `labeled_last5min` = 583 windows / 36 groups. Both are
heavily imbalanced — `distress` and `eustress` each come from only **2
independent survey events** in the whole dataset. This is the main
limiting factor on current model quality (see §6.10), not a pipeline bug.

### 6.4 `config.py`

```python
PHASE_ROOT, DATA_DIR, INCOMING_DIR, DATASET_DIR, RESULTS_DIR   # paths

PARTICIPANTS = {
    "ring01_pid104": {
        "dir": DATA_DIR / "Ring_01_PID_104",
        "lineages": {
            "all_continuous":   "1_All_continous",     # note: "continous" typo in the real folder name, preserved as-is
            "labeled_last5min": "2_labeled_last5min",
        },
    },
}

WINDOW_SAMPLES = 112
ROW_IN_WINDOW_COL = "Row_In_Window"
WINDOW_ID_COL = "Global_Window"
GROUP_COL = "Survey_ID"
CLASS_LABEL_COL = "Label"
REGRESSION_LABEL_COL = "Stress_0to100"
RAW_CHANNELS = ["LED1","LED2","LED3","TempC_x100","HumidityRH_x100","AccX_mg","AccY_mg","AccZ_mg"]
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]
N_FOLDS = 5
RANDOM_STATE = 42

CNN_PARAMS = { filters_1=32, kernel_1=7, pool_1=2, filters_2=64, kernel_2=5, pool_2=2,
               filters_3=128, kernel_3=3, conv_dropout=0.25, dense_1=64, dense_dropout=0.30,
               learning_rate=1e-3, batch_size=32, epochs=200, patience=20, val_split=0.15,
               random_state=42 }   # same values as the root/pilot_b CNN_PARAMS, but a
                                   # SEPARATE dict instance (self-containment rule)

TUNE_GRID = [ {}, {"learning_rate": 5e-4},
              {"dense_1": 32, "conv_dropout": 0.35, "dense_dropout": 0.40},
              {"filters_1": 24, "filters_2": 48, "filters_3": 96, "learning_rate": 5e-4} ]
TUNE_EPOCHS = 60
TUNE_PATIENCE = 8
```

**To register a new participant**: add an entry to `PARTICIPANTS` with
their data directory and lineage-folder-name mapping. If their column
layout differs from §6.3, update the column constants here *and*
`REQUIRED_COLUMNS` in `loader.py` — don't assume it's identical without
checking the actual file first (open it with pandas and print `.columns`,
exactly as was done to establish the spec in §6.3).

### 6.5 `src/loader.py`

Public API: `load_lineage_dataset(participant: str, lineage: str, use_cache=True, force_reload=False) -> LineageDataset`.

`LineageDataset` fields: `X` `(N, 112, 8)` float32, `y_label` `(N,)` str,
`y_stress` `(N,)` float32, `groups` `(N,)` str (`Survey_ID`), `source_file`
`(N,)` str, `n_windows_dropped: int`.

Behavior:
1. Globs `*.xlsx` in the lineage's directory, reads each with
   `pandas.read_excel`.
2. Groups rows by `Global_Window`, sorts each group by `Row_In_Window`.
3. **Validates** every window: exactly 112 rows, exactly one distinct
   `Label`/`Stress_0to100`/`Survey_ID` value. Anything else is dropped
   with a `warnings.warn` (never silently) and counted in
   `n_windows_dropped`. (For the `ring01_pid104` drop: 0 windows dropped
   in either lineage.)
4. Concatenates across files, then checks `Survey_ID` is globally unique
   within the lineage (raises `ValueError` if the same `Survey_ID`
   appears under two different source files with different content).
5. Caches the result as `data/dataset/<participant>/<lineage>.npz`
   (gitignored — see `.gitignore`'s `PilotPhaseC/data/dataset/**/*.npz`
   entry). Subsequent loads read the cache unless `force_reload=True`.

Run standalone for a quick dataset report:
```bash
python -m PilotPhaseC.src.loader --lineage labeled_last5min [--participant ring01_pid104] [--force-reload]
```

### 6.6 `src/models.py`

Two self-contained classes (no import from the root `src/models/cnn_model.py`
— this is the one piece of real duplication the self-containment rule
causes, and it's intentional).

**Shared conv backbone** (`_conv_backbone`, identical for both):
```
Input (112, 8)
  -> Conv1D(filters_1, k=kernel_1) -> BN -> ReLU -> MaxPool(pool_1) -> Dropout(conv_dropout)
  -> Conv1D(filters_2, k=kernel_2) -> BN -> ReLU -> MaxPool(pool_2) -> Dropout(conv_dropout)
  -> Conv1D(filters_3, k=kernel_3) -> BN -> ReLU -> GlobalAvgPool    -> Dropout(conv_dropout)
```

`CNNClassifier(params)`: head = `Dense(dense_1, relu) -> Dropout(dense_dropout) -> Dense(n_classes, softmax)`.
Loss `sparse_categorical_crossentropy`. `fit(X, y)` label-encodes y, computes
`compute_class_weight("balanced")`, shuffles, trains with EarlyStopping
(`monitor="val_loss"`, `patience`) + ReduceLROnPlateau. `predict()` returns
original string labels (inverse-transformed); `predict_proba()` returns
softmax output.

`CNNRegressor(params)`: head = `Dense(dense_1, relu) -> Dropout(dense_dropout) -> Dense(1, linear)`.
Loss `mse`, metric `mae`. **Trains and predicts in whatever scale `y` is
passed in** — it does not scale the target itself; the caller (see §6.9)
is responsible for z-scoring the target before `fit()` and inverting
`predict()`'s output. No class weighting (not applicable to regression).

Both: each `fit()` call rebuilds the Keras model from scratch (safe to
reuse one Python object across CV folds), seeds `np.random` + `tf.random`
from `params["random_state"]`, accepts an `epochs=`/`patience=` override on
`fit()` (used by the tuner for a cheaper search budget without touching
`config.CNN_PARAMS`).

### 6.7 `src/common.py`

- `fit_channel_scaler(X_train)` / `apply_channel_scaler(X, mean, std)` —
  per-channel mean/std over `(N, T)` axes, fit on train only, zero-std
  guarded to 1.0.
- `fit_target_scaler(y_train)` / `apply_target_scaler` / `invert_target_scaler` —
  same idea, scalar mean/std, for the regression target.
- `export_tflite(keras_model, out_path) -> int` (bytes written) — plain
  float32 `TFLiteConverter.from_keras_model(...).convert()`.
- `quantize_int8(keras_model, X_calib, random_state, n_calib=200) -> bytes` —
  full-integer INT8 quantization (`TFLITE_BUILTINS_INT8`, int8 in/out),
  calibrated on `n_calib` randomly-sampled rows of `X_calib` (should be the
  already-scaled training data).
- `estimate_tensor_arena(tflite_bytes) -> dict` — the flatbuffer-parsing
  RAM estimator described in §3; returns `estimated_arena_bytes`,
  `largest_single_activation_bytes`, `total_weight_bytes`, `n_tensors`,
  `n_ops`.

### 6.8 `src/tuning.py`

`tune_classifier(X, y, groups, base_params, random_state) -> (best_params, log)`
and `tune_regressor(...)` (same signature, `y` = `Stress_0to100`).

Both: take **one** `GroupShuffleSplit(test_size=0.2)` split respecting
`groups` (so the search itself can't leak across `Survey_ID`), fit+scale
on the train side of that split, evaluate each candidate in
`config.TUNE_GRID` at the reduced `TUNE_EPOCHS`/`TUNE_PATIENCE` budget,
pick the best by validation macro-F1 (classifier) or RMSE (regressor).
**This is a cheap search, not full nested CV** — it exists to avoid
training the full 5-fold CV + final model on an arbitrary/default
hyperparameter set, at a cost of one extra lightweight pass. The winning
`params` dict is what the caller then uses for the full CV and final fit
at the *full* `epochs`/`patience` budget from `CNN_PARAMS`.

### 6.9 `src/train_classifier.py` / `src/train_regressor.py`

```bash
python -m PilotPhaseC.src.train_classifier --lineage all_continuous [--participant ring01_pid104] [--n_folds 5] [--no-tune] [--output_dir ...]
python -m PilotPhaseC.src.train_regressor   --lineage labeled_last5min ...
```

Pipeline (identical shape for both, different model/CV/metrics):
1. `load_lineage_dataset(...)` (cached).
2. Tune (skippable with `--no-tune`, which just uses `CNN_PARAMS` as-is).
3. Full CV at the winning config: `StratifiedGroupKFold` (classifier) or
   `GroupKFold` (regressor), `n_folds` splits, grouped by `groups`
   (`Survey_ID`). Per-fold: fit channel scaler (+ target scaler, regressor
   only) on train rows of that fold, apply to test rows, train, predict,
   score. Regression scores are computed **after inverting back to real
   0-100 units** — nothing in `fold_results.csv`/`cv_report.txt` is in
   z-scored space.
4. Save `fold_results.csv` + `cv_report.txt` (classification: accuracy/F1
   macro/F1 weighted + full `classification_report` + confusion matrix;
   regression: MAE/RMSE/R²).
5. Retrain on **100% of the lineage's data** at the winning config → this
   is the deployed model.
6. Export: `cnn_1d_model.pkl` (joblib), `cnn_1d_scaler.npz` (channel
   `mean`/`std`, plus `target_mean`/`target_std` for regression),
   `cnn_1d_model.tflite` (float32), `cnn_1d_model_int8.tflite` (INT8) +
   `int8_quantization_report.txt` (includes a float32-vs-INT8 fidelity
   check on a random slice of training data — **not** a generalization
   test, explicitly labeled as such in the report).
7. Write `manifest.json`: task, participant, lineage, timestamp, git
   commit, `n_windows`/`n_groups`/`n_windows_dropped`, classes (or target
   name), `n_folds`, whether tuning ran + its full log, the winning
   `best_params`, `cv_metrics`, and `final_model` (file sizes, fidelity
   check, arena estimate). This is the single file that answers "what
   exactly produced this model" for any run.

**Regression target-scaling detail worth remembering**: the Keras model
is trained and predicts in z-scored space (`(Stress_0to100 - mean) / std`,
fit on train only). The `.pkl`/float32-`.tflite` usage path in
`train_regressor.py` always inverts before reporting/exporting metrics.
**The raw INT8 `.tflite` file's own output tensor is still in z-scored
space** — if you load that file directly (e.g. for firmware), you must
apply `y = scaled_output * target_std + target_mean` yourself using the
`target_mean`/`target_std` values from that run's `cnn_1d_scaler.npz` or
`manifest.json`. This is called out explicitly in each run's
`int8_quantization_report.txt`.

### 6.10 Current results (`ring01_pid104`, first drop)

Full detail + caveats: `PilotPhaseC/results/ring01_pid104/RESULTS_SUMMARY.md`.
Headline numbers:

| Lineage | Task | CV metric |
|---|---|---|
| `all_continuous` | Classification | F1 macro 0.294 ± 0.022 |
| `all_continuous` | Regression | R² -0.527 ± 0.495 (MAE 9.7) |
| `labeled_last5min` | Classification | F1 macro 0.327 ± 0.095 |
| `labeled_last5min` | Regression | R² -0.053 ± 0.511 (MAE 10.6) |

**Why these are modest, and why that's not a pipeline defect**: `distress`
and `eustress` each have only 2 independent `Survey_ID` groups in the
entire first drop; most CV folds see those classes only in train or only
in test, never both — essentially unlearnable with this little
independent evidence. The regression target effectively takes only
~20-25 distinct real values (one scalar per survey, replicated across
every window in that survey's group) spread across 36-43 groups total.
Float32-vs-INT8 agreement was high (classification 0.955-1.000;
regression off by 0.14-0.28 stress points) — quantization is not where
accuracy is being lost. **The fix is more data** (more participants, or
more days for this participant, especially more surveys landing on the
rare classes), not more tuning. Re-running the exact same `train_classifier.py`/
`train_regressor.py` commands against a future, larger drop is the
expected path forward — the pipeline doesn't need to change, just the
data registered in `PARTICIPANTS`.

### 6.11 Adding a new participant / data drop

1. Drop raw files under `PilotPhaseC/data/<Ring_ID>_<PID>/<lineage>/*.xlsx`,
   matching the column layout in §6.3 — **verify this with pandas first**
   (`pd.read_excel(path).columns`) rather than assuming a new source
   matches; devices/exports have varied before (e.g. Pilot B's humidity
   sensor only on 2 of 5 participants).
2. Add an entry to `config.py`'s `PARTICIPANTS` dict.
3. Run `python -m PilotPhaseC.src.loader --participant <key> --lineage <name>`
   for each lineage to sanity-check parsing (window count, dropped-window
   count, label distribution) before training anything.
4. Run `train_classifier.py` / `train_regressor.py` for each lineage.
5. Consider whether this new data should also be pooled with
   `ring01_pid104` into a cross-participant model (not built yet — there
   is currently no generalized/pooled training path in `PilotPhaseC`,
   unlike `pilot_b`'s `train_generalized.py`; if multiple participants
   accumulate, that's the natural next piece to add, following
   `pilot_b/src/train_generalized.py`'s pattern as a reference, but
   reimplemented self-contained inside `PilotPhaseC/`).

## 7. Known repo-wide gotchas

- **One pre-existing, unrelated test failure**: `tests/test_merge_labels.py::TestMerge::test_xlsx_feature_file_accepted`
  fails on Windows with a `PermissionError` on a temp `.xlsx` file (file
  locking quirk, not a real bug — confirmed to fail identically on commits
  from before any Pilot B/Pilot Phase C work existed). Everything else in
  `tests/` passes (51 tests).
- **Windows CRLF warnings**: `git add`/`commit` print
  `warning: ... LF will be replaced by CRLF ...` for most files — harmless,
  not an error, no action needed.
- **`.gitignore` additions**: `*.h5`/`*.h` (old model-binary convention,
  mostly superseded — current CNN exports use `.pkl`+`.tflite`, which
  *are* committed despite being regenerable, for sharing convenience) and
  `PilotPhaseC/data/dataset/**/*.npz` (pure derived cache, regenerate via
  `loader.py` rather than expecting it in a fresh clone).
- **Zip files**: several `results/pilot_b_*.zip`-style and
  `PilotPhaseC/results.zip` files exist as manual "bundle for sharing"
  copies of already-tracked directories. They're redundant but harmless;
  don't be surprised that their contents duplicate a sibling directory.
- **Typo preserved on purpose**: `PilotPhaseC/config.py`'s lineage folder
  name `"1_All_continous"` really is missing the second "u" in
  "continuous" — that's the actual folder name on disk, not a bug in the
  config.

## 8. Where to look for more detail

| Question | Doc |
|---|---|
| Original pipeline narrative, design decisions, full results | `README.md`, `IMPLEMENTATION_REPORT.md` |
| Deploying/running inference with any trained model (original + Pilot B) | `INFERENCE_GUIDE.md` |
| CNN architecture sizing tradeoffs (why these filter counts) | `CNN_ARCH_TUNING_GUIDE.md` |
| Status handoff for the tabular models | `HANDOFF_ANISHA.md` |
| Pilot B generalized/small/personalized model results | `pilot_b/results/*/`.md summaries |
| Pilot Phase C data format + workflow (human-facing) | `PilotPhaseC/README.md` |
| Pilot Phase C `ring01_pid104` results + caveats | `PilotPhaseC/results/ring01_pid104/RESULTS_SUMMARY.md` |
| Exact reproducibility detail for any one Pilot Phase C model | that model's own `manifest.json` |
