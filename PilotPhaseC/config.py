"""
Phase-specific configuration for Pilot Phase C.

Unlike the original pipeline (data/) and the Pilot B cohort (pilot_b/), this
phase's incoming data arrives **already labeled** -- there is no survey/
classify_survey/merge_labels step here. The pipeline is just:

    raw per-lineage Excel files -> windowed dataset -> tune -> CV -> final model -> export

Everything in this file is self-contained to PilotPhaseC -- nothing here is
imported from (or imports) the root config.py or root src/, by design.

First real data drop (2026-10-21): Ring_01_PID_104, one participant, two
"lineages" of pre-labeled data, each expected to produce its own bespoke
model (see README.md for the full data-format writeup):

    1_All_continous      -- every window across the full continuous
                            recording is labeled (mostly by carrying a
                            survey's answer forward across the whole gap
                            until the next survey) -- larger, noisier
    2_labeled_last5min   -- only the ~5 minutes immediately preceding each
                            survey are labeled -- smaller, more precise
"""

from pathlib import Path

PHASE_ROOT    = Path(__file__).parent
DATA_DIR      = PHASE_ROOT / "data"
INCOMING_DIR  = DATA_DIR / "incoming"   # unused by this first drop; reserved for a
                                         # future batch-manifest convention if one
                                         # ever ends up being needed again
DATASET_DIR   = DATA_DIR / "dataset"    # windowed-dataset .npz caches (one per lineage)
RESULTS_DIR   = PHASE_ROOT / "results"

# ---------------------------------------------------------------------------
# Participant / lineage registry
#
# Each entry is one person's raw data drop. New participants will keep
# arriving over the coming weeks as new subdirectories under DATA_DIR,
# each following the same two-lineage layout learned from this first drop.
# ---------------------------------------------------------------------------
PARTICIPANTS = {
    "ring01_pid104": {
        "dir": DATA_DIR / "Ring_01_PID_104",
        "lineages": {
            "all_continuous":    "1_All_continous",
            "labeled_last5min":  "2_labeled_last5min",
        },
    },
}

# ---------------------------------------------------------------------------
# Raw Excel column layout (verified against the Ring_01_PID_104 files)
#
#   G  Row_In_Window     0..111, cycles -- sorts rows within one window
#   D  Global_Window     segments the flat row stream into 112-row windows
#                        (verified 1:1 with Row_In_Window's cycle; used
#                        instead of re-deriving window boundaries from the
#                        0->111 reset pattern, since it's already provided
#                        and already validated)
#   I-P                  the 8 sensor columns from Excel (RAW_CHANNELS below);
#                        two derived cross-LED diffs are computed in loader.py
#                        (DERIVED_CHANNELS) for MODEL_CHANNELS fed to the CNN
#   R  Survey_ID         NOT part of the user's column spec, but required:
#                        groups every window belonging to the same labeling
#                        interval together, so temporally-adjacent (highly
#                        correlated) windows can't split across train/test
#                        in cross-validation. Verified globally unique
#                        across every file in a lineage (no cross-file
#                        collisions) -- used directly as the CV group key.
#   W  Stress_0to100     regression target
#   X  Label             classification target
# ---------------------------------------------------------------------------
WINDOW_SAMPLES   = 112
ROW_IN_WINDOW_COL = "Row_In_Window"
WINDOW_ID_COL     = "Global_Window"
GROUP_COL         = "Survey_ID"
CLASS_LABEL_COL   = "Label"
REGRESSION_LABEL_COL = "Stress_0to100"
PERIOD_COL = "Period"  # optional in Excel; used for training weights on all_continuous

RAW_CHANNELS = [
    "LED1", "LED2", "LED3",
    "TempC_x100", "HumidityRH_x100",
    "AccX_mg", "AccY_mg", "AccZ_mg",
]

# Per-sample cross-LED differences (same convention as root train_cnn / pilot_b)
DERIVED_CHANNELS = ["LED3_LED1", "LED3_LED2"]
MODEL_CHANNELS = RAW_CHANNELS + DERIVED_CHANNELS  # CNN input: (112, 10)

# ---------------------------------------------------------------------------
# Label taxonomy (verified present in the Ring_01_PID_104 data; order
# matches the root project's established taxonomy for the same study, kept
# consistent on purpose so confusion-matrix axes line up across phases)
# ---------------------------------------------------------------------------
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

# ---------------------------------------------------------------------------
# Cross-validation / training
# ---------------------------------------------------------------------------
N_FOLDS       = 5
RANDOM_STATE  = 42
VAL_GROUP_FRACTION = 0.15  # early-stopping holdout: fraction of Survey_ID groups

# Per-window training multipliers (× survey-equal weights). All weights 1.0:
# window labels are treated as authoritative everywhere they appear.
LINEAGE_PERIOD_WEIGHTS = {
    "all_continuous": {
        "current_5min": 1.0,
        "past_5to30min": 1.0,
        "before_30to60min": 1.0,
        "baseline_since_previous_survey": 1.0,
    },
    "labeled_last5min": {
        "current_5min": 1.0,
        "baseline_since_previous_survey": 1.0,
    },
}
DEFAULT_PERIOD_WEIGHT = 1.0

# Training policy (fixed for this phase)
CLASSIFIER_SURVEY_EQUAL_WEIGHTS = False
CLASSIFIER_CLASS_BALANCED_WEIGHTS = True
REGRESSOR_SURVEY_EQUAL_WEIGHTS = False

# ---------------------------------------------------------------------------
# CNN hyperparameters -- shared conv backbone for both the classifier and
# the regressor (same architecture family as the rest of the project's raw
# 1D-CNN models, re-implemented self-contained here rather than imported).
# TUNE_GRID (in train_classifier.py / train_regressor.py) searches a few of
# these; the values below are the defaults/fallback and the final-model
# training budget (epochs/patience are NOT searched -- only used at full
# budget for the winning config's CV + final fit).
# ---------------------------------------------------------------------------
CNN_PARAMS = {
    "filters_1":     32,
    "kernel_1":       7,
    "pool_1":         2,

    "filters_2":     64,
    "kernel_2":       5,
    "pool_2":         2,

    "filters_3":    128,
    "kernel_3":       3,

    "conv_dropout":  0.25,

    "dense_1":       64,
    "dense_dropout": 0.30,

    "learning_rate": 1e-3,
    "batch_size":    32,
    "epochs":        200,
    "patience":      20,
    "val_split":     0.15,

    "random_state":  RANDOM_STATE,
}

# Lightweight hyperparameter search space used before CV (see
# src/tuning.py) -- intentionally small: a handful of candidates evaluated
# on one group-aware validation split, at a reduced epoch budget, just to
# pick a sane learning_rate/capacity/dropout combo before committing to the
# full CV + final-fit budget above.
TUNE_GRID = [
    {},  # CNN_PARAMS defaults
    {"learning_rate": 5e-4},
    {"dense_1": 32, "conv_dropout": 0.35, "dense_dropout": 0.40},
    {"filters_1": 24, "filters_2": 48, "filters_3": 96, "learning_rate": 5e-4},
]
TUNE_EPOCHS   = 60   # reduced budget for the search phase only
TUNE_PATIENCE = 8
