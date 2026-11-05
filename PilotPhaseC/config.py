"""
Phase-specific configuration for Pilot Phase C.

Unlike the original pipeline (data/) and the Pilot B cohort (pilot_b/), this
phase's incoming data arrives **already labeled** -- there is no survey/
classify_survey/merge_labels step here. The pipeline is just:

    raw per-lineage Excel files -> windowed dataset -> tune -> CV -> final model -> export

Everything in this file is self-contained to PilotPhaseC -- nothing here is
imported from (or imports) the root config.py or root src/, by design.
"""

from pathlib import Path

PHASE_ROOT    = Path(__file__).parent
DATA_DIR      = PHASE_ROOT / "data"
INCOMING_DIR  = DATA_DIR / "incoming"
DATASET_DIR   = DATA_DIR / "dataset"
RESULTS_DIR   = PHASE_ROOT / "results"

# Native window lengths by acquisition format
WINDOW_SAMPLES_LEGACY = 112
WINDOW_SAMPLES_NEW    = 224
# Default for Ring_01 paths; Ring 3/4 deploy uses legacy 112-row windows only
WINDOW_SAMPLES   = WINDOW_SAMPLES_LEGACY

ROW_IN_WINDOW_COL = "Row_In_Window"
WINDOW_ID_COL     = "Global_Window"
GROUP_COL         = "Survey_ID"
CLASS_LABEL_COL   = "Label"
REGRESSION_LABEL_COL = "Stress_0to100"
PERIOD_COL = "Period"

RAW_CHANNELS = [
    "LED1", "LED2", "LED3",
    "TempC_x100", "HumidityRH_x100",
    "AccX_mg", "AccY_mg", "AccZ_mg",
]

DERIVED_CHANNELS = ["LED3_LED1", "LED3_LED2"]
MODEL_CHANNELS = RAW_CHANNELS + DERIVED_CHANNELS

CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

# Binary deploy mapping: Good = relax + eustress; Bad = eustress+distress + distress
CLASSES_BINARY = ["good", "bad"]
BINARY_GOOD_FOUR_CLASS = {"relax", "eustress"}
BINARY_BAD_FOUR_CLASS = {"distress", "eustress+distress"}

PARTICIPANTS = {
    "ring01_pid104": {
        "dir": DATA_DIR / "Ring_01_PID_104",
        "lineages": {
            "all_continuous":    "1_All_continous",
            "labeled_last5min":  "2_labeled_last5min",
        },
    },
    "ring03_pid102": {
        "dir": DATA_DIR / "Ring_03_PID_102",
        "lineages": {
            "all_continuous": "All_continous",
        },
        "deploy_file_set": "legacy",
    },
    "ring04_pid101": {
        "dir": DATA_DIR / "Ring_04_PID_101",
        "lineages": {
            "all_continuous": "All_continous",
        },
        "deploy_file_set": "legacy",
    },
}

# Pooled generalized classifier: all participants, legacy 112-row windows only (no new-format files)
GENERALIZED_CLASSIFICATION_PARTICIPANTS = [
    "ring01_pid104",
    "ring03_pid102",
    "ring04_pid101",
]

# Disabled — team uses legacy-only data; no new-format / upsample fallback path
LEGACY_FALLBACK_PARTICIPANTS: set[str] = set()
LEGACY_FALLBACK_ACCURACY_THRESHOLD = 0.65

# Motion score = std(AccX)+std(AccY)+std(AccZ) per window (raw units, before scaling)
MOTION_SCORE_NONWEAR_MAX = 8.0   # drop windows below (likely not worn)
MOTION_SCORE_HIGH_MIN = None     # optional upper cut; None = disabled

N_FOLDS       = 5
RANDOM_STATE  = 42
VAL_GROUP_FRACTION = 0.15

LINEAGE_PERIOD_WEIGHTS = {
    "all_continuous": {
        "current_5min": 1.0,
        "past_5to30min": 0.75,
        "before_30to60min": 0.50,
        "baseline_since_previous_survey": 1.0,
    },
    "labeled_last5min": {
        "current_5min": 1.0,
        "baseline_since_previous_survey": 1.0,
    },
}
DEFAULT_PERIOD_WEIGHT = 1.0
USE_PERIOD_WEIGHTS = True

CLASSIFIER_SURVEY_EQUAL_WEIGHTS = False
CLASSIFIER_CLASS_BALANCED_WEIGHTS = True
REGRESSOR_SURVEY_EQUAL_WEIGHTS = False

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

TUNE_GRID = [
    {},
    {"learning_rate": 5e-4},
    {"dense_1": 32, "conv_dropout": 0.35, "dense_dropout": 0.40},
    {"filters_1": 24, "filters_2": 48, "filters_3": 96, "learning_rate": 5e-4},
]
TUNE_EPOCHS   = 60
TUNE_PATIENCE = 8
