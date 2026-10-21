from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
SURVEY_DIR   = DATA_DIR / "survey"
FEATURES_DIR = DATA_DIR / "features"
RAW_DIR      = DATA_DIR / "raw"
RESULTS_DIR  = ROOT / "results"

LABELED_FEATURES_FILE    = FEATURES_DIR / "features_labeled.csv"
LABELED_FEATURES_MF_FILE = FEATURES_DIR / "features_labeled_mf.csv"

RAW_SESSION1_FILE     = RAW_DIR / "session1_raw.csv"
RAW_SESSION2_FILE     = RAW_DIR / "session2_raw.csv"
RAW_MF_SESSION1_FILE  = RAW_DIR / "session1_raw_mf.xlsx"
RAW_MF_SESSION2_FILE  = RAW_DIR / "session2_raw_mf.xlsx"

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
# Metadata columns that identify a row but are not predictive features.
INDEX_COLS = [
    "Block_ID", "SubWindow_ID",
    "Block_Start_s", "Block_End_s", "Block_Duration_s", "Fs_Est_Hz",
]

LABEL_COL = "Label"

# Group identifier for StratifiedGroupKFold.
# All sub-windows within the same block must stay in the same fold to
# prevent correlated-sample leakage between train and test.
GROUP_COL = "Block_ID"

# ---------------------------------------------------------------------------
# Feature columns (derived from Farhad's M3X extraction pipeline)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # LED1 optical pulse features
    "PA1", "PA_STD_LED1", "PA_RMSSD_LED1", "PA_CV_LED1", "mean_LED1",
    # LED1 heart rate and HRV
    "HR_LED1", "SDNN_LED1", "RMSSD_LED1",
    # LED2 optical pulse features
    "PA2", "PA_STD_LED2", "PA_RMSSD_LED2", "PA_CV_LED2", "mean_LED2",
    # LED3 optical pulse features (new M3X channel)
    "PA3", "PA_STD_LED3", "PA_RMSSD_LED3", "PA_CV_LED3", "LED3_mean",
    # Cross-channel raw mean differences
    "LED1_LED2_Raw_Mean_Diff", "LED1_LED2_Raw_Mean_AbsDiff",
    "LED3_LED1_Raw_Mean_Diff", "LED3_LED1_Raw_Mean_AbsDiff",
    "LED3_LED2_Raw_Mean_Diff", "LED3_LED2_Raw_Mean_AbsDiff",
    # Cross-LED pulse amplitude differences and ratios
    "PA1_PA2_mean_diff", "PA1_PA2_AbsDiff",
    "PA1_PA3_mean_diff", "PA1_PA3_AbsDiff",
    "PA2_PA3_mean_diff", "PA2_PA3_AbsDiff",
    "PA1_PA3_Ratio", "PA2_PA3_Ratio", "PA1_PA2_Ratio",
    # Temperature
    "mean_temperature",
    # Accelerometer
    "AccX_STD", "AccY_STD", "AccZ_STD", "Acc_Total_STD",
]

# ---------------------------------------------------------------------------
# Label taxonomy  (order matters for confusion-matrix axis labels)
# ---------------------------------------------------------------------------
CLASSES = ["relax", "eustress", "distress", "eustress+distress"]

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_FOLDS = 5

# ---------------------------------------------------------------------------
# SVM hyperparameters
# ---------------------------------------------------------------------------
SVM_PARAMS = {
    "C":            6.552,
    "kernel":       "rbf",
    "gamma":        0.3003,
    "class_weight": "balanced",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Random Forest hyperparameters
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators":    406,
    "max_depth":       15,
    "min_samples_split": 6,
    "min_samples_leaf":  3,
    "max_features":    "sqrt",
    "class_weight":    "balanced",
    "random_state":    42,
    "n_jobs":          -1,
}

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# Tuned for a small tabular dataset (~861 samples, 16 features, 3 classes).
# Shallow trees + moderate learning rate + subsampling prevent overfitting.
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "eval_metric":       "mlogloss",
    "verbosity":         0,
    "random_state":      42,
    "n_jobs":            -1,
}

# ---------------------------------------------------------------------------
# 1D CNN on raw sensor data — hyperparameters
#
# Input shape : (RAW_WINDOW_SAMPLES, len(RAW_CHANNELS)) = (112, 9)
# Architecture: 3 × Conv1D block → GlobalAvgPool → Dense head → Softmax
# ---------------------------------------------------------------------------

# Raw data windowing constants
RAW_SAMPLING_HZ         = 25        # nominal M3X sampling rate
RAW_WINDOW_S            = 4.5       # window duration in seconds
RAW_WINDOW_SAMPLES      = 112       # = floor(RAW_WINDOW_S * RAW_SAMPLING_HZ)
RAW_N_WINDOWS_PER_BURST = 2         # sub-windows per duty-cycle ON burst
RAW_GAP_THRESHOLD_MS    = 1_000     # inter-sample gap (ms) that marks duty-cycle OFF
RAW_MIN_BURST_S         = 8.0       # bursts shorter than this are discarded
RAW_CHANNELS            = [         # 7 raw + 2 derived cross-LED differences
    "LED1", "LED2", "LED3",
    "Temp", "AccX", "AccY", "AccZ",
    "LED3_LED1",    # LED3 - LED1
    "LED3_LED2",    # LED3 - LED2
]

CNN_PARAMS = {
    # Conv block 1  (temporal kernel spans ~280 ms at 25 Hz)
    "filters_1":     32,
    "kernel_1":       7,
    "pool_1":         2,

    # Conv block 2  (kernel spans ~200 ms)
    "filters_2":     64,
    "kernel_2":       5,
    "pool_2":         2,

    # Conv block 3  (kernel spans ~120 ms; GlobalAvgPool after)
    "filters_3":    128,
    "kernel_3":       3,

    "conv_dropout":  0.25,

    # FC head
    "dense_1":       64,
    "dense_dropout": 0.30,

    # Training
    "learning_rate": 1e-3,
    "batch_size":    32,
    "epochs":        200,
    "patience":      20,
    "val_split":     0.15,

    "random_state":  42,
}
