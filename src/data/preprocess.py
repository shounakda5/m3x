"""
Feature preprocessing pipeline.

Responsibilities:
  - Load a labeled feature CSV and validate the Label column is present.
  - Separate index columns from predictive features.
  - Clean features: replace infinities, impute missing values with column medians.
  - Fit / apply StandardScaler (used by SVM; RF skips scaling).

Scaler fitting must always happen on the training fold only.  Never fit on
the full dataset before splitting — that leaks test-set statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, INDEX_COLS, LABEL_COL, GROUP_COL


def load_labeled_dataset(path: str) -> pd.DataFrame:
    """Load a merged feature+label CSV. Raises clearly if the Label column is absent."""
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(
            f"'{path}' has no '{LABEL_COL}' column.\n"
            "Run  python -m src.data.merge_labels  first to attach survey labels."
        )
    return df


def extract_Xy(df: pd.DataFrame):
    """
    Return (X, y, groups, feature_cols) from a labeled DataFrame.

    X            — feature matrix restricted to FEATURE_COLS defined in config
    y            — label Series
    groups       — Block_ID Series used for GroupKFold grouping
    feature_cols — list of feature column names present in this DataFrame
    """
    # Use only columns that are in FEATURE_COLS AND present in this DataFrame
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X      = df[feature_cols].copy()
    y      = df[LABEL_COL].copy()
    groups = df[GROUP_COL].copy() if GROUP_COL in df.columns else None
    return X, y, groups, feature_cols


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace ±inf with NaN, then impute each column with its median.
    Median is more robust to outlier sensor spikes than mean.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X


def fit_scaler(X_train: pd.DataFrame):
    """Fit a StandardScaler on training data; return (scaler, X_train_scaled)."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return scaler, X_scaled


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a feature matrix without refitting."""
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
