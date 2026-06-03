"""
XGBoost classifier for physiological stress classification.

XGBoost is a gradient-boosted tree ensemble.  Each successive tree corrects
the residual errors of the previous ensemble, making it more sample-efficient
than a bagged ensemble (Random Forest) on small datasets.

Key design choices
------------------
* requires_scaling() → False  — tree-based models are scale-invariant.
* LabelEncoder is used internally so the public API always speaks in string
  class names (Distress / Mixed / Relaxed), matching every other model.
* Per-sample class weights (compute_sample_weight("balanced")) are computed
  inside fit() to address the 60/23/17 label imbalance without needing a
  class_weight parameter that XGBoost's multiclass objective does not support.
* The underlying XGBBase is rebuilt on each fit() call so the same wrapper
  instance can be reused across cross-validation folds without stale state.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier as XGBBase

from config import XGB_PARAMS
from src.models.base import BaseClassifier


class XGBClassifier(BaseClassifier):
    name = "XGBoost"

    def __init__(self, **overrides):
        self._params = {**XGB_PARAMS, **overrides}
        self._label_encoder = LabelEncoder()
        self._model: XGBBase | None = None

    def requires_scaling(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y_enc = self._label_encoder.fit_transform(y).astype(np.int32)

        # Per-sample weights so minority classes (Distress, Relaxed) receive
        # proportionally higher gradients during boosting.
        sample_weights = compute_sample_weight("balanced", y_enc)

        # Rebuild so repeated fit() calls across CV folds start from scratch.
        self._model = XGBBase(**self._params)
        self._model.fit(X.values, y_enc, sample_weight=sample_weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_enc = self._model.predict(X.values).astype(np.int32)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.values)
