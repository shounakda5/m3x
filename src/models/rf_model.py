import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import RF_PARAMS
from src.models.base import BaseClassifier


class RFClassifier(BaseClassifier):
    name = "RandomForest"

    def __init__(self, **overrides):
        params = {**RF_PARAMS, **overrides}
        self._model = RandomForestClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self, feature_cols: list) -> pd.Series:
        """Mean decrease in impurity across all trees, sorted descending."""
        return (
            pd.Series(self._model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
