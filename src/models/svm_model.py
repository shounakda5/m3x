import numpy as np
import pandas as pd
from sklearn.svm import SVC

from config import SVM_PARAMS
from src.models.base import BaseClassifier


class SVMClassifier(BaseClassifier):
    name = "SVM"

    def __init__(self, **overrides):
        params = {**SVM_PARAMS, **overrides}
        # probability=True enables predict_proba via Platt scaling
        self._model = SVC(**params, probability=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def requires_scaling(self) -> bool:
        return True
