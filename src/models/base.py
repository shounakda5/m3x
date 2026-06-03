from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Minimal interface every classifier must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def requires_scaling(self) -> bool:
        """Return True if this model needs StandardScaler applied to features.

        Distance-based models (SVM, KNN, logistic regression) return True.
        Tree-based models (RF, XGBoost) return False.
        """
        return False
