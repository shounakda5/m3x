"""
1D CNN classifier for physiological stress classification.

Architecture
------------
  Input  : (n_features, 1)  — feature vector treated as a 1-D sequence
  Block 1: Conv1D → BatchNorm → ReLU → MaxPool → Dropout
  Block 2: Conv1D → BatchNorm → ReLU → GlobalAveragePool → Dropout
  FC 1   : Dense(128) + ReLU + Dropout
  FC 2   : Dense(64)  + ReLU + Dropout
  FC 3   : Dense(32)  + ReLU + Dropout
  Output : Dense(n_classes) + Softmax       ← 4th dense layer (incl. output)

The model requires feature scaling (returns True from requires_scaling()).
Class weights are computed inside fit() to handle label imbalance.
Each fit() call rebuilds the model from scratch, so the same instance
can be reused safely across cross-validation folds.
"""

import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF C++ warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import CNN_PARAMS
from src.models.base import BaseClassifier


class CNNClassifier(BaseClassifier):
    name = "CNN_1D"

    def __init__(self, **overrides):
        self._params = {**CNN_PARAMS, **overrides}
        self._model = None
        self._label_encoder = LabelEncoder()

    def requires_scaling(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        seed = self._params["random_state"]
        np.random.seed(seed)
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        tf.random.set_seed(seed)

    def _build_model(self, n_features: int, n_classes: int):
        import tensorflow as tf
        from tensorflow import keras

        p = self._params
        inputs = keras.Input(shape=(n_features, 1))

        # --- Conv block 1 ---
        x = keras.layers.Conv1D(
            p["filters_1"], kernel_size=p["kernel_size"], padding="same"
        )(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)
        x = keras.layers.Dropout(p["conv_dropout"])(x)

        # --- Conv block 2 ---
        x = keras.layers.Conv1D(
            p["filters_2"], kernel_size=p["kernel_size"], padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dropout(p["conv_dropout"])(x)

        # --- FC 1 ---
        x = keras.layers.Dense(p["dense_1"], activation="relu")(x)
        x = keras.layers.Dropout(p["dense_dropout"])(x)

        # --- FC 2 ---
        x = keras.layers.Dense(p["dense_2"], activation="relu")(x)
        x = keras.layers.Dropout(p["dense_dropout"] * 0.75)(x)

        # --- FC 3 ---
        x = keras.layers.Dense(p["dense_3"], activation="relu")(x)
        x = keras.layers.Dropout(p["dense_dropout"] * 0.5)(x)

        # --- Output (FC 4) ---
        outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import tensorflow as tf
        from tensorflow import keras

        self._set_seeds()

        # Encode string labels → integers (fit on training labels only)
        y_enc = self._label_encoder.fit_transform(y).astype(np.int32)
        n_features = X.shape[1]
        n_classes = len(self._label_encoder.classes_)

        # Reshape to (samples, timesteps, channels) for Conv1D
        X_arr = X.values.reshape(-1, n_features, 1).astype(np.float32)

        # Shuffle before validation_split so the held-out slice is representative
        rng = np.random.RandomState(self._params["random_state"])
        idx = rng.permutation(len(X_arr))
        X_arr, y_enc = X_arr[idx], y_enc[idx]

        # Class weights to counteract label imbalance
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_enc,
        )
        class_weight_dict = dict(enumerate(weights))

        self._model = self._build_model(n_features, n_classes)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._params["patience"],
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        self._model.fit(
            X_arr,
            y_enc,
            epochs=self._params["epochs"],
            batch_size=self._params["batch_size"],
            validation_split=self._params["validation_split"],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values.reshape(-1, X.shape[1], 1).astype(np.float32)
        proba = self._model.predict(X_arr, verbose=0)
        idx = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(idx)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values.reshape(-1, X.shape[1], 1).astype(np.float32)
        return self._model.predict(X_arr, verbose=0)
