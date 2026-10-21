"""
models.py -- 1D CNN classifier and regressor for Pilot Phase C raw sensor
windows. Self-contained (does not import the root project's src/models/
CNNClassifier) per this phase's self-containment requirement, though the
architecture family is deliberately the same as the rest of the project's
raw-CNN models for consistency.

Input shape: (batch, WINDOW_SAMPLES, len(RAW_CHANNELS)) = (N, 112, 8)

Architecture (shared conv backbone)
------------------------------------
  Block 1: Conv1D(f1, k1) -> BN -> ReLU -> MaxPool(p1) -> Dropout
  Block 2: Conv1D(f2, k2) -> BN -> ReLU -> MaxPool(p2) -> Dropout
  Block 3: Conv1D(f3, k3) -> BN -> ReLU -> GlobalAvgPool -> Dropout
  Head (classifier): Dense(dense_1, ReLU) -> Dropout -> Dense(n_classes, Softmax)
  Head (regressor):  Dense(dense_1, ReLU) -> Dropout -> Dense(1, linear)

Design notes
------------
* Accepts raw numpy arrays (N, T, C), already per-channel scaled by the
  caller (see common.py's fit_channel_scaler/apply_channel_scaler) -- these
  classes don't scale inputs themselves.
* The regressor trains against a target that the CALLER has already scaled
  (see common.py's fit_target_scaler) -- it predicts in that scaled space;
  the caller inverts. This mirrors how input-channel scaling is handled.
* Each fit() call rebuilds the model from scratch, so instances are safe to
  reuse across CV folds without carrying over weights.
"""

import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def _conv_backbone(inp, p):
    from tensorflow import keras

    x = keras.layers.Conv1D(p["filters_1"], kernel_size=p["kernel_1"], padding="same")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=p["pool_1"])(x)
    x = keras.layers.Dropout(p["conv_dropout"])(x)

    x = keras.layers.Conv1D(p["filters_2"], kernel_size=p["kernel_2"], padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=p["pool_2"])(x)
    x = keras.layers.Dropout(p["conv_dropout"])(x)

    x = keras.layers.Conv1D(p["filters_3"], kernel_size=p["kernel_3"], padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(p["conv_dropout"])(x)
    return x


def _callbacks(p):
    from tensorflow import keras
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=p["patience"], restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, p["patience"] // 2), min_lr=1e-6, verbose=0,
        ),
    ]


class CNNClassifier:
    """Temporal 1D CNN classifier on raw sensor windows."""

    name = "CNN_1D_classifier"

    def __init__(self, params: dict, **overrides):
        self._params = {**params, **overrides}
        self._model = None
        self._label_encoder = LabelEncoder()

    def requires_scaling(self) -> bool:
        return True

    def _build_model(self, n_timesteps: int, n_channels: int, n_classes: int):
        from tensorflow import keras
        p = self._params

        inp = keras.Input(shape=(n_timesteps, n_channels))
        x = _conv_backbone(inp, p)
        x = keras.layers.Dense(p["dense_1"], activation="relu")(x)
        x = keras.layers.Dropout(p["dense_dropout"])(x)
        out = keras.layers.Dense(n_classes, activation="softmax")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, patience: int = None) -> None:
        p = dict(self._params)
        if epochs is not None:
            p["epochs"] = epochs
        if patience is not None:
            p["patience"] = patience

        _set_seeds(p["random_state"])

        y_enc = self._label_encoder.fit_transform(y).astype(np.int32)
        n_classes = len(self._label_encoder.classes_)
        n_t, n_c = X.shape[1], X.shape[2]

        rng = np.random.RandomState(p["random_state"])
        idx = rng.permutation(len(X))
        X, y_enc = X[idx], y_enc[idx]

        weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_enc)
        class_weight_dict = dict(enumerate(weights))

        self._model = self._build_model(n_t, n_c, n_classes)
        self._model.fit(
            X, y_enc,
            epochs=p["epochs"], batch_size=p["batch_size"], validation_split=p["val_split"],
            class_weight=class_weight_dict, callbacks=_callbacks(p), verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict(X, verbose=0)
        idx = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0)


class CNNRegressor:
    """Temporal 1D CNN regressor on raw sensor windows. Predicts in scaled
    target space -- the caller is responsible for scaling y before fit()
    and inverting predictions after predict()."""

    name = "CNN_1D_regressor"

    def __init__(self, params: dict, **overrides):
        self._params = {**params, **overrides}
        self._model = None

    def requires_scaling(self) -> bool:
        return True

    def _build_model(self, n_timesteps: int, n_channels: int):
        from tensorflow import keras
        p = self._params

        inp = keras.Input(shape=(n_timesteps, n_channels))
        x = _conv_backbone(inp, p)
        x = keras.layers.Dense(p["dense_1"], activation="relu")(x)
        x = keras.layers.Dropout(p["dense_dropout"])(x)
        out = keras.layers.Dense(1, activation="linear")(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit(self, X: np.ndarray, y_scaled: np.ndarray, epochs: int = None, patience: int = None) -> None:
        p = dict(self._params)
        if epochs is not None:
            p["epochs"] = epochs
        if patience is not None:
            p["patience"] = patience

        _set_seeds(p["random_state"])
        n_t, n_c = X.shape[1], X.shape[2]

        rng = np.random.RandomState(p["random_state"])
        idx = rng.permutation(len(X))
        X, y_scaled = X[idx], y_scaled[idx]

        self._model = self._build_model(n_t, n_c)
        self._model.fit(
            X, y_scaled,
            epochs=p["epochs"], batch_size=p["batch_size"], validation_split=p["val_split"],
            callbacks=_callbacks(p), verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).reshape(-1)
