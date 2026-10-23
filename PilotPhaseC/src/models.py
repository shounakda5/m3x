"""
models.py -- 1D CNN classifier and regressor for Pilot Phase C raw sensor
windows. Self-contained (does not import the root project's src/models/
CNNClassifier) per this phase's self-containment requirement, though the
architecture family is deliberately the same as the rest of the project's
raw-CNN models for consistency.

Input shape: (batch, WINDOW_SAMPLES, n_channels) — n_channels from the
loaded dataset (10 with derived LED3_LED1 / LED3_LED2 channels).

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = None,
        patience: int = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        val_group_fraction: float | None = None,
    ) -> None:
        from PilotPhaseC.config import VAL_GROUP_FRACTION
        from PilotPhaseC.src.common import split_train_val_by_group

        p = dict(self._params)
        if epochs is not None:
            p["epochs"] = epochs
        if patience is not None:
            p["patience"] = patience

        from PilotPhaseC.config import CLASSES

        _set_seeds(p["random_state"])

        self._label_encoder.classes_ = np.array(CLASSES, dtype=object)
        y_enc = self._label_encoder.transform(y).astype(np.int32)
        n_classes = len(CLASSES)
        n_t, n_c = X.shape[1], X.shape[2]

        sw = sample_weight
        if groups is not None and len(np.unique(groups)) >= 3:
            tr_m, va_m = split_train_val_by_group(
                groups, val_group_fraction or VAL_GROUP_FRACTION, p["random_state"],
            )
            X_tr, y_tr = X[tr_m], y_enc[tr_m]
            X_va, y_va = X[va_m], y_enc[va_m]
            sw_tr = sw[tr_m] if sw is not None else None
            val_data = (X_va, y_va)
            fit_X, fit_y, fit_sw = X_tr, y_tr, sw_tr
        else:
            val_data = None
            fit_X, fit_y, fit_sw = X, y_enc, sw

        rng = np.random.RandomState(p["random_state"])
        idx = rng.permutation(len(fit_X))
        fit_X, fit_y = fit_X[idx], fit_y[idx]
        if fit_sw is not None:
            fit_sw = fit_sw[idx]

        self._model = self._build_model(n_t, n_c, n_classes)
        fit_kw = dict(
            epochs=p["epochs"], batch_size=p["batch_size"],
            callbacks=_callbacks(p), verbose=0,
        )
        if fit_sw is not None:
            fit_kw["sample_weight"] = fit_sw
        else:
            from sklearn.utils.class_weight import compute_class_weight
            present = np.unique(fit_y)
            cw_present = compute_class_weight("balanced", classes=present, y=fit_y)
            fit_kw["class_weight"] = {int(c): float(w) for c, w in zip(present, cw_present)}
        if val_data is not None:
            fit_kw["validation_data"] = val_data
        else:
            fit_kw["validation_split"] = p["val_split"]
        self._model.fit(fit_X, fit_y, **fit_kw)

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

    def fit(
        self,
        X: np.ndarray,
        y_scaled: np.ndarray,
        epochs: int = None,
        patience: int = None,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        val_group_fraction: float | None = None,
    ) -> None:
        from PilotPhaseC.config import VAL_GROUP_FRACTION
        from PilotPhaseC.src.common import split_train_val_by_group

        p = dict(self._params)
        if epochs is not None:
            p["epochs"] = epochs
        if patience is not None:
            p["patience"] = patience

        _set_seeds(p["random_state"])
        n_t, n_c = X.shape[1], X.shape[2]

        sw = sample_weight
        if groups is not None and len(np.unique(groups)) >= 3:
            tr_m, va_m = split_train_val_by_group(
                groups, val_group_fraction or VAL_GROUP_FRACTION, p["random_state"],
            )
            fit_X, fit_y = X[tr_m], y_scaled[tr_m]
            val_data = (X[va_m], y_scaled[va_m])
            fit_sw = sw[tr_m] if sw is not None else None
        else:
            val_data = None
            fit_X, fit_y, fit_sw = X, y_scaled, sw

        rng = np.random.RandomState(p["random_state"])
        idx = rng.permutation(len(fit_X))
        fit_X, fit_y = fit_X[idx], fit_y[idx]
        if fit_sw is not None:
            fit_sw = fit_sw[idx]

        self._model = self._build_model(n_t, n_c)
        fit_kw = dict(
            epochs=p["epochs"], batch_size=p["batch_size"],
            callbacks=_callbacks(p), verbose=0,
        )
        if fit_sw is not None:
            fit_kw["sample_weight"] = fit_sw
        if val_data is not None:
            fit_kw["validation_data"] = val_data
        else:
            fit_kw["validation_split"] = p["val_split"]
        self._model.fit(fit_X, fit_y, **fit_kw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).reshape(-1)
