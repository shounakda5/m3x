"""
1D CNN classifier for raw M3X physiological time-series data.

Input shape  : (batch, RAW_WINDOW_SAMPLES, len(RAW_CHANNELS)) = (N, 112, 9)
               7 raw channels (LED1/2/3, Temp, AccX/Y/Z) + 2 derived
               cross-LED differences (LED3-LED1, LED3-LED2)

Architecture
------------
  Block 1: Conv1D(32, k=7) → BN → ReLU → MaxPool(2) → Dropout
  Block 2: Conv1D(64, k=5) → BN → ReLU → MaxPool(2) → Dropout
  Block 3: Conv1D(128, k=3) → BN → ReLU → GlobalAvgPool → Dropout
  Head   : Dense(64, ReLU) → Dropout → Dense(4, Softmax)

Temporal kernels are tuned to the 25 Hz sampling rate:
  k=7  ≈ 280 ms  (captures one PPG cycle at ~70 bpm)
  k=5  ≈ 200 ms
  k=3  ≈ 120 ms  (fine-grained feature refinement after pooling)

Design notes
------------
* Accepts raw numpy arrays (N, T, C) — NOT pandas DataFrames.
* requires_scaling() → True: per-channel z-normalisation is applied by the
  training script on training data only and stored as a scaler artefact.
* Each fit() call rebuilds the model from scratch for safe CV reuse.
* Class weights are computed inside fit() to handle label imbalance.
"""

import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import CNN_PARAMS


class CNNClassifier:
    """Temporal 1D CNN on raw sensor windows."""

    name = "CNN_1D"

    def __init__(self, **overrides):
        self._params = {**CNN_PARAMS, **overrides}
        self._model  = None
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
        tf.random.set_seed(seed)

    def _build_model(self, n_timesteps: int, n_channels: int, n_classes: int):
        import tensorflow as tf
        from tensorflow import keras
        p = self._params

        inp = keras.Input(shape=(n_timesteps, n_channels))

        # --- Conv block 1 ---
        x = keras.layers.Conv1D(p["filters_1"], kernel_size=p["kernel_1"], padding="same")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(pool_size=p["pool_1"])(x)
        x = keras.layers.Dropout(p["conv_dropout"])(x)

        # --- Conv block 2 ---
        x = keras.layers.Conv1D(p["filters_2"], kernel_size=p["kernel_2"], padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(pool_size=p["pool_2"])(x)
        x = keras.layers.Dropout(p["conv_dropout"])(x)

        # --- Conv block 3 → global pooling ---
        x = keras.layers.Conv1D(p["filters_3"], kernel_size=p["kernel_3"], padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dropout(p["conv_dropout"])(x)

        # --- FC head ---
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train on pre-scaled windows.

        Parameters
        ----------
        X : (N, T, C) float32  — already z-normalised per channel
        y : (N,) str           — string class labels
        """
        import tensorflow as tf
        from tensorflow import keras

        self._set_seeds()
        p = self._params

        y_enc     = self._label_encoder.fit_transform(y).astype(np.int32)
        n_classes = len(self._label_encoder.classes_)
        n_t, n_c  = X.shape[1], X.shape[2]

        # Shuffle before val split so the held-out slice is representative
        rng = np.random.RandomState(p["random_state"])
        idx = rng.permutation(len(X))
        X, y_enc = X[idx], y_enc[idx]

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_enc,
        )
        class_weight_dict = dict(enumerate(weights))

        self._model = self._build_model(n_t, n_c, n_classes)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=p["patience"],
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
            X, y_enc,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            validation_split=p["val_split"],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._model.predict(X, verbose=0)
        idx   = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0)
