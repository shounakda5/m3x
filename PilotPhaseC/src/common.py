"""
common.py -- Shared, lineage/task-agnostic helpers for Pilot Phase C:
per-channel input scaling, target scaling (regression), TFLite export, and
INT8 post-training quantization with a tensor-arena (RAM) estimate.

Self-contained duplicate of the same conventions used elsewhere in this
project (root train_cnn.py, pilot_b/src/quantize.py) -- re-implemented here
rather than imported, per this phase's self-containment requirement.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Per-channel input scaling (z-normalisation, fit on train fold only)
# ---------------------------------------------------------------------------

def fit_channel_scaler(X_train: np.ndarray):
    """mean/std over all training windows and timesteps. X_train: (N, T, C)."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_channel_scaler(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Target scaling (regression only, fit on train fold only)
# ---------------------------------------------------------------------------

def fit_target_scaler(y_train: np.ndarray):
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0:
        std = 1.0
    return mean, std


def apply_target_scaler(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (y - mean) / std


def invert_target_scaler(y_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_scaled * std + mean


# ---------------------------------------------------------------------------
# Survey-equal training weights + group-level metrics
# ---------------------------------------------------------------------------

def survey_equal_weights(groups: np.ndarray) -> np.ndarray:
    """One unit of loss per Survey_ID: w_i = 1 / (# windows in that group)."""
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (1.0 / counts[inv]).astype(np.float32)


def period_weights_for_lineage(
    period: np.ndarray | None, lineage: str,
) -> np.ndarray:
    from PilotPhaseC.config import DEFAULT_PERIOD_WEIGHT, LINEAGE_PERIOD_WEIGHTS

    n = len(period) if period is not None else 0
    if period is None or n == 0:
        return np.ones(0, dtype=np.float32)
    mapping = LINEAGE_PERIOD_WEIGHTS.get(lineage, {})
    return np.array(
        [float(mapping.get(str(p), DEFAULT_PERIOD_WEIGHT)) for p in period],
        dtype=np.float32,
    )


def classifier_sample_weights(
    groups: np.ndarray,
    y: np.ndarray,
    period: np.ndarray | None = None,
    lineage: str | None = None,
) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    from PilotPhaseC.config import (
        CLASSES,
        CLASSIFIER_SURVEY_EQUAL_WEIGHTS,
        CLASSIFIER_CLASS_BALANCED_WEIGHTS,
    )

    sw = (
        survey_equal_weights(groups)
        if CLASSIFIER_SURVEY_EQUAL_WEIGHTS
        else np.ones(len(groups), dtype=np.float32)
    )
    if CLASSIFIER_CLASS_BALANCED_WEIGHTS:
        le = LabelEncoder()
        le.classes_ = np.array(CLASSES, dtype=object)
        y_enc = le.transform(y)
        present = np.unique(y_enc)
        cw_present = compute_class_weight("balanced", classes=present, y=y_enc)
        cw_full = np.ones(len(CLASSES), dtype=np.float64)
        for cls_idx, w in zip(present, cw_present):
            cw_full[int(cls_idx)] = w
        sw = sw * cw_full[y_enc]
    if lineage and period is not None and len(period) == len(sw):
        sw = sw * period_weights_for_lineage(period, lineage)
    return sw.astype(np.float32)


def _class_preds_from_log_proba(log_p: np.ndarray, classes: np.ndarray, biases: np.ndarray | None = None) -> np.ndarray:
    scores = log_p + biases.reshape(1, -1) if biases is not None else log_p
    return classes[np.argmax(scores, axis=1)]


def apply_logit_biases(proba: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """biases shape (n_classes,) added to log-probability before argmax."""
    from PilotPhaseC.config import CLASSES

    classes = np.asarray(CLASSES, dtype=object)
    log_p = np.log(np.clip(proba, 1e-12, 1.0))
    return _class_preds_from_log_proba(log_p, classes, biases)


def predict_with_logit_biases(model, X: np.ndarray, biases: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return apply_logit_biases(proba, biases)


def tune_logit_biases(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str] | None = None,
    n_random: int = 4000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Random search over per-class logit biases to maximize macro-F1 (OOF)."""
    from sklearn.metrics import f1_score

    from PilotPhaseC.config import CLASSES as DEFAULT_CLASSES

    classes = np.asarray(classes or DEFAULT_CLASSES, dtype=object)
    log_p = np.log(np.clip(proba, 1e-12, 1.0))

    rng = np.random.RandomState(random_state)
    best_b = np.zeros(len(classes), dtype=np.float64)
    best_score = f1_score(
        y_true, _class_preds_from_log_proba(log_p, classes),
        average="macro", labels=classes, zero_division=0,
    )

    for _ in range(n_random):
        b = rng.uniform(-1.5, 1.5, size=len(classes))
        pred = _class_preds_from_log_proba(log_p, classes, b)
        score = f1_score(y_true, pred, average="macro", labels=classes, zero_division=0)
        if score > best_score:
            best_score, best_b = score, b.copy()

    # Coordinate refinement
    step = 0.25
    for _ in range(3):
        improved = True
        while improved:
            improved = False
            for j in range(len(classes)):
                for delta in (-step, step):
                    b_try = best_b.copy()
                    b_try[j] += delta
                    pred = _class_preds_from_log_proba(log_p, classes, b_try)
                    score = f1_score(y_true, pred, average="macro", labels=classes, zero_division=0)
                    if score > best_score + 1e-6:
                        best_score, best_b = score, b_try
                        improved = True
        step *= 0.5

    return best_b.astype(np.float32), float(best_score)


def regressor_sample_weights(
    groups: np.ndarray,
    period: np.ndarray | None = None,
    lineage: str | None = None,
) -> np.ndarray:
    from PilotPhaseC.config import REGRESSOR_SURVEY_EQUAL_WEIGHTS

    sw = (
        survey_equal_weights(groups)
        if REGRESSOR_SURVEY_EQUAL_WEIGHTS
        else np.ones(len(groups), dtype=np.float32)
    )
    if lineage and period is not None and len(period) == len(sw):
        sw = sw * period_weights_for_lineage(period, lineage)
    return sw.astype(np.float32)


def split_train_val_by_group(
    groups: np.ndarray, val_fraction: float, random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, val) holding out whole Survey_ID groups."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    if n_val >= len(unique):
        n_val = max(1, len(unique) // 5)
    rng = np.random.RandomState(random_state)
    val_groups = set(rng.choice(unique, size=n_val, replace=False))
    val_mask = np.array([g in val_groups for g in groups], dtype=bool)
    return ~val_mask, val_mask


def effective_classification_folds(y: np.ndarray, groups: np.ndarray, n_folds_max: int) -> int:
    import pandas as pd

    gt = pd.DataFrame({"g": groups, "y": y}).drop_duplicates("g")
    min_groups = int(gt.groupby("y").size().min())
    if min_groups < 2:
        return max(2, min(n_folds_max, min_groups if min_groups > 0 else 2))
    return max(2, min(n_folds_max, min_groups))


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def export_tflite(keras_model, out_path) -> int:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    return len(tflite_bytes)


# ---------------------------------------------------------------------------
# INT8 post-training quantization
# ---------------------------------------------------------------------------

def quantize_int8(keras_model, X_calib: np.ndarray, random_state: int, n_calib: int = 200) -> bytes:
    import tensorflow as tf

    rng = np.random.default_rng(random_state)
    n_calib = min(n_calib, len(X_calib))
    calib = X_calib[rng.choice(len(X_calib), n_calib, replace=False)]

    def representative_dataset():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def estimate_tensor_arena(tflite_bytes: bytes) -> dict:
    """Best-effort RAM estimate via a greedy/linear memory planner over the
    flatbuffer's own op execution order and tensor live ranges -- a
    reasonable lower bound, not a measurement on real hardware."""
    from tensorflow.lite.python import schema_py_generated as schema_fb

    model = schema_fb.Model.GetRootAsModel(tflite_bytes, 0)
    graph = model.Subgraphs(0)

    n_tensors = graph.TensorsLength()
    tensor_bytes = np.zeros(n_tensors, dtype=np.int64)
    is_constant = np.zeros(n_tensors, dtype=bool)

    dtype_sizes = {0: 4, 9: 1, 6: 1, 3: 4, 1: 2}  # FLOAT32, INT8 (x2 schema variants), INT32, FLOAT16

    for i in range(n_tensors):
        t = graph.Tensors(i)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        n_elem = int(np.prod(shape)) if shape else 1
        tensor_bytes[i] = dtype_sizes.get(t.Type(), 1) * n_elem

        buf_idx = t.Buffer()
        buf = model.Buffers(buf_idx) if buf_idx < model.BuffersLength() else None
        is_constant[i] = buf is not None and buf.DataLength() > 0

    n_ops = graph.OperatorsLength()
    first_use = np.full(n_tensors, n_ops, dtype=np.int64)
    last_use = np.full(n_tensors, -1, dtype=np.int64)

    graph_inputs = set(graph.Inputs(j) for j in range(graph.InputsLength()))
    graph_outputs = set(graph.Outputs(j) for j in range(graph.OutputsLength()))
    for t_idx in graph_inputs:
        first_use[t_idx] = -1

    for op_i in range(n_ops):
        op = graph.Operators(op_i)
        for j in range(op.InputsLength()):
            t_idx = op.Inputs(j)
            if t_idx < 0:
                continue
            last_use[t_idx] = max(last_use[t_idx], op_i)
            first_use[t_idx] = min(first_use[t_idx], op_i)
        for j in range(op.OutputsLength()):
            t_idx = op.Outputs(j)
            if t_idx < 0:
                continue
            first_use[t_idx] = min(first_use[t_idx], op_i)

    for t_idx in graph_outputs:
        last_use[t_idx] = n_ops

    peak_bytes = 0
    for op_i in range(-1, n_ops + 1):
        live = sum(
            tensor_bytes[t_idx]
            for t_idx in range(n_tensors)
            if not is_constant[t_idx] and first_use[t_idx] <= op_i <= last_use[t_idx]
        )
        peak_bytes = max(peak_bytes, live)

    largest_activation = int(max(
        (tensor_bytes[i] for i in range(n_tensors) if not is_constant[i]), default=0,
    ))
    total_weight_bytes = int(tensor_bytes[is_constant].sum())

    return dict(
        estimated_arena_bytes=int(peak_bytes),
        largest_single_activation_bytes=largest_activation,
        total_weight_bytes=total_weight_bytes,
        n_tensors=n_tensors,
        n_ops=n_ops,
    )
