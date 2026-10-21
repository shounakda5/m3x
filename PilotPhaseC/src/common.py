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
