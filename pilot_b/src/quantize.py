"""
quantize.py -- Post-training full-integer INT8 quantization of
the Pilot B generalized CNN, for MCU deployment.

Follows the same recipe already used for the original cnn_raw_baseline model
(see INFERENCE_GUIDE.md): 200 calibration windows drawn from the training
pool, full-integer quantization with int8 input/output.

In addition to file size (the flash-storage footprint), this script gives a
best-effort estimate of the runtime **tensor arena** (RAM) requirement --
the working memory TFLite Micro needs at inference time, which is a
different and usually much smaller number than the flash file size, driven
by activation tensor sizes and their live ranges rather than weight count.

The estimate is computed by parsing the .tflite flatbuffer directly (the
same schema TFLite Micro itself uses) and simulating a greedy/linear memory
planner: each non-constant tensor is "born" when produced and "freed" after
its last consumer runs; the peak simultaneously-live byte count across the
execution order is the estimated minimum arena. This is optimistic relative
to a real allocator (which has alignment/fragmentation overhead), so treat
it as a lower bound -- the only way to get an exact number is to build and
run on the actual target with TFLite Micro's RecordingMicroInterpreter.

Usage
-----
    python pilot_b/src/quantize.py
    python pilot_b/src/quantize.py --output_dir pilot_b/results/cnn_generalized
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CLASSES
from pilot_b.config import PILOT_B_RESULTS_DIR, PILOT_B_RANDOM_STATE
from pilot_b.src.loader import load_pilot_b_dataset
from pilot_b.src.train_generalized import split_aisha_holdout, uniform_sample_by_participant_total


# ---------------------------------------------------------------------------
# Step 1: rebuild the exact calibration + Aisha-holdout sets used at training
# time (same RNG seed -> same split/sample), so INT8 accuracy is measured
# against genuinely held-out data.
# ---------------------------------------------------------------------------

def _rebuild_pool_and_holdout():
    X, y, groups, participants, _ = load_pilot_b_dataset()
    rng = np.random.RandomState(PILOT_B_RANDOM_STATE)

    is_holdout, _ = split_aisha_holdout(y, groups, participants, rng)
    X_aisha_holdout, y_aisha_holdout = X[is_holdout], y[is_holdout]

    pool_mask = ~is_holdout
    X_pool, y_pool = X[pool_mask], y[pool_mask]
    groups_pool, participants_pool = groups[pool_mask], participants[pool_mask]

    balance_mask, _ = uniform_sample_by_participant_total(y_pool, groups_pool, participants_pool, rng)
    X_train_pool = X_pool[balance_mask]

    return X_train_pool, X_aisha_holdout, y_aisha_holdout


# ---------------------------------------------------------------------------
# Step 2: INT8 conversion
# ---------------------------------------------------------------------------

def quantize(model, X_train_scaled: np.ndarray, n_calib: int = 200):
    import tensorflow as tf

    rng = np.random.default_rng(PILOT_B_RANDOM_STATE)
    n_calib = min(n_calib, len(X_train_scaled))
    calib = X_train_scaled[rng.choice(len(X_train_scaled), n_calib, replace=False)]

    def representative_dataset():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


# ---------------------------------------------------------------------------
# Step 3: INT8 accuracy check on Aisha's held-out slice
# ---------------------------------------------------------------------------

def evaluate_int8(tflite_bytes: bytes, X_aisha_scaled: np.ndarray, y_aisha: np.ndarray):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    inp_scale, inp_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]

    x_i8 = np.clip(np.round(X_aisha_scaled / inp_scale + inp_zp), -128, 127).astype(np.int8)

    preds = []
    for i in range(len(x_i8)):
        interpreter.set_tensor(inp_d["index"], x_i8[i:i + 1])
        interpreter.invoke()
        out_i8 = interpreter.get_tensor(out_d["index"])[0]
        probs = (out_i8.astype(np.float32) - out_zp) * out_scale
        preds.append(np.argmax(probs))

    return np.array(preds), dict(
        inp_scale=float(inp_scale), inp_zp=int(inp_zp),
        out_scale=float(out_scale), out_zp=int(out_zp),
    )


# ---------------------------------------------------------------------------
# Step 4: best-effort tensor-arena (RAM) estimate via flatbuffer parsing
# ---------------------------------------------------------------------------

def estimate_tensor_arena(tflite_bytes: bytes) -> dict:
    from tensorflow.lite.python import schema_py_generated as schema_fb

    model = schema_fb.Model.GetRootAsModel(tflite_bytes, 0)
    graph = model.Subgraphs(0)

    n_tensors = graph.TensorsLength()
    tensor_bytes = np.zeros(n_tensors, dtype=np.int64)
    is_constant = np.zeros(n_tensors, dtype=bool)

    dtype_sizes = {
        0: 4,   # FLOAT32
        9: 1,   # INT8
        6: 1,   # INT8 (some schema versions)
        3: 4,   # INT32
        1: 2,   # FLOAT16
    }

    for i in range(n_tensors):
        t = graph.Tensors(i)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        n_elem = int(np.prod(shape)) if shape else 1
        size = dtype_sizes.get(t.Type(), 1) * n_elem
        tensor_bytes[i] = size

        buf_idx = t.Buffer()
        buf = model.Buffers(buf_idx) if buf_idx < model.BuffersLength() else None
        has_data = buf is not None and buf.DataLength() > 0
        is_constant[i] = has_data

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
    peak_at_op = -1
    for op_i in range(-1, n_ops + 1):
        live = 0
        for t_idx in range(n_tensors):
            if is_constant[t_idx]:
                continue
            if first_use[t_idx] <= op_i <= last_use[t_idx]:
                live += tensor_bytes[t_idx]
        if live > peak_bytes:
            peak_bytes = live
            peak_at_op = op_i

    largest_activation = int(max(
        (tensor_bytes[i] for i in range(n_tensors) if not is_constant[i]),
        default=0,
    ))
    total_weight_bytes = int(tensor_bytes[is_constant].sum())

    return dict(
        estimated_arena_bytes=int(peak_bytes),
        peak_at_op_index=int(peak_at_op),
        largest_single_activation_bytes=largest_activation,
        total_weight_bytes=total_weight_bytes,
        n_tensors=n_tensors,
        n_ops=n_ops,
    )


def main():
    parser = argparse.ArgumentParser(description="INT8-quantize the Pilot B generalized CNN.")
    parser.add_argument("--output_dir", default=str(PILOT_B_RESULTS_DIR / "cnn_generalized"))
    args = parser.parse_args()
    out = Path(args.output_dir)

    print("Loading final float32 model + scaler ...")
    model = joblib.load(out / "cnn_1d_model.pkl")
    scaler = np.load(out / "cnn_1d_scaler.npz")
    mean, std = scaler["mean"], scaler["std"]

    print("Rebuilding calibration pool + Aisha holdout (same split as training) ...")
    X_train_pool, X_aisha_holdout, y_aisha_holdout = _rebuild_pool_and_holdout()
    X_train_scaled = ((X_train_pool - mean) / std).astype(np.float32)
    X_aisha_scaled = ((X_aisha_holdout - mean) / std).astype(np.float32)

    print("Converting to full-integer INT8 TFLite ...")
    tflite_i8 = quantize(model, X_train_scaled, n_calib=200)
    int8_path = out / "cnn_1d_model_int8.tflite"
    int8_path.write_bytes(tflite_i8)

    float32_path = out / "cnn_1d_model.tflite"
    f32_size = float32_path.stat().st_size if float32_path.exists() else None
    i8_size = len(tflite_i8)

    print("\nEvaluating INT8 model on Aisha's held-out slice ...")
    y_pred_idx, quant_params = evaluate_int8(tflite_i8, X_aisha_scaled, y_aisha_holdout)
    # LabelEncoder used by CNNClassifier is alphabetical; reconstruct that mapping
    alphabetical_classes = sorted(CLASSES)
    y_pred = np.array([alphabetical_classes[i] for i in y_pred_idx])

    int8_acc = accuracy_score(y_aisha_holdout, y_pred)
    int8_f1m = f1_score(y_aisha_holdout, y_pred, average="macro", labels=CLASSES, zero_division=0)

    print("\nEstimating tensor arena (RAM) via flatbuffer analysis ...")
    mem = estimate_tensor_arena(tflite_i8)

    report_lines = []
    report_lines.append("Pilot B generalized CNN -- INT8 quantization report\n")
    report_lines.append(f"Float32 file size : {f32_size/1024:.1f} KB" if f32_size else "Float32 file size : (not found)")
    report_lines.append(f"INT8 file size    : {i8_size/1024:.1f} KB  <- this is what gets flashed to the chip")
    report_lines.append("")
    report_lines.append(f"INT8 accuracy on Aisha holdout : {int8_acc:.4f}")
    report_lines.append(f"INT8 F1 macro on Aisha holdout : {int8_f1m:.4f}")
    report_lines.append("(compare against the float32 numbers in aisha_holdout_report.txt)")
    report_lines.append("")
    report_lines.append("Quantization parameters (hardcode these in firmware):")
    report_lines.append(f"  Input  scale={quant_params['inp_scale']:.8f}  zero_point={quant_params['inp_zp']}")
    report_lines.append(f"  Output scale={quant_params['out_scale']:.8f}  zero_point={quant_params['out_zp']}")
    report_lines.append("")
    report_lines.append("Estimated RAM (tensor arena) -- best-effort, see caveat below:")
    report_lines.append(f"  Estimated peak arena       : {mem['estimated_arena_bytes']/1024:.2f} KB")
    report_lines.append(f"  Largest single activation  : {mem['largest_single_activation_bytes']/1024:.2f} KB")
    report_lines.append(f"  Total weight bytes (flash) : {mem['total_weight_bytes']/1024:.2f} KB")
    report_lines.append(f"  ({mem['n_tensors']} tensors, {mem['n_ops']} ops)")
    report_lines.append("")
    report_lines.append(
        "CAVEAT: the arena estimate is computed by simulating a greedy/linear\n"
        "memory planner over the flatbuffer's own op execution order and tensor\n"
        "live ranges (same idea TFLite Micro's allocator uses), so it is a\n"
        "reasonable *lower bound* -- a real allocator adds alignment padding and\n"
        "some fixed interpreter overhead. Treat this as directional, not exact;\n"
        "confirm on the actual target by reading interpreter.arena_used_bytes()\n"
        "(or the RecordingMicroInterpreter's PrintAllocations()) after building\n"
        "with TFLite Micro on the real chip."
    )

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out / "int8_quantization_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\nSaved: {int8_path}")
    print(f"Saved: {out / 'int8_quantization_report.txt'}")


if __name__ == "__main__":
    main()
