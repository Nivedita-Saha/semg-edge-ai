"""
Step 6.1 — Benchmark every model, fairly.

The rule: every model goes through the SAME conversion, the SAME
interpreter, and the SAME timing method. Otherwise we are measuring
plumbing differences, not model differences.

Two confounds this fixes:
  - Keras .predict() latency is dominated by Python overhead, not compute
  - .keras and .h5 files carry different metadata, so their sizes are
    not comparable

Measured here: accuracy, size on disk (TFLite), and median latency.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import json

from model import build_model
from distil import build_student

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
RESULTS = "results"

d = np.load(DATA_PATH)
X_train = d["X_train"]
X_test, y_test = d["X_test"], d["y_test"]


def to_tflite(model, path, quantise=False):
    """Convert a Keras model to TFLite. Optionally quantise to int8."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantise:
        def rep_data():
            for i in range(300):
                yield [X_train[i:i+1].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_data
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8

    tflite = conv.convert()
    with open(path, "wb") as f:
        f.write(tflite)
    return path


def benchmark_tflite(path, n_timing_runs=200):
    """Accuracy, size, and median latency — all through one interpreter."""
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    is_int8 = inp["dtype"] == np.int8
    if is_int8:
        scale, zp = inp["quantization"]

    def prep(i):
        if is_int8:
            return (X_test[i:i+1] / scale + zp).astype(np.int8)
        return X_test[i:i+1].astype(np.float32)

    # --- accuracy ---
    correct = 0
    for i in range(len(X_test)):
        interp.set_tensor(inp["index"], prep(i))
        interp.invoke()
        if np.argmax(interp.get_tensor(out["index"])) == y_test[i]:
            correct += 1
    accuracy = correct / len(X_test)

    # --- latency: median of many single-sample inferences ---
    # Median, not mean: one OS hiccup would skew a mean badly.
    x0 = prep(0)
    for _ in range(20):                      # warm-up, not timed
        interp.set_tensor(inp["index"], x0)
        interp.invoke()

    times = []
    for _ in range(n_timing_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x0)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "accuracy": accuracy * 100,
        "size_kb": os.path.getsize(path) / 1024,
        "latency_ms": float(np.median(times)),
    }


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)

    # --- rebuild each model and load its weights ---
    baseline = build_model(X_train.shape[1:])
    baseline.set_weights([v for v in np.load(f"{MODEL_DIR}/baseline_weights.npz").values()])

    pruned = build_model(X_train.shape[1:])
    pruned.set_weights([v for v in np.load(f"{MODEL_DIR}/pruned_weights.npz").values()])

    student_core = build_student(X_train.shape[1:])
    student_core.set_weights([v for v in np.load(f"{MODEL_DIR}/student_weights.npz").values()])
    student = keras.Sequential([student_core, keras.layers.Softmax()])
    student.build((None,) + X_train.shape[1:])

    models = [
        ("Baseline",              baseline, False, baseline.count_params()),
        ("Baseline + int8",       baseline, True,  baseline.count_params()),
        ("Pruned (48%)",          pruned,   False, pruned.count_params()),
        ("Pruned + int8",         pruned,   True,  pruned.count_params()),
        ("Student (distilled)",   student,  False, student_core.count_params()),
        ("Student + int8",        student,  True,  student_core.count_params()),
    ]

    results = []
    for name, model, quant, params in models:
        tag = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "").replace("%", "")
        path = f"{MODEL_DIR}/bench_{tag}.tflite"
        to_tflite(model, path, quantise=quant)
        r = benchmark_tflite(path)
        r["name"] = name
        r["params"] = params
        results.append(r)
        print(f"  done: {name}")

    with open(f"{RESULTS}/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- report ---
    print()
    print("=" * 78)
    print("BENCHMARK — all models, same pipeline, same measurement")
    print("=" * 78)
    print(f"{'Model':<22} {'Params':>8} {'Size (KB)':>11} {'Latency (ms)':>13} {'Accuracy':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<22} {r['params']:>8,} {r['size_kb']:>11.1f} "
              f"{r['latency_ms']:>13.3f} {r['accuracy']:>9.2f}%")
    print("-" * 78)

    base = results[0]
    print()
    print("Relative to baseline:")
    for r in results[1:]:
        print(f"  {r['name']:<22} {base['size_kb']/r['size_kb']:>5.1f}x smaller   "
              f"{base['latency_ms']/r['latency_ms']:>5.2f}x faster   "
              f"{r['accuracy']-base['accuracy']:+6.2f} pp accuracy")

    print(f"\nSaved: {RESULTS}/benchmark.json")
