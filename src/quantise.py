"""
Step 5.1 — Quantisation.

Convert the trained model to TensorFlow Lite, then to 8-bit integers.

IMPORTANT: we measure THREE sizes, not two.
  Keras  -> TFLite float32  : this is CONVERSION overhead removal
  TFLite float32 -> int8    : this is the REAL quantisation win
Reporting only Keras -> int8 would overstate the effect of quantisation.
"""

import numpy as np
import tensorflow as tf
import os

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"


def evaluate_tflite(tflite_path, X_test, y_test):
    """Run a TFLite model over the test set and return accuracy."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    correct = 0
    for i in range(len(X_test)):
        x = X_test[i:i+1].astype(inp["dtype"])

        # An int8 model expects int8 input. The converter gives us the
        # scale and zero-point needed to map floats onto that integer grid.
        if inp["dtype"] == np.int8:
            scale, zero_point = inp["quantization"]
            x = (X_test[i:i+1] / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(out["index"])

        if np.argmax(pred) == y_test[i]:
            correct += 1

    return correct / len(X_test)


if __name__ == "__main__":

    d = np.load(DATA_PATH)
    X_train = d["X_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    model = tf.keras.models.load_model(f"{MODEL_DIR}/baseline.keras")

    keras_kb = os.path.getsize(f"{MODEL_DIR}/baseline.keras") / 1024
    _, keras_acc = model.evaluate(X_test, y_test, verbose=0)

    # ---------------------------------------------------------------
    # 1. Convert to TFLite float32 — NO quantisation.
    #    This isolates how much is just conversion overhead.
    # ---------------------------------------------------------------
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = conv.convert()

    f32_path = f"{MODEL_DIR}/model_float32.tflite"
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)

    f32_kb = os.path.getsize(f32_path) / 1024
    f32_acc = evaluate_tflite(f32_path, X_test, y_test)

    # ---------------------------------------------------------------
    # 2. Convert to TFLite int8 — FULL quantisation.
    #
    #    To quantise, TFLite needs to know the typical RANGE of values
    #    flowing through the network. It learns that by running a few
    #    hundred real examples through — the "representative dataset".
    #    We use TRAINING data for this. Using test data would be a leak.
    # ---------------------------------------------------------------
    def representative_dataset():
        for i in range(300):
            yield [X_train[i:i+1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    tflite_int8 = conv.convert()

    int8_path = f"{MODEL_DIR}/model_int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)

    int8_kb = os.path.getsize(int8_path) / 1024
    int8_acc = evaluate_tflite(int8_path, X_test, y_test)

    # ---------------------------------------------------------------
    # Report — honestly
    # ---------------------------------------------------------------
    print()
    print("=" * 62)
    print("QUANTISATION RESULTS")
    print("=" * 62)
    print(f"{'Model':<22} {'Size (KB)':>10} {'Accuracy':>10}")
    print("-" * 62)
    print(f"{'Keras baseline':<22} {keras_kb:>10.1f} {keras_acc*100:>9.2f}%")
    print(f"{'TFLite float32':<22} {f32_kb:>10.1f} {f32_acc*100:>9.2f}%")
    print(f"{'TFLite int8':<22} {int8_kb:>10.1f} {int8_acc*100:>9.2f}%")
    print("-" * 62)
    print()
    print(f"Keras -> TFLite float32 : {keras_kb/f32_kb:.1f}x smaller")
    print("    ^ this is CONVERSION overhead, not quantisation")
    print()
    print(f"TFLite float32 -> int8  : {f32_kb/int8_kb:.1f}x smaller")
    print(f"    ^ THIS is the real quantisation win")
    print(f"    accuracy cost: {(f32_acc - int8_acc)*100:+.2f} percentage points")
    print()
    print(f"Keras -> int8 overall   : {keras_kb/int8_kb:.1f}x smaller")