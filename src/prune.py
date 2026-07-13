"""
Step 5.2 — Pruning.

Remove the least-useful connections (weights nearest zero), gradually,
while retraining so the network adapts.

Honest note: pruning's benefit here is SIZE, via compressibility.
It does NOT speed up inference on a standard CPU, which still multiplies
by zero. Real speedups need sparsity-aware hardware.
"""

# tensorflow_model_optimization requires Keras 2. This MUST be set
# before TensorFlow is imported, or the wrong Keras loads.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zipfile

from model import build_model

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
TARGET_SPARSITY = 0.50

tf.random.set_seed(42)
np.random.seed(42)


def zipped_size_kb(path):
    """Zeros only save space once compressed. Zipped size is the fair measure."""
    zip_path = path + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname=os.path.basename(path))
    kb = os.path.getsize(zip_path) / 1024
    os.remove(zip_path)
    return kb


def count_zeros(model):
    total, zeros = 0, 0
    for w in model.get_weights():
        total += w.size
        zeros += np.sum(w == 0)
    return zeros / total


if __name__ == "__main__":

    d = np.load(DATA_PATH)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    baseline = build_model(input_shape=X_train.shape[1:])
    wz = np.load(f"{MODEL_DIR}/baseline_weights.npz")
    baseline.set_weights([wz[k] for k in wz.files])

    _, base_acc = baseline.evaluate(X_test, y_test, verbose=0)
    base_sparsity = count_zeros(baseline)

    print(f"Baseline accuracy : {base_acc*100:.2f}%   <- must match 74.13%")
    print(f"Baseline sparsity : {base_sparsity*100:.1f}%")

    EPOCHS = 30
    BATCH = 32
    end_step = int(np.ceil(len(X_train) / BATCH)) * EPOCHS

    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_SPARSITY,
        begin_step=0,
        end_step=end_step,
    )

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        baseline, pruning_schedule=schedule
    )
    model_for_pruning.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nPruning to {TARGET_SPARSITY*100:.0f}% sparsity over {EPOCHS} epochs...\n")

    model_for_pruning.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=2,
    )

    pruned = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    pruned.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    _, pruned_acc = pruned.evaluate(X_test, y_test, verbose=0)
    pruned_sparsity = count_zeros(pruned)

    pruned_path = f"{MODEL_DIR}/pruned.h5"
    pruned.save(pruned_path)
    np.savez(f"{MODEL_DIR}/pruned_weights.npz", *pruned.get_weights())

    base_zip = zipped_size_kb(f"{MODEL_DIR}/baseline.keras")
    pruned_zip = zipped_size_kb(pruned_path)

    print()
    print("=" * 64)
    print("PRUNING RESULTS")
    print("=" * 64)
    print(f"{'Model':<20} {'Sparsity':>10} {'Zipped KB':>12} {'Accuracy':>10}")
    print("-" * 64)
    print(f"{'Baseline':<20} {base_sparsity*100:>9.1f}% {base_zip:>12.1f} {base_acc*100:>9.2f}%")
    print(f"{'Pruned (50%)':<20} {pruned_sparsity*100:>9.1f}% {pruned_zip:>12.1f} {pruned_acc*100:>9.2f}%")
    print("-" * 64)
    print()
    print(f"Size reduction  : {base_zip/pruned_zip:.2f}x  (zipped)")
    print(f"Accuracy change : {(pruned_acc - base_acc)*100:+.2f} percentage points")
    print()
    print("Sizes are ZIPPED - zeros only save space once compressed.")
    print("Pruning does NOT reduce latency on a standard CPU.")
    print(f"\nSaved: {pruned_path}")
