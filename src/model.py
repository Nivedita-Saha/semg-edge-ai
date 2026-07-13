"""
Phase 4 — Build and train the baseline 1D-CNN.

Step 4.1: architecture
Step 4.2: train, and record test accuracy  <- the BASELINE
Step 4.3: save the model, note its size and speed

Every compressed model in Phase 5 gets compared against this.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import time

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
N_CLASSES = 17

# Reproducibility — so you get the same result if you run it again.
tf.random.set_seed(42)
np.random.seed(42)


def build_model(input_shape, n_classes=N_CLASSES):
    """
    A small 1D-CNN for sEMG gesture recognition.

    Standard architecture. The design choices lean deliberately small,
    because the point of this project is fitting on constrained hardware.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),                 # (40 timesteps, 16 channels)

        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),               # 40 -> 20

        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),               # 20 -> 10

        layers.Dropout(0.3),

        # Global average pooling, NOT Flatten.
        # Flatten -> huge dense layer -> big model. We want small.
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),  # one score per gesture
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":

    # --- Load the prepared data ---
    d = np.load(DATA_PATH)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")
    print(f"Classes: {N_CLASSES}  (chance level = {100/N_CLASSES:.1f}%)")

    # --- Build ---
    model = build_model(input_shape=X_train.shape[1:])
    print()
    model.summary()

    # --- Train ---
    # EarlyStopping watches validation loss. When it stops improving,
    # training halts and the best weights are restored. This prevents
    # overfitting and saves you guessing how many epochs to run.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nTraining...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Evaluate on the TEST set — touched once, right now ---
    print("\n" + "=" * 55)
    print("BASELINE RESULT")
    print("=" * 55)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy : {test_acc * 100:.2f}%")
    print(f"Chance level  : {100/N_CLASSES:.1f}%")

    # --- Save ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f"{MODEL_DIR}/baseline.keras"
    model.save(model_path)

    size_kb = os.path.getsize(model_path) / 1024
    n_params = model.count_params()

    # --- Time one prediction (latency) ---
    single = X_test[:1]
    model.predict(single, verbose=0)          # warm-up run, don't time this
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model.predict(single, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    latency_ms = np.median(times)

    print()
    print(f"Parameters    : {n_params:,}")
    print(f"Model size    : {size_kb:.1f} KB")
    print(f"Latency       : {latency_ms:.2f} ms per prediction")
    print(f"\nSaved: {model_path}")
    print("\nThese three numbers — accuracy, size, latency — are the")
    print("spine of the whole project. Write them in your tracker.")