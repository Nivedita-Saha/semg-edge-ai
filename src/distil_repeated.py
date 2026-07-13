"""
Step 5.3 (proper) — Knowledge distillation, repeated across seeds.

A single run cannot support a claim about distillation: run-to-run
variance from random initialisation alone was observed to be ~4.6
percentage points, larger than the effect we are trying to measure.

We therefore run N seeds and report mean +/- standard deviation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from distil import build_student, Distiller, TEMPERATURE, ALPHA

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
N_SEEDS = 5

d = np.load(DATA_PATH)
X_train, y_train = d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
X_test, y_test = d["X_test"], d["y_test"]

teacher = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")
teacher.trainable = False
_, teacher_acc = teacher.evaluate(X_test, y_test, verbose=0)

distilled_accs, control_accs = [], []

for seed in range(N_SEEDS):
    print(f"\n--- seed {seed} ---")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    es = lambda: keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20,
        restore_best_weights=True, mode="max", verbose=0)

    # distilled
    student = build_student(X_train.shape[1:])
    dist = Distiller(student=student, teacher=teacher)
    dist.compile(optimizer=keras.optimizers.Adam(1e-3))
    dist.fit(X_train, y_train, validation_data=(X_val, y_val),
             epochs=100, batch_size=32, callbacks=[es()], verbose=0)
    acc_d = np.mean(np.argmax(student.predict(X_test, verbose=0), axis=1) == y_test)

    # control
    control = build_student(X_train.shape[1:])
    control.compile(optimizer=keras.optimizers.Adam(1e-3),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"])
    control.fit(X_train, y_train, validation_data=(X_val, y_val),
                epochs=100, batch_size=32, callbacks=[es()], verbose=0)
    acc_c = np.mean(np.argmax(control.predict(X_test, verbose=0), axis=1) == y_test)

    distilled_accs.append(acc_d)
    control_accs.append(acc_c)
    print(f"  distilled {acc_d*100:.2f}%   control {acc_c*100:.2f}%   diff {(acc_d-acc_c)*100:+.2f}")

distilled_accs = np.array(distilled_accs)
control_accs = np.array(control_accs)
diffs = distilled_accs - control_accs

print()
print("=" * 68)
print(f"DISTILLATION, {N_SEEDS} SEEDS")
print("=" * 68)
print(f"Teacher              : {teacher_acc*100:.2f}%  (18,545 params)")
print(f"Student (distilled)  : {distilled_accs.mean()*100:.2f}% +/- {distilled_accs.std()*100:.2f}")
print(f"Student (control)    : {control_accs.mean()*100:.2f}% +/- {control_accs.std()*100:.2f}")
print("-" * 68)
print(f"Distillation effect  : {diffs.mean()*100:+.2f} pp +/- {diffs.std()*100:.2f}")
print()
if abs(diffs.mean()) < diffs.std():
    print(">>> The effect is SMALLER than its own variance.")
    print(">>> Conclusion: distillation shows no reliable benefit on this task.")
else:
    print(">>> The effect appears larger than run-to-run noise.")

np.savez(f"{MODEL_DIR}/distil_seeds.npz",
         distilled=distilled_accs, control=control_accs)
