"""
Step 5.3 — Knowledge distillation.

Train a small "student" model to imitate the big "teacher" model.

The student learns from the teacher's full probability distribution, not
just the correct label. Those near-miss probabilities ("this looks 72%
like gesture 7 but 19% like gesture 3") encode which gestures resemble
each other — information a hard label throws away. Hinton called this
"dark knowledge".

We also train an identical student from scratch on hard labels only, as a
control. Without that control, we cannot tell whether the student is good
because of distillation or simply because the architecture is adequate.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
N_CLASSES = 17

TEMPERATURE = 4.0   # softens the teacher's output so dark knowledge shows
ALPHA = 0.3         # weight on the true labels; 1-ALPHA on the teacher

tf.random.set_seed(42)
np.random.seed(42)


def build_student(input_shape, n_classes=N_CLASSES):
    """A deliberately smaller CNN. Same shape of design, fewer filters."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(16, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes),      # logits, no softmax — distillation needs raw
    ], name="student")


class Distiller(keras.Model):
    """Trains the student on two objectives at once."""

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def compile(self, optimizer, alpha=ALPHA, temperature=TEMPERATURE):
        super().compile(optimizer=optimizer)
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.distill_loss_fn = keras.losses.KLDivergence()
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, x):
        return self.student(x)

    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)

            # 1. The ordinary loss: did the student get the right answer?
            student_loss = self.student_loss_fn(y, student_pred)

            # 2. The distillation loss: does the student's whole probability
            #    distribution match the teacher's? Temperature softens both
            #    so the small "near-miss" probabilities become visible.
            T = self.temperature
            soft_teacher = tf.nn.softmax(tf.math.log(teacher_pred + 1e-9) / T)
            soft_student = tf.nn.softmax(student_pred / T)
            distill_loss = self.distill_loss_fn(soft_teacher, soft_student) * (T ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.acc_metric.update_state(y, student_pred)
        return {"loss": loss, "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        self.acc_metric.update_state(y, student_pred)
        return {"accuracy": self.acc_metric.result()}

    @property
    def metrics(self):
        return [self.acc_metric]


if __name__ == "__main__":

    d = np.load(DATA_PATH)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    teacher = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")
    teacher.trainable = False
    _, teacher_acc = teacher.evaluate(X_test, y_test, verbose=0)

    print(f"Teacher : {teacher.count_params():,} params, {teacher_acc*100:.2f}% accuracy")

    # -------------------------------------------------------------
    # A. The student, trained by DISTILLATION
    # -------------------------------------------------------------
    student = build_student(X_train.shape[1:])
    print(f"Student : {student.count_params():,} params "
          f"({teacher.count_params()/student.count_params():.1f}x smaller)\n")

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(optimizer=keras.optimizers.Adam(1e-3))

    print("Training student by distillation...\n")
    distiller.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, mode="max", verbose=1)],
        verbose=2,
    )

    student_logits = student.predict(X_test, verbose=0)
    distilled_acc = np.mean(np.argmax(student_logits, axis=1) == y_test)

    # -------------------------------------------------------------
    # B. THE CONTROL — an identical student, trained on hard labels only.
    #    Without this we cannot attribute anything to distillation.
    # -------------------------------------------------------------
    print("\nTraining CONTROL student (no teacher, hard labels only)...\n")

    control = build_student(X_train.shape[1:])
    control.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    control.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, mode="max", verbose=1)],
        verbose=2,
    )

    control_logits = control.predict(X_test, verbose=0)
    control_acc = np.mean(np.argmax(control_logits, axis=1) == y_test)

    # -------------------------------------------------------------
    # Save + report
    # -------------------------------------------------------------
    # Save the CONTROL student too — it is our best small model.
    control_full = keras.Sequential([control, layers.Softmax()])
    control_full.build((None,) + X_train.shape[1:])
    control_full.save(f"{MODEL_DIR}/control_student.keras")
    np.savez(f"{MODEL_DIR}/control_student_weights.npz", *control.get_weights())

    student_full = keras.Sequential([student, layers.Softmax()])
    student_full.build((None,) + X_train.shape[1:])
    student_full.save(f"{MODEL_DIR}/student.keras")
    np.savez(f"{MODEL_DIR}/student_weights.npz", *student.get_weights())

    print()
    print("=" * 66)
    print("KNOWLEDGE DISTILLATION RESULTS")
    print("=" * 66)
    print(f"{'Model':<28} {'Params':>10} {'Accuracy':>12}")
    print("-" * 66)
    print(f"{'Teacher (baseline)':<28} {teacher.count_params():>10,} {teacher_acc*100:>11.2f}%")
    print(f"{'Student (distilled)':<28} {student.count_params():>10,} {distilled_acc*100:>11.2f}%")
    print(f"{'Student (control, no teacher)':<28} {control.count_params():>10,} {control_acc*100:>11.2f}%")
    print("-" * 66)
    print()
    print(f"Compression         : {teacher.count_params()/student.count_params():.1f}x fewer parameters")
    print(f"Cost vs teacher     : {(distilled_acc - teacher_acc)*100:+.2f} pp")
    print(f"DISTILLATION EFFECT : {(distilled_acc - control_acc)*100:+.2f} pp")
    print("    ^ distilled student MINUS control student.")
    print("      This isolates what the teacher actually contributed.")
