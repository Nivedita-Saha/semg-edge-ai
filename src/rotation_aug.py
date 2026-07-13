"""
Testing the rotation hypothesis.

Phase 8 found a 46.5pp cross-subject collapse. Phase 7 suggested a
mechanism: gestures relying on a single dominant electrode transferred;
gestures relying on distributed spatial patterns did not. That points at
ARMBAND ROTATION as the culprit — a rotated band scrambles the spatial
pattern but leaves a strong local activation roughly intact.

If that is right, then simulating rotation during training should improve
cross-subject transfer. If it is not, this will do nothing, and we will
have learned that rotation is not the dominant factor.

Either outcome is informative. That is the point.

NOTE: DB5 uses TWO Myo armbands of 8 electrodes each, stacked along the
forearm. We therefore roll WITHIN each band of 8 (7 wraps to 0, 15 wraps
to 8), not across all 16 — rolling across 16 would wrap from one armband
to the other, which is physically meaningless. Both bands roll by the
same amount, since they sit on the same rotating arm.
"""

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from model import build_model
from preprocess import preprocess
from windowing import make_windows

DATA_PATH = "data/processed/db5_s1_e2.npz"
S2_PATH = "data/raw/s2/S2_E2_A1.mat"
MAX_SHIFT = 2       # roll by -2 .. +2 electrode positions
N_SEEDS = 3         # we learned our lesson: never trust one run


def roll_bands(X, shift):
    """
    Rotate the armband by `shift` electrode positions.
    Channels 0-7 are band 1, 8-15 are band 2. Roll each independently
    but by the same amount — one arm, one rotation.
    """
    band1 = np.roll(X[:, :, :8], shift, axis=2)
    band2 = np.roll(X[:, :, 8:], shift, axis=2)
    return np.concatenate([band1, band2], axis=2)


class RotationAugment(keras.utils.Sequence):
    """Feeds batches, applying a random rotation to each."""

    def __init__(self, X, y, batch_size=32, max_shift=MAX_SHIFT):
        self.X, self.y = X, y
        self.bs = batch_size
        self.max_shift = max_shift
        self.idx = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, i):
        ids = self.idx[i * self.bs:(i + 1) * self.bs]
        xb, yb = self.X[ids], self.y[ids]
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift != 0:
            xb = roll_bands(xb, shift)
        return xb, yb

    def on_epoch_end(self):
        np.random.shuffle(self.idx)


# ---------------------------------------------------------------
# Load subject 1 (train/val/test) and subject 2 (unseen)
# ---------------------------------------------------------------
d = np.load(DATA_PATH)
X_train, y_train = d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
X_test, y_test = d["X_test"], d["y_test"]
mean, std = d["mean"], d["std"]

data2 = scipy.io.loadmat(S2_PATH)
X2, y2, _ = make_windows(preprocess(data2["emg"]),
                         data2["restimulus"].flatten(),
                         data2["rerepetition"].flatten())
keep = y2 != 0
X2, y2 = X2[keep], y2[keep] - 1
X2 = (X2 - mean) / std      # subject 1's stats — as a deployed model must

print(f"Subject 1 test : {X_test.shape[0]:,} windows")
print(f"Subject 2      : {X2.shape[0]:,} windows\n")


def train_once(seed, augment):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(X_train.shape[1:])

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=0)

    if augment:
        gen = RotationAugment(X_train, y_train)
        model.fit(gen, validation_data=(X_val, y_val),
                  epochs=150, callbacks=[es], verbose=0)
    else:
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=150, batch_size=32, callbacks=[es], verbose=0)

    _, acc_s1 = model.evaluate(X_test, y_test, verbose=0)
    _, acc_s2 = model.evaluate(X2, y2, verbose=0)
    return acc_s1 * 100, acc_s2 * 100


results = {"baseline": [], "augmented": []}

for seed in range(N_SEEDS):
    print(f"--- seed {seed} ---")
    s1, s2 = train_once(seed, augment=False)
    results["baseline"].append((s1, s2))
    print(f"  baseline  : subj1 {s1:5.2f}%   subj2 {s2:5.2f}%")

    s1, s2 = train_once(seed, augment=True)
    results["augmented"].append((s1, s2))
    print(f"  augmented : subj1 {s1:5.2f}%   subj2 {s2:5.2f}%")

b = np.array(results["baseline"])
a = np.array(results["augmented"])

print()
print("=" * 66)
print(f"ROTATION AUGMENTATION — {N_SEEDS} seeds, roll +/-{MAX_SHIFT} electrodes")
print("=" * 66)
print(f"{'':<14} {'Subject 1 (seen)':>20} {'Subject 2 (unseen)':>22}")
print("-" * 66)
print(f"{'Baseline':<14} {b[:,0].mean():>13.2f} +/- {b[:,0].std():<4.2f} "
      f"{b[:,1].mean():>14.2f} +/- {b[:,1].std():<4.2f}")
print(f"{'Augmented':<14} {a[:,0].mean():>13.2f} +/- {a[:,0].std():<4.2f} "
      f"{a[:,1].mean():>14.2f} +/- {a[:,1].std():<4.2f}")
print("-" * 66)

d_s1 = a[:,0].mean() - b[:,0].mean()
d_s2 = a[:,1].mean() - b[:,1].mean()

print(f"\nEffect on subject 1 (within) : {d_s1:+.2f} pp")
print(f"Effect on subject 2 (cross)  : {d_s2:+.2f} pp")
print()

if d_s2 > 3:
    print(">>> ROTATION HYPOTHESIS SUPPORTED.")
    print(">>> Simulating armband rotation improves cross-subject transfer.")
    print(">>> A rotation-invariant architecture (circular convolution over")
    print(">>> the electrode ring) is now well motivated.")
elif d_s2 > 0:
    print(">>> Weak/ambiguous. Rotation may contribute but is not dominant.")
else:
    print(">>> ROTATION HYPOTHESIS NOT SUPPORTED.")
    print(">>> Rotation is not the main driver of the cross-subject gap.")
    print(">>> Look instead at anatomy, skin conductivity, or differences")
    print(">>> in how subjects perform the gestures.")

np.savez("results/rotation_aug.npz", baseline=b, augmented=a)

# --- figure ---
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, [b[:,0].mean(), b[:,1].mean()], w,
       yerr=[b[:,0].std(), b[:,1].std()], capsize=4,
       label="Baseline", color="#888888")
ax.bar(x + w/2, [a[:,0].mean(), a[:,1].mean()], w,
       yerr=[a[:,0].std(), a[:,1].std()], capsize=4,
       label=f"+ rotation augmentation (+/-{MAX_SHIFT})", color="#2ca02c")
ax.axhline(100/17, ls="--", color="grey", alpha=0.7, label="chance")
ax.set_xticks(x)
ax.set_xticklabels(["Subject 1\n(trained on)", "Subject 2\n(never seen)"])
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Does simulating armband rotation improve cross-subject transfer?",
             fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figures/step_8_2_rotation_aug.png", dpi=180, bbox_inches="tight")
print("\nSaved: figures/step_8_2_rotation_aug.png")
plt.show()
