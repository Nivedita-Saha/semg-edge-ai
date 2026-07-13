"""
Decomposing the gap, part 2: is there a correctable ROTATION OFFSET?

Augmentation FAILED because it destroyed electrode-identity information —
telling the model "which electrode doesn't matter", which we showed is
false. Alignment does the opposite: it assumes electrode identity matters
enormously, and tries to RECOVER the correct mapping on a new arm.

Method: subject 2 performs ONE reference gesture. We find which electrode
fires hardest, compare to subject 1's, and roll by the difference.
Cost to the user: one gesture. No labels needed for the other 16.

We also brute-force all 8 possible rotations to find the ORACLE best —
the ceiling on what any alignment method could achieve. If even the oracle
fails, rotation is definitively not the problem.
"""

import numpy as np
import scipy.io
from tensorflow import keras

from preprocess import preprocess
from windowing import make_windows

DATA_PATH = "data/processed/db5_s1_e2.npz"
S1_PATH = "data/raw/s1/S1_E2_A1.mat"
S2_PATH = "data/raw/s2/S2_E2_A1.mat"
MODEL_DIR = "models"

REFERENCE_GESTURE = 1   # the one gesture we ask the new user to perform

d = np.load(DATA_PATH)
X_test_s1, y_test_s1 = d["X_test"], d["y_test"]
mean_s1, std_s1 = d["mean"], d["std"]

model = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")
_, acc_s1 = model.evaluate(X_test_s1, y_test_s1, verbose=0)


def load_subject(path):
    m = scipy.io.loadmat(path)
    X, y, _ = make_windows(preprocess(m["emg"]),
                           m["restimulus"].flatten(),
                           m["rerepetition"].flatten())
    keep = y != 0
    return X[keep], y[keep] - 1     # labels 0-16


def roll_bands(X, shift):
    """Rotate the armband: roll within each 8-electrode band."""
    b1 = np.roll(X[:, :, :8], shift, axis=2)
    b2 = np.roll(X[:, :, 8:], shift, axis=2)
    return np.concatenate([b1, b2], axis=2)


def dominant_channel(X, y, gesture):
    """Which electrode fires hardest for this gesture? (band 1 only)"""
    windows = X[y == gesture - 1]
    energy = np.abs(windows).mean(axis=(0, 1))   # per channel
    return int(np.argmax(energy[:8])), energy[:8]


X1, y1 = load_subject(S1_PATH)
X2, y2 = load_subject(S2_PATH)

print(f"Subject 1 reference : {acc_s1*100:.2f}%\n")

# ---------------------------------------------------------------
# 1. Estimate the offset from ONE reference gesture
# ---------------------------------------------------------------
ch1, e1 = dominant_channel(X1, y1, REFERENCE_GESTURE)
ch2, e2 = dominant_channel(X2, y2, REFERENCE_GESTURE)

offset = (ch1 - ch2) % 8

print(f"Reference gesture   : {REFERENCE_GESTURE}")
print(f"  Subject 1 dominant electrode : {ch1}")
print(f"  Subject 2 dominant electrode : {ch2}")
print(f"  Estimated rotation offset    : {offset} positions\n")


def evaluate_rolled(shift):
    Xr = roll_bands(X2, shift)
    Xr = (Xr - mean_s1) / std_s1
    _, acc = model.evaluate(Xr, y2, verbose=0)
    return acc * 100


# ---------------------------------------------------------------
# 2. Try EVERY rotation — the oracle upper bound
# ---------------------------------------------------------------
print("Trying every possible rotation:")
print("-" * 40)
accs = []
for shift in range(8):
    a = evaluate_rolled(shift)
    accs.append(a)
    tags = []
    if shift == 0:
        tags.append("<- no correction (baseline)")
    if shift == offset:
        tags.append("<- ESTIMATED offset")
    bar = "#" * int(a / 2)
    print(f"  shift {shift} : {a:5.2f}%  {bar} {' '.join(tags)}")

accs = np.array(accs)
best_shift = int(np.argmax(accs))
baseline_acc = accs[0]
estimated_acc = accs[offset]
oracle_acc = accs[best_shift]

print()
print("=" * 62)
print("DOES ALIGNMENT HELP?")
print("=" * 62)
print(f"No correction (shift 0)      : {baseline_acc:6.2f}%")
print(f"Estimated offset (shift {offset})    : {estimated_acc:6.2f}%   "
      f"{estimated_acc - baseline_acc:+.2f} pp")
print(f"ORACLE best (shift {best_shift})         : {oracle_acc:6.2f}%   "
      f"{oracle_acc - baseline_acc:+.2f} pp")
print("-" * 62)
print()

if oracle_acc - baseline_acc < 3:
    print(">>> Even the ORACLE best rotation barely helps.")
    print(">>> There is no correctable rotation offset between these subjects.")
    print(">>> ROTATION IS DEFINITIVELY NOT THE PROBLEM.")
    print(">>> Combined with the augmentation refutation, this closes the")
    print(">>> rotation hypothesis entirely.")
elif estimated_acc - baseline_acc > 3:
    print(">>> Alignment works, AND the one-gesture estimate finds it.")
    print(">>> A deployable calibration: perform one gesture, roll, done.")
else:
    print(">>> A rotation offset EXISTS (oracle helps) but the one-gesture")
    print(">>> estimate does not find it. A better offset estimator is needed.")

np.savez("results/alignment.npz",
         accs=accs, offset=offset, best_shift=best_shift)
