"""
Decomposing the cross-subject gap, part 1: AMPLITUDE.

The baseline normalises subject 2 using subject 1's statistics. If the two
subjects simply have different signal amplitudes — different muscle mass,
adipose thickness, skin conductivity, band tightness — then every channel
is mis-scaled before the model sees it, and the gesture pattern could be
perfectly intact yet unrecognisable.

We test three normalisation strategies. None uses subject 2's LABELS, so
all three are deployable: a real device can ask a new user to wear the band
and move for a few seconds.

  A. Subject 1 stats        (current baseline)
  B. Subject 2's own stats  (computed from all their data)
  C. Subject 2's REST stats (computed from rest periods only)

C is the most realistic: "put the band on and hold still for five seconds."
"""

import numpy as np
import scipy.io
from tensorflow import keras

from preprocess import preprocess
from windowing import make_windows

DATA_PATH = "data/processed/db5_s1_e2.npz"
S2_PATH = "data/raw/s2/S2_E2_A1.mat"
MODEL_DIR = "models"

d = np.load(DATA_PATH)
X_test_s1, y_test_s1 = d["X_test"], d["y_test"]
mean_s1, std_s1 = d["mean"], d["std"]

model = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")
_, acc_s1 = model.evaluate(X_test_s1, y_test_s1, verbose=0)

# --- subject 2, cleaned and windowed, NOT yet normalised ---
data2 = scipy.io.loadmat(S2_PATH)
emg2_clean = preprocess(data2["emg"])
labels2 = data2["restimulus"].flatten()
reps2 = data2["rerepetition"].flatten()

X2_raw, y2, _ = make_windows(emg2_clean, labels2, reps2)

# rest windows — we keep these separately, to compute rest-only stats
rest_mask = y2 == 0
X2_rest = X2_raw[rest_mask]

# gesture windows — what we actually evaluate on
keep = y2 != 0
X2_gest, y2 = X2_raw[keep], y2[keep] - 1

print(f"Subject 2: {X2_gest.shape[0]:,} gesture windows, "
      f"{X2_rest.shape[0]:,} rest windows\n")


def evaluate(X, mean, std, label):
    Xn = (X - mean) / std
    _, acc = model.evaluate(Xn, y2, verbose=0)
    print(f"{label:<34} {acc*100:6.2f}%")
    return acc * 100


print(f"{'Subject 1 (reference)':<34} {acc_s1*100:6.2f}%")
print("-" * 44)

# A — current baseline: subject 1's statistics
acc_A = evaluate(X2_gest, mean_s1, std_s1,
                 "A. Subject 1 stats (baseline)")

# B — subject 2's own statistics, from all their data
mean_B = X2_gest.mean(axis=(0, 1), keepdims=True)
std_B = X2_gest.std(axis=(0, 1), keepdims=True)
acc_B = evaluate(X2_gest, mean_B, std_B,
                 "B. Subject 2's own stats")

# C — subject 2's REST statistics only.
# The most realistic calibration: "hold still for five seconds."
if len(X2_rest) > 0:
    mean_C = X2_rest.mean(axis=(0, 1), keepdims=True)
    std_C = X2_rest.std(axis=(0, 1), keepdims=True)
    acc_C = evaluate(X2_gest, mean_C, std_C,
                     "C. Subject 2's REST stats only")
else:
    acc_C = None
    print("C. no rest windows available")

print()
print("=" * 60)
print("HOW MUCH OF THE GAP IS AMPLITUDE?")
print("=" * 60)
gap = acc_s1 * 100 - acc_A
print(f"Original gap (S1 -> S2)     : {gap:.2f} pp")
print(f"Recovered by B (own stats)  : {acc_B - acc_A:+.2f} pp "
      f"({(acc_B - acc_A)/gap*100:.0f}% of the gap)")
if acc_C is not None:
    print(f"Recovered by C (rest stats) : {acc_C - acc_A:+.2f} pp "
          f"({(acc_C - acc_A)/gap*100:.0f}% of the gap)")

print()
best = max(acc_B, acc_C if acc_C else 0)
if best - acc_A > 10:
    print(">>> A large fraction of the gap is simple amplitude mis-scaling.")
    print(">>> Cheap, label-free calibration recovers much of it.")
elif best - acc_A > 3:
    print(">>> Amplitude accounts for a modest part of the gap.")
    print(">>> The remainder is genuine pattern mismatch.")
else:
    print(">>> Amplitude is NOT the problem. The gesture patterns themselves")
    print(">>> differ between subjects. Normalisation cannot fix this.")

np.savez("results/norm_test.npz",
         acc_s1=acc_s1*100, acc_A=acc_A, acc_B=acc_B,
         acc_C=acc_C if acc_C else np.nan)
