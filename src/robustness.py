"""
Phase 8 — Cross-subject robustness.

Take the model trained entirely on subject 1 and run it, unchanged, on
subject 2. No retraining, no fine-tuning.

Expect a large drop. The Myo armband sits at a different rotation on a
different arm, so a given electrode sits over a different muscle. The
model learned "channel 3 means X" for subject 1's arm; on subject 2's
arm, channel 3 means something else.

This is THE open problem in wearable sEMG, and the reason commercial
gesture-control devices have repeatedly failed. Reporting it honestly is
more valuable than hiding it.
"""

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from preprocess import preprocess
from windowing import make_windows

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
S2_PATH = "data/raw/s2/S2_E2_A1.mat"

# --- The model, and subject 1's normalisation statistics ---
d = np.load(DATA_PATH)
X_test_s1, y_test_s1 = d["X_test"], d["y_test"]
mean, std = d["mean"], d["std"]

model = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")

_, acc_s1 = model.evaluate(X_test_s1, y_test_s1, verbose=0)
print(f"Subject 1 (seen)   : {acc_s1*100:.2f}%")

# --- Subject 2: same pipeline, completely unseen person ---
print("\nProcessing subject 2...")
data2 = scipy.io.loadmat(S2_PATH)
emg2 = preprocess(data2["emg"])
labels2 = data2["restimulus"].flatten()
reps2 = data2["rerepetition"].flatten()

X2, y2, r2 = make_windows(emg2, labels2, reps2)

# drop rest, relabel to 0-16 — exactly as for subject 1
keep = y2 != 0
X2, y2 = X2[keep], y2[keep] - 1

# CRITICAL: normalise with SUBJECT 1's statistics.
# This is what a deployed model would have to do — it cannot see the new
# user's data in advance. Using subject 2's own stats would be cheating.
X2 = (X2 - mean) / std

print(f"Subject 2 windows  : {X2.shape[0]:,}")

_, acc_s2 = model.evaluate(X2, y2, verbose=0)

print()
print("=" * 58)
print("CROSS-SUBJECT RESULT")
print("=" * 58)
print(f"Subject 1 (trained on)  : {acc_s1*100:6.2f}%")
print(f"Subject 2 (never seen)  : {acc_s2*100:6.2f}%")
print(f"Chance level            : {100/17:6.2f}%")
print("-" * 58)
print(f"ACCURACY DROP           : {(acc_s1 - acc_s2)*100:6.2f} percentage points")
print()

if acc_s2 * 100 < 15:
    print(">>> Near chance. The model has learned subject 1's specific")
    print(">>> electrode-to-muscle mapping, and it does not transfer.")
elif acc_s2 * 100 < 40:
    print(">>> Substantial degradation. Some structure transfers, but")
    print(">>> the model is far from deployable on a new user.")
else:
    print(">>> Better transfer than typical. Worth investigating why.")


# --- Per-gesture breakdown: does ANYTHING survive? ---
pred2 = np.argmax(model.predict(X2, verbose=0), axis=1)

print()
print("Per-gesture accuracy on subject 2:")
per_class = []
for c in range(17):
    mask = y2 == c
    if mask.sum() == 0:
        per_class.append(0)
        continue
    a = (pred2[mask] == c).mean()
    per_class.append(a)
    bar = "#" * int(a * 30)
    print(f"  gesture {c+1:2d} : {a*100:5.1f}%  {bar}")


# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.bar(["Subject 1\n(trained on)", "Subject 2\n(never seen)"],
        [acc_s1 * 100, acc_s2 * 100],
        color=["#2ca02c", "#d62728"], width=0.55)
ax1.axhline(100 / 17, ls="--", color="grey",
            label=f"chance ({100/17:.1f}%)")
ax1.set_ylabel("Test accuracy (%)")
ax1.set_title("The cross-subject gap", fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3, axis="y")
for i, v in enumerate([acc_s1 * 100, acc_s2 * 100]):
    ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")

cm = confusion_matrix(y2, pred2, labels=range(17), normalize="true")
im = ax2.imshow(cm, cmap="Reds", vmin=0, vmax=1)
ax2.set_xlabel("Predicted gesture")
ax2.set_ylabel("True gesture")
ax2.set_title("Subject 2 — confusion matrix\n"
              "a strong vertical stripe = model collapsing onto one class",
              fontweight="bold")
ax2.set_xticks(range(0, 17, 2))
ax2.set_yticks(range(0, 17, 2))
ax2.set_xticklabels(range(1, 18, 2))
ax2.set_yticklabels(range(1, 18, 2))
fig.colorbar(im, ax=ax2, label="proportion")

plt.tight_layout()
plt.savefig("figures/step_8_1_cross_subject.png", dpi=180, bbox_inches="tight")
print("\nSaved: figures/step_8_1_cross_subject.png")

np.savez("results/cross_subject.npz",
         acc_s1=acc_s1, acc_s2=acc_s2, per_class=np.array(per_class))

plt.show()
