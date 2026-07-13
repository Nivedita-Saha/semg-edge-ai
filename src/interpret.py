"""
Phase 7 — Interpretability with SHAP.

Which sensor channels, and which moments in time, drive each gesture decision?

This matters for EnduRAI specifically: a factory worker wearing a device
that silently classifies their muscle activity will not trust it unless
someone can say WHY it decided what it decided. Accuracy alone does not
earn trust.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/db5_s1_e2.npz"
MODEL_DIR = "models"
FS = 200

d = np.load(DATA_PATH)
X_train = d["X_train"]
X_test, y_test = d["X_test"], d["y_test"]

model = keras.models.load_model(f"{MODEL_DIR}/baseline.keras")

# SHAP needs a background sample to define "what normal looks like".
# It compares each prediction against this baseline expectation.
background = X_train[np.random.choice(len(X_train), 100, replace=False)]

print("Computing SHAP values (this takes a minute)...")
explainer = shap.GradientExplainer(model, background)

# Explain a handful of test windows.
# Sample evenly across ALL 17 classes.
# The test set is ordered by gesture, so taking the first N windows
# would give us only gesture 1 — and every other row of the channel
# importance map would be empty.
PER_CLASS = 8
idx = np.concatenate([
    np.where(y_test == c)[0][:PER_CLASS] for c in range(17)
])
X_explain = X_test[idx]
y_explain = y_test[idx]
print(f"Explaining {len(idx)} windows across {len(np.unique(y_explain))} classes")

shap_values = explainer.shap_values(X_explain)
shap_values = np.array(shap_values)   # (N, 40, 16, 17)

print(f"SHAP values shape: {shap_values.shape}")
print("  -> (windows, timesteps, channels, classes)")


# =====================================================================
# FIGURE 1 — one gesture, explained: which channel, which moment
# =====================================================================
# Pick a window the model got RIGHT, so we're explaining a good decision.
preds = np.argmax(model.predict(X_explain, verbose=0), axis=1)
correct = np.where(preds == y_explain)[0]
i = correct[0]
true_class = y_explain[i]

# SHAP contribution for the PREDICTED class, at every (time, channel)
attr = shap_values[i, :, :, true_class]   # (40 timesteps, 16 channels)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                               gridspec_kw={"height_ratios": [1, 1.3]})

# top: the actual signal
time = np.arange(40) / FS * 1000   # ms
for ch in range(16):
    ax1.plot(time, X_explain[i, :, ch], linewidth=0.8, alpha=0.5)
ax1.set_ylabel("Normalised\nactivation")
ax1.set_title(f"The input — gesture {true_class + 1}, all 16 channels", fontweight="bold")
ax1.grid(alpha=0.3)

# bottom: what the model actually used
vmax = np.abs(attr).max()
im = ax2.imshow(attr.T, aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                extent=[0, 200, 15.5, -0.5])
ax2.set_xlabel("Time within window (ms)")
ax2.set_ylabel("Sensor channel")
ax2.set_title("What the model used — SHAP attribution\n"
              "red = pushed TOWARDS this gesture, blue = pushed away",
              fontweight="bold")
ax2.set_yticks(range(0, 16, 2))

cbar = fig.colorbar(im, ax=ax2, orientation="vertical", pad=0.02)
cbar.set_label("SHAP value")

plt.tight_layout()
plt.savefig("figures/step_7_1_shap_single.png", dpi=180, bbox_inches="tight")
print("\nSaved: figures/step_7_1_shap_single.png")


# =====================================================================
# FIGURE 2 — which channels matter, across ALL gestures
# =====================================================================
# Mean absolute SHAP per channel, per class. Which muscles does each
# gesture depend on? This is the figure that shows physiological structure.
importance = np.zeros((17, 16))
for cls in range(17):
    mask = y_explain == cls
    if mask.sum() == 0:
        continue
    # |SHAP| averaged over windows of this class and over time
    importance[cls] = np.abs(shap_values[mask][:, :, :, cls]).mean(axis=(0, 1))

# normalise each row so gestures are comparable
row_max = importance.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
importance_norm = importance / row_max

plt.figure(figsize=(12, 7))
plt.imshow(importance_norm, aspect="auto", cmap="viridis")
plt.colorbar(label="Relative channel importance")
plt.xlabel("Sensor channel (16 electrodes, two Myo armbands)")
plt.ylabel("Gesture")
plt.yticks(range(17), [f"gesture {i+1}" for i in range(17)], fontsize=8)
plt.xticks(range(16))
plt.title("Which muscles drive which gesture?\n"
          "Mean |SHAP| per channel, normalised within each gesture",
          fontweight="bold")
plt.tight_layout()
plt.savefig("figures/step_7_2_shap_channels.png", dpi=180, bbox_inches="tight")
print("Saved: figures/step_7_2_shap_channels.png")

np.savez("results/shap_importance.npz", importance=importance)

print()
print("=" * 60)
print("Look at figure 2. If different gestures rely on DIFFERENT")
print("channels, the model has learned real muscle physiology —")
print("not a shortcut. That is the claim worth making.")
print("=" * 60)

plt.show()
