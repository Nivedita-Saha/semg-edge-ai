"""
Phase 3 — Prepare the data for the model.

Step 3.1: drop rest, normalise
Step 3.2: split by REPETITION (not randomly — that would leak)
Step 3.3: save to disk

Key principle: split FIRST, then compute normalisation stats from
the training set only. Anything else leaks the test set into training.
"""

import scipy.io
import numpy as np
import os
from preprocess import preprocess, FS
from windowing import make_windows

FILE_PATH = "data/raw/s1/S1_E2_A1.mat"
OUT_DIR = "data/processed"

# Split by repetition. Repetitions are near-identical, so a random
# split would put near-duplicates in both train and test.
TRAIN_REPS = [1, 3, 4, 6]
VAL_REPS = [2]
TEST_REPS = [5]


# --- 1. Load, clean, window ---
print("Loading and cleaning...")
data = scipy.io.loadmat(FILE_PATH)
emg_raw = data["emg"]
labels = data["restimulus"].flatten()
reps = data["rerepetition"].flatten()

emg_clean = preprocess(emg_raw)
X, y, r = make_windows(emg_clean, labels, reps)

print(f"\nAll windows: {X.shape}")


# --- 2. Drop rest (label 0) ---
keep = y != 0
X, y, r = X[keep], y[keep], r[keep]

print(f"After dropping rest: {X.shape}")
print(f"Classes remaining: {np.unique(y)}  ({len(np.unique(y))} gestures)")


# --- 3. Relabel 1-17 as 0-16 ---
# Keras expects class indices to start at 0.
y = y - 1


# --- 4. Split BY REPETITION ---
train_mask = np.isin(r, TRAIN_REPS)
val_mask = np.isin(r, VAL_REPS)
test_mask = np.isin(r, TEST_REPS)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print()
print("=" * 55)
print("SPLIT (by repetition — no leakage)")
print("=" * 55)
print(f"Train  (reps {TRAIN_REPS}) : {X_train.shape[0]:5,} windows")
print(f"Val    (reps {VAL_REPS}) : {X_val.shape[0]:5,} windows")
print(f"Test   (reps {TEST_REPS}) : {X_test.shape[0]:5,} windows")


# --- 5. Normalise — stats from TRAINING SET ONLY ---
# This is the step where leakage usually happens. We compute the mean
# and std per channel from the training data, then apply those exact
# numbers to val and test. The test set never influences the stats.
mean = X_train.mean(axis=(0, 1), keepdims=True)   # per channel
std = X_train.std(axis=(0, 1), keepdims=True)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print()
print("Normalised using TRAINING statistics only.")
print(f"  Train mean: {X_train.mean():.4f}  std: {X_train.std():.4f}  (should be ~0 and ~1)")
print(f"  Test  mean: {X_test.mean():.4f}  std: {X_test.std():.4f}  (will NOT be exactly 0/1 — that's correct)")


# --- 6. Save ---
os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(
    f"{OUT_DIR}/db5_s1_e2.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    mean=mean, std=std,
)

size_mb = os.path.getsize(f"{OUT_DIR}/db5_s1_e2.npz") / 1e6
print(f"\nSaved: {OUT_DIR}/db5_s1_e2.npz  ({size_mb:.1f} MB)")
print("You never have to redo the slow cleaning steps again.")