"""
Step 2.4 — Cut the cleaned signal into short, labelled, overlapping windows.

Each window becomes one training example for the model.
We carry the repetition number through so we can split honestly later.
"""

import scipy.io
import numpy as np
from preprocess import preprocess, FS

FILE_PATH = "data/raw/s1/S1_E2_A1.mat"

WINDOW_MS = 200   # length of each window
STEP_MS = 100     # how far we slide between windows (= 50% overlap)

WINDOW_SIZE = int(WINDOW_MS * FS / 1000)   # 40 samples
STEP_SIZE = int(STEP_MS * FS / 1000)       # 20 samples


def make_windows(emg, labels, reps, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """
    Slide a window along the signal, keeping only 'pure' windows —
    ones where the label does not change from start to finish.
    """
    X, y, r = [], [], []
    discarded = 0

    for start in range(0, len(emg) - window_size + 1, step):
        end = start + window_size

        window_labels = labels[start:end]
        window_reps = reps[start:end]

        # Purity check: is this window entirely one gesture?
        if len(np.unique(window_labels)) != 1:
            discarded += 1
            continue
        if len(np.unique(window_reps)) != 1:
            discarded += 1
            continue

        X.append(emg[start:end])
        y.append(window_labels[0])
        r.append(window_reps[0])

    print(f"Discarded {discarded:,} impure windows (label changed mid-window)")
    return np.array(X), np.array(y), np.array(r)

if __name__ == "__main__":

# --- Load and clean ---
    data = scipy.io.loadmat(FILE_PATH)
emg_raw = data["emg"]
labels = data["restimulus"].flatten()
reps = data["rerepetition"].flatten()

print("Cleaning signal (Phase 2 pipeline)...")
emg_clean = preprocess(emg_raw)

# --- Window it ---
print(f"\nWindow size : {WINDOW_SIZE} samples ({WINDOW_MS} ms)")
print(f"Step size   : {STEP_SIZE} samples ({STEP_MS} ms, 50% overlap)\n")

X, y, r = make_windows(emg_clean, labels, reps)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print(f"X (the windows) : {X.shape}")
print(f"  -> {X.shape[0]:,} examples, each {X.shape[1]} timesteps x {X.shape[2]} channels")
print(f"y (the labels)  : {y.shape}")
print(f"r (repetitions) : {r.shape}")

print("\nHow many windows per gesture:")
for gesture in np.unique(y):
    count = np.sum(y == gesture)
    name = "REST" if gesture == 0 else f"gesture {gesture}"
    print(f"  {name:12} : {count:6,} windows")

print("\nHow many windows per repetition:")
for rep in np.unique(r):
    print(f"  rep {rep} : {np.sum(r == rep):6,} windows")