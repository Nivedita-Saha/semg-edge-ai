"""
Step 2.4 — Cut the cleaned signal into short, labelled, overlapping windows.
"""

import scipy.io
import numpy as np
from preprocess import preprocess, FS

FILE_PATH = "data/raw/s1/S1_E2_A1.mat"

WINDOW_MS = 200
STEP_MS = 100

WINDOW_SIZE = int(WINDOW_MS * FS / 1000)
STEP_SIZE = int(STEP_MS * FS / 1000)


def make_windows(emg, labels, reps, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """Keep only 'pure' windows — where the label does not change."""
    X, y, r = [], [], []
    discarded = 0

    for start in range(0, len(emg) - window_size + 1, step):
        end = start + window_size
        window_labels = labels[start:end]
        window_reps = reps[start:end]

        if len(np.unique(window_labels)) != 1:
            discarded += 1
            continue
        if len(np.unique(window_reps)) != 1:
            discarded += 1
            continue

        X.append(emg[start:end])
        y.append(window_labels[0])
        r.append(window_reps[0])

    print(f"Discarded {discarded:,} impure windows")
    return np.array(X), np.array(y), np.array(r)


if __name__ == "__main__":
    data = scipy.io.loadmat(FILE_PATH)
    emg_raw = data["emg"]
    labels = data["restimulus"].flatten()
    reps = data["rerepetition"].flatten()

    print("Cleaning signal...")
    emg_clean = preprocess(emg_raw)

    X, y, r = make_windows(emg_clean, labels, reps)

    print(f"\nX: {X.shape}")
    print(f"y: {y.shape}")
    print(f"r: {r.shape}")
