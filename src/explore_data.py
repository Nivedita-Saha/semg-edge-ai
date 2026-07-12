"""
Step 1.2 — Load one Ninapro DB5 file and inspect its contents.

Goal: understand the raw material before touching it.
We are not processing anything here — just looking.
"""

import scipy.io
import numpy as np

# Point at one subject, one exercise.
FILE_PATH = "data/raw/s1/S1_E2_A1.mat"

# A .mat file is a MATLAB file. scipy can read it.
# It comes back as a dictionary: keys are variable names, values are the data.
data = scipy.io.loadmat(FILE_PATH)

print("=" * 60)
print("WHAT IS IN THIS FILE?")
print("=" * 60)

# Show every variable in the file and the shape of its data.
# Keys starting with "__" are MATLAB's own metadata — ignore those.
for key in data:
    if key.startswith("__"):
        continue
    value = data[key]
    print(f"{key:12} shape={str(value.shape):20} type={value.dtype}")

print()
print("=" * 60)
print("THE THREE THINGS THAT MATTER")
print("=" * 60)

emg = data["emg"]          # the muscle signals
labels = data["restimulus"]  # which gesture was being made
reps = data["rerepetition"]  # which repetition of that gesture

print(f"emg          : {emg.shape}")
print(f"  -> {emg.shape[0]:,} time samples, {emg.shape[1]} sensor channels")

print(f"\nrestimulus   : {labels.shape}")
unique_gestures = np.unique(labels)
print(f"  -> gesture labels present: {unique_gestures}")
print(f"  -> that is {len(unique_gestures)} distinct values (0 = rest, no gesture)")

print(f"\nrerepetition : {reps.shape}")
print(f"  -> repetitions present: {np.unique(reps)}")

print()
print("=" * 60)
print("SANITY CHECK ON THE SIGNAL ITSELF")
print("=" * 60)
print(f"Smallest value : {emg.min():.6f}")
print(f"Largest value  : {emg.max():.6f}")
print(f"Mean value     : {emg.mean():.6f}")
print("\nFirst 5 time samples, all 16 channels:")
print(emg[:5])