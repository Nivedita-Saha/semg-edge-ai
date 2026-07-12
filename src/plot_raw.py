"""
Step 1.3 — Plot the raw sEMG signal and check it against the gesture labels.

Goal: see with your own eyes that the muscle bursts line up with the
gestures. This is a sanity check, not analysis.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "data/raw/s1/S1_E2_A1.mat"
FS = 200  # sampling rate in Hz — 200 samples per second

data = scipy.io.loadmat(FILE_PATH)
emg = data["emg"]
labels = data["restimulus"].flatten()  # flatten turns (179901,1) into (179901,)

# --- Look at a 60-second slice, not the whole 15 minutes ---
# Plotting 180,000 points is unreadable. A window is enough.
start = 20_000          # start at sample 20,000 (= 100 seconds in)
end = start + FS * 60   # 60 seconds' worth

emg_slice = emg[start:end]
label_slice = labels[start:end]

# Build a time axis in seconds, so the x-axis means something.
time = np.arange(start, end) / FS

# --- Draw two panels, stacked, sharing the same x-axis ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# Panel 1: one sensor channel (channel 0 of 16)
ax1.plot(time, emg_slice[:, 0], linewidth=0.5, color="steelblue")
ax1.set_ylabel("EMG amplitude\n(channel 0)")
ax1.set_title("Raw sEMG signal — Subject 1, Exercise 2, channel 0")
ax1.grid(alpha=0.3)

# Panel 2: what gesture was happening at each moment
ax2.plot(time, label_slice, linewidth=1.5, color="darkorange")
ax2.set_ylabel("Gesture label\n(0 = rest)")
ax2.set_xlabel("Time (seconds)")
ax2.set_title("The answer key — which gesture was being performed")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/raw_signal_check.png", dpi=150)
print("Saved: figures/raw_signal_check.png")
plt.show()