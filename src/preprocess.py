"""
Phase 2 — Clean the raw sEMG signals.

Step 2.1: band-pass filter  (keep 20-99 Hz)
Step 2.2: notch filter      (kill 50 Hz mains hum)
Step 2.3: rectify + smooth  (extract the activation envelope)

The functions below are imported by other files.
The demo/plot code at the bottom only runs when this file is run directly.
"""

import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "data/raw/s1/S1_E2_A1.mat"
FS = 200  # sampling rate, Hz


def bandpass_filter(signal, low_hz=20, high_hz=99, fs=FS, order=4):
    """Keep only frequencies between low_hz and high_hz."""
    nyquist = fs / 2
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal, notch_hz=50, quality=30, fs=FS):
    """Remove the mains hum at notch_hz."""
    b, a = iirnotch(notch_hz, quality, fs)
    return filtfilt(b, a, signal, axis=0)


def rectify(signal):
    """Absolute value — the sign carries no information, the size does."""
    return np.abs(signal)


def smooth(signal, window_ms=50, fs=FS):
    """Moving average — turns jitter into a clean activation envelope."""
    window_samples = int(window_ms * fs / 1000)
    cutoff_hz = fs / window_samples
    nyquist = fs / 2
    b, a = butter(2, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, signal, axis=0)


def preprocess(emg_raw):
    """The full cleaning pipeline, in order. This is the whole of Phase 2."""
    x = bandpass_filter(emg_raw)
    x = notch_filter(x)
    x = rectify(x)
    x = smooth(x)
    return x


# ============================================================
# Everything below runs ONLY when this file is run directly.
# It does NOT run when another file does: from preprocess import preprocess
# ============================================================
if __name__ == "__main__":

    data = scipy.io.loadmat(FILE_PATH)
    emg_raw = data["emg"]
    labels = data["restimulus"].flatten()

    emg_bp = bandpass_filter(emg_raw)
    emg_notch = notch_filter(emg_bp)
    emg_rect = rectify(emg_notch)
    emg_env = smooth(emg_rect)

    print(f"Raw       : {emg_raw.shape}")
    print(f"Envelope  : {emg_env.shape}")
    print(f"\nEnvelope range: {emg_env.min():.2f} to {emg_env.max():.2f}")
    print("(should be all positive — rectification made it so)")

    start, end = 22_500, 22_500 + FS * 12
    time = np.arange(start, end) / FS

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(time, emg_raw[start:end, 0], linewidth=0.5, color="grey")
    axes[0].set_title("1. RAW — jagged, symmetric around zero")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(time, emg_notch[start:end, 0], linewidth=0.5, color="steelblue")
    axes[1].set_title("2. FILTERED — band-pass + notch applied")
    axes[1].set_ylabel("amplitude")

    axes[2].plot(time, emg_rect[start:end, 0], linewidth=0.5, color="darkorange")
    axes[2].set_title("3. RECTIFIED — all positive, but still jittery")
    axes[2].set_ylabel("amplitude")

    axes[3].plot(time, emg_env[start:end, 0], linewidth=1.8, color="crimson")
    axes[3].set_title("4. ENVELOPE — smoothed. THIS is what the model learns from")
    axes[3].set_ylabel("activation")
    axes[3].set_xlabel("Time (seconds)")

    gesture_mask = labels[start:end] > 0
    for ax in axes:
        ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=gesture_mask, color="green", alpha=0.12)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/step_2_3_envelope.png", dpi=150)
    print("\nSaved: figures/step_2_3_envelope.png")
    plt.show()