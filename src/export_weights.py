"""
Bridge between Keras 3 and Keras 2.

tensorflow_model_optimization (used for pruning) still requires Keras 2,
which cannot read a model file saved by Keras 3. Weights, however, are
just arrays — they transfer fine.

Runs under Keras 3, loads the baseline, dumps its weights to a plain .npz.
"""

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/baseline.keras")
weights = model.get_weights()

np.savez("models/baseline_weights.npz", *weights)

print(f"Exported {len(weights)} weight arrays.")
for i, w in enumerate(weights):
    print(f"  [{i}] shape {w.shape}")
print("\nSaved: models/baseline_weights.npz")
