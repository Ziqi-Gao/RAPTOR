"""Run the real RAPTOR API on synthetic data; this is not a paper result."""

import numpy as np
from raptor.core import split_indices
from raptor.experiments.run_raptor import fit_raptor_layer

rng = np.random.default_rng(42)
y = np.repeat([0, 1], 125)
X = rng.normal(size=(250, 16)).astype(np.float32)
X[:, :3] += (2 * y - 1)[:, None]
train, val, test = split_indices(y, test_size=0.2, val_size=0.2, seed=42)

probe = fit_raptor_layer(
    X[train], y[train], X[val], y[val], X[test], y[test], seed=42
)
logits = X[test] @ probe["W"] + probe["B"]  # raw coordinates
predictions = (logits > 0).astype(int)
print("Selected C:", probe["best_C"])
print("Synthetic test accuracy:", probe["acc_test"])
print("Direction shape:", probe["W"].shape)  # (16,)
