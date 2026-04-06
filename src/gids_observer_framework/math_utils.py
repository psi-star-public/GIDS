from __future__ import annotations

import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def binary_cross_entropy_from_logits(logits, y):
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean(np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits))))


def safe_mean(values, default=0.0):
    values = list(values)
    if not values:
        return default
    return float(np.mean(values))
