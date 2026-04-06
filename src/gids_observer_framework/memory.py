from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def memory_field(trace_vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    traces = np.asarray(trace_vectors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if traces.ndim != 2:
        raise ValueError("trace_vectors must be a 2D array-like object.")
    if len(weights) != len(traces):
        raise ValueError("weights and trace_vectors must align on the first dimension.")
    return (weights[:, None] * traces).sum(axis=0)


def update_trace_weights(
    trace_vectors: Sequence[np.ndarray],
    current_weights: Sequence[float],
    proposition: np.ndarray,
    context_gain: float = 0.5,
) -> np.ndarray:
    traces = np.asarray(trace_vectors, dtype=float)
    weights = np.asarray(current_weights, dtype=float)
    proposition = np.asarray(proposition, dtype=float)
    similarity = traces @ proposition
    raw = weights * np.exp(context_gain * similarity)
    if raw.sum() == 0:
        return np.ones_like(raw) / len(raw)
    return raw / raw.sum()
