from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def task_projection(embedding: np.ndarray, projector: np.ndarray) -> np.ndarray:
    return np.asarray(projector, dtype=float) @ np.asarray(embedding, dtype=float)


def salience_slice(projected_embedding: np.ndarray, attention: np.ndarray) -> np.ndarray:
    projected_embedding = np.asarray(projected_embedding, dtype=float)
    attention = np.asarray(attention, dtype=float)
    if projected_embedding.shape != attention.shape:
        raise ValueError("projected_embedding and attention must have the same shape.")
    return attention * projected_embedding


def estimate_slow_embedding(
    feature_blocks: Mapping[str, np.ndarray],
    weight_matrices: Mapping[str, np.ndarray],
    bias: np.ndarray | None = None,
    apply_tanh: bool = True,
) -> np.ndarray:
    keys = list(feature_blocks.keys())
    if set(keys) != set(weight_matrices.keys()):
        raise ValueError("feature_blocks and weight_matrices must share the same keys.")
    total = None
    for key in keys:
        contribution = np.asarray(weight_matrices[key], dtype=float) @ np.asarray(feature_blocks[key], dtype=float)
        total = contribution if total is None else total + contribution
    if bias is not None:
        total = total + np.asarray(bias, dtype=float)
    return np.tanh(total) if apply_tanh else total
