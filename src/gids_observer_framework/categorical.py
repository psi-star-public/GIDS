from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


TokenObservation = Mapping[str, str]
TokenBags = Mapping[Tuple[str, str], Sequence[str]]


def contextual_lift(raw_tokens: Sequence[TokenObservation], context: Mapping[str, str] | None = None):
    """
    Convert raw category observations into typed tokens before comparison or pooling.

    The default rule is intentionally simple:
    - preserve family and label
    - preserve an explicit axis if it exists
    - otherwise back off to the current regime if one was supplied
    """
    context = context or {}
    regime = context.get("regime", "general")
    typed = []
    for token in raw_tokens:
        axis = token.get("axis", regime)
        typed.append(
            {
                "family": token["family"],
                "label": token["label"],
                "axis": axis,
                "typed_token": f"{token['family']}::{axis}::{token['label']}",
            }
        )
    return typed


def pool_slot(tokens: Sequence[str], embedding_table: Mapping[str, np.ndarray], null_vector: np.ndarray):
    if len(tokens) > 0:
        pooled = np.mean([embedding_table[token] for token in tokens], axis=0)
        mask = 1.0
    else:
        pooled = np.asarray(null_vector, dtype=float).copy()
        mask = 0.0
    return np.asarray(pooled, dtype=float), mask


def build_event_categorical_embedding(
    token_bags: TokenBags,
    embedding_tables: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
    null_vectors: Mapping[Tuple[str, str], np.ndarray],
    projections: Mapping[Tuple[str, str], np.ndarray],
    family_order: Sequence[str],
    source_order: Sequence[str],
):
    parts: List[float] = []
    slot_masks = {}
    for family in family_order:
        for source in source_order:
            key = (family, source)
            pooled, mask = pool_slot(token_bags.get(key, []), embedding_tables[key], null_vectors[key])
            projected = projections[key] @ pooled
            parts.extend(projected.tolist())
            parts.append(mask)
            slot_masks[key] = mask
    return np.asarray(parts, dtype=float), slot_masks


def build_slow_bank(
    event_history: Sequence[Mapping[str, object]],
    regimes: Sequence[str],
    event_dim: int,
):
    parts: List[float] = []
    for regime in regimes:
        selected = [event for event in event_history if event["rho"] == regime]
        if selected:
            weights = np.asarray([event["beta_slow"] for event in selected], dtype=float)
            vectors = np.stack([np.asarray(event["e_cat"], dtype=float) for event in selected])
            pooled = (weights[:, None] * vectors).sum(axis=0) / weights.sum()
            mask = 1.0
        else:
            pooled = np.zeros(event_dim, dtype=float)
            mask = 0.0
        parts.extend(pooled.tolist())
        parts.append(mask)
    return np.asarray(parts, dtype=float)


def build_fast_pool(
    event_history: Sequence[Mapping[str, object]],
    event_dim: int,
    current_time: int | None = None,
    recency_tau: float = 3.0,
):
    if len(event_history) == 0:
        return np.zeros(event_dim, dtype=float), np.asarray([], dtype=float)

    if current_time is None:
        current_time = int(event_history[-1]["t"])

    raw_weights = []
    vectors = []
    for event in event_history:
        age = max(current_time - int(event["t"]), 0) + 1
        recency = np.exp(-age / recency_tau)
        relevance = float(event["task_relevance"])
        action_intensity = 1.0 + 1.5 * float(event["action_intensity"])
        exposure = 1.0 + 0.25 * float(event["weak_exposure"]) * float(event["susceptibility"])
        source_reliability = float(event["source_reliability"])
        raw_weights.append(recency * relevance * action_intensity * exposure * source_reliability)
        vectors.append(np.asarray(event["e_cat"], dtype=float))

    raw_weights = np.asarray(raw_weights, dtype=float)
    if raw_weights.sum() == 0:
        alpha = np.ones_like(raw_weights) / len(raw_weights)
    else:
        alpha = raw_weights / raw_weights.sum()
    vectors = np.stack(vectors)
    pooled = (alpha[:, None] * vectors).sum(axis=0)
    return pooled, alpha
