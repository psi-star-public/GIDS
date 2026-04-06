from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .math_utils import sigmoid


@dataclass
class OperationalState:
    T_hat: np.ndarray
    z_fast: np.ndarray
    context: np.ndarray
    world: np.ndarray


def world_model_step(
    state: OperationalState,
    proposition: np.ndarray,
    g_fast: np.ndarray,
    transition_params: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.tanh(
        transition_params["A_z"] @ np.asarray(state.z_fast, dtype=float)
        + transition_params["B_x"] @ np.asarray(proposition, dtype=float)
        + transition_params["C_fast"] @ np.asarray(g_fast, dtype=float)
        + transition_params["D_T"] @ np.asarray(state.T_hat, dtype=float)
        + transition_params["E_c"] @ np.asarray(state.context, dtype=float)
        + transition_params["F_w"] @ np.asarray(state.world, dtype=float)
    )


def readout_probability(q_next: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> float:
    return float(sigmoid(np.asarray(weights, dtype=float) @ np.asarray(q_next, dtype=float) + bias))


def proposition_score(
    state: OperationalState,
    proposition: np.ndarray,
    g_fast: np.ndarray,
    transition_params: Mapping[str, np.ndarray],
    readout_weights: np.ndarray,
    utility_fn: Callable[[float], float] | None = None,
) -> float:
    q_next = world_model_step(state=state, proposition=proposition, g_fast=g_fast, transition_params=transition_params)
    probability = readout_probability(q_next, readout_weights)
    if utility_fn is None:
        return probability
    return float(utility_fn(probability))


def best_proposition(
    state: OperationalState,
    propositions: Mapping[str, np.ndarray],
    g_fast: np.ndarray,
    transition_params: Mapping[str, np.ndarray],
    readout_weights: np.ndarray,
    utility_fn: Callable[[float], float] | None = None,
):
    scores = {
        name: proposition_score(
            state=state,
            proposition=vector,
            g_fast=g_fast,
            transition_params=transition_params,
            readout_weights=readout_weights,
            utility_fn=utility_fn,
        )
        for name, vector in propositions.items()
    }
    best_name = max(scores, key=scores.get)
    return best_name, scores


def ema_slow_update(current_embedding: np.ndarray, new_target_embedding: np.ndarray, alpha: float) -> np.ndarray:
    current_embedding = np.asarray(current_embedding, dtype=float)
    new_target_embedding = np.asarray(new_target_embedding, dtype=float)
    return (1.0 - alpha) * current_embedding + alpha * new_target_embedding
