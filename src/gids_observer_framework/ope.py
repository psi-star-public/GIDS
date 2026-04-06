from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def ips_value(
    rewards: Sequence[float],
    actions: Sequence[int],
    logged_propensities: Sequence[float],
    target_actions: Sequence[int] | None = None,
    target_action_probabilities: Sequence[float] | None = None,
) -> float:
    rewards = np.asarray(rewards, dtype=float)
    actions = np.asarray(actions)
    propensities = np.asarray(logged_propensities, dtype=float)

    if target_action_probabilities is not None:
        target_probs = np.asarray(target_action_probabilities, dtype=float)
        weights = target_probs / propensities
    elif target_actions is not None:
        target_actions = np.asarray(target_actions)
        weights = (actions == target_actions).astype(float) / propensities
    else:
        raise ValueError("Provide either target_actions for a deterministic policy or target_action_probabilities for a stochastic policy.")

    return float(np.mean(weights * rewards))


def simulate_ips_demo(seed: int = 123, n: int = 200_000):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=n)

    def mu_prob(s):
        return 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * s)))

    def reward_prob(s, action):
        return 1.0 / (1.0 + np.exp(-(-0.5 + 0.7 * s + 1.0 * action - 1.2 * action * s)))

    behavior_probs = mu_prob(state)
    actions = rng.binomial(1, behavior_probs)
    rewards = rng.binomial(1, reward_prob(state, actions))
    target_actions = (state < 0).astype(int)

    ips_estimate = ips_value(
        rewards=rewards,
        actions=actions,
        logged_propensities=np.where(actions == 1, behavior_probs, 1.0 - behavior_probs),
        target_actions=target_actions,
    )

    true_value = float(np.mean(reward_prob(state, target_actions)))
    naive_logged_mean = float(np.mean(rewards[actions == target_actions]))

    return {
        "n": n,
        "ips_estimate": ips_estimate,
        "true_value": true_value,
        "naive_logged_mean": naive_logged_mean,
    }
