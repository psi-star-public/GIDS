from __future__ import annotations

import numpy as np
import pandas as pd

from ..objective import run_loss_search
from ..state import ema_slow_update


UPDATE_FORMS = {
    "ema_target": r"\hat T \leftarrow (1-\alpha)\hat T + \alpha\hat T^{new}",
    "additive_delta": r"\hat T \leftarrow \hat T + \alpha (\hat T^{new} - \hat T)",
    "literal_current_delta": r"\hat T \leftarrow (1-\alpha)\hat T + \alpha (\hat T^{new} - \hat T)",
    "overwrite_target": r"\hat T \leftarrow \hat T^{new}",
    "no_update": r"\hat T \leftarrow \hat T",
}


def simulate_true_slow(T: int = 120, d: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    true = np.zeros((T, d), dtype=float)
    centers = [
        np.array([0.0, 0.5, -0.5, 0.2]),
        np.array([0.8, 0.2, -0.2, 0.4]),
        np.array([1.0, -0.3, 0.1, 0.7]),
    ]
    change_points = [0, 40, 80]
    current = centers[0].copy()
    center_idx = 0
    for t in range(T):
        if center_idx + 1 < len(change_points) and t >= change_points[center_idx + 1]:
            center_idx += 1
        target = centers[center_idx]
        current = 0.95 * current + 0.05 * target + rng.normal(scale=0.01, size=d)
        true[t] = current
    return true


def generate_evidence(true_path: np.ndarray, noise: float = 0.18, seed: int = 0, shock: bool = False, shock_t: int = 60, shock_mag: float = 2.0):
    rng = np.random.default_rng(seed + 1)
    evidence = true_path + rng.normal(scale=noise, size=true_path.shape)
    if shock:
        shock_vec = rng.normal(size=true_path.shape[1])
        shock_vec = shock_mag * shock_vec / np.linalg.norm(shock_vec)
        evidence[shock_t] += shock_vec
    return evidence


def run_slow_update_search(alpha_values=(0.02, 0.05, 0.1, 0.2, 0.4), seeds=range(50)):
    rows = []
    for form, equation in UPDATE_FORMS.items():
        for alpha in alpha_values:
            for seed in seeds:
                true_path = simulate_true_slow(seed=seed)
                evidence = generate_evidence(true_path, seed=seed)
                estimate = np.zeros(true_path.shape[1], dtype=float)
                history = []

                for target in evidence:
                    delta = target - estimate
                    if form == "ema_target":
                        estimate = ema_slow_update(estimate, target, alpha)
                    elif form == "additive_delta":
                        estimate = estimate + alpha * delta
                    elif form == "literal_current_delta":
                        estimate = (1.0 - alpha) * estimate + alpha * delta
                    elif form == "overwrite_target":
                        estimate = target.copy()
                    elif form == "no_update":
                        estimate = estimate
                    else:
                        raise ValueError(form)
                    history.append(estimate.copy())
                history = np.asarray(history)

                rows.append(
                    {
                        "experiment": "slow_update_search",
                        "form": form,
                        "equation": equation,
                        "alpha": alpha,
                        "seed": seed,
                        "rmse": float(np.sqrt(((history - true_path) ** 2).mean())),
                        "avg_step": float(np.linalg.norm(np.diff(history, axis=0), axis=1).mean()),
                        "cp_error": float(np.mean([np.linalg.norm(history[t] - true_path[t]) for t in [40, 41, 42, 80, 81, 82]])),
                    }
                )

    for form in ["ema_target", "additive_delta", "literal_current_delta", "overwrite_target"]:
        equation = UPDATE_FORMS[form]
        for alpha in (0.05, 0.1, 0.2, 0.4):
            for seed in range(100):
                true_path = simulate_true_slow(seed=seed)
                evidence = generate_evidence(true_path, seed=seed, shock=True)
                estimate = np.zeros(true_path.shape[1], dtype=float)
                history = []

                for target in evidence:
                    delta = target - estimate
                    if form == "ema_target":
                        estimate = ema_slow_update(estimate, target, alpha)
                    elif form == "additive_delta":
                        estimate = estimate + alpha * delta
                    elif form == "literal_current_delta":
                        estimate = (1.0 - alpha) * estimate + alpha * delta
                    elif form == "overwrite_target":
                        estimate = target.copy()
                    history.append(estimate.copy())

                history = np.asarray(history)
                rows.append(
                    {
                        "experiment": "slow_update_shock",
                        "form": form,
                        "equation": equation,
                        "alpha": alpha,
                        "seed": seed,
                        "shock_deviation": float(np.linalg.norm(history[60] - true_path[60])),
                        "recovery_error": float(np.mean([np.linalg.norm(history[t] - true_path[t]) for t in [61, 62, 63, 64]])),
                    }
                )
    return pd.DataFrame(rows)


def run_candidate_search():
    loss_df = pd.DataFrame(run_loss_search())
    slow_df = run_slow_update_search()
    return loss_df, slow_df
