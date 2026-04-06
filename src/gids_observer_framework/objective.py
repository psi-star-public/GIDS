from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .math_utils import binary_cross_entropy_from_logits, sigmoid


def total_loss(main_loss: float, probe_losses: Sequence[float], lam_probes: Sequence[float], reg_term: float, lam_reg: float) -> float:
    if len(probe_losses) != len(lam_probes):
        raise ValueError("probe_losses and lam_probes must have the same length.")
    return float(main_loss + np.sum(np.asarray(lam_probes, dtype=float) * np.asarray(probe_losses, dtype=float)) + lam_reg * reg_term)


LOSS_FORMS = {
    "current_minus_minus": "L = main - λ_probe * probe - λ_reg * reg",
    "plus_plus": "L = main + λ_probe * probe + λ_reg * reg",
    "plus_probe_minus_reg": "L = main + λ_probe * probe - λ_reg * reg",
    "minus_probe_plus_reg": "L = main - λ_probe * probe + λ_reg * reg",
    "main_only": "L = main + λ_reg * reg",
    "plus_plus_no_reg": "L = main + λ_probe * probe",
}


def _loss_coefficients(form: str, lam_probe: float, lam_reg: float):
    if form == "current_minus_minus":
        return 1.0, -lam_probe, -lam_reg
    if form == "plus_plus":
        return 1.0, lam_probe, lam_reg
    if form == "plus_probe_minus_reg":
        return 1.0, lam_probe, -lam_reg
    if form == "minus_probe_plus_reg":
        return 1.0, -lam_probe, lam_reg
    if form == "main_only":
        return 1.0, 0.0, lam_reg
    if form == "plus_plus_no_reg":
        return 1.0, lam_probe, 0.0
    raise ValueError(f"Unknown loss form: {form}")


def _generate_loss_search_data(n: int = 800, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.3, -0.8, 0.6])[:d]
    probe_offset = np.array([0.1, 0.05, -0.05])[:d]
    main_logit = X @ w_true + 0.2 * rng.normal(size=n)
    probe_logit = X @ (0.9 * w_true + probe_offset) + 0.2 * rng.normal(size=n) - 0.2
    y_main = (main_logit > 0).astype(float)
    y_probe = (probe_logit > 0).astype(float)
    idx = rng.permutation(n)
    train_idx = idx[: int(0.7 * n)]
    val_idx = idx[int(0.7 * n) :]
    return X[train_idx], y_main[train_idx], y_probe[train_idx], X[val_idx], y_main[val_idx], y_probe[val_idx]


def run_loss_search(lam_probe_values=(0.1, 0.5, 1.0), lam_reg_values=(1e-3, 1e-2), seeds=range(10), lr: float = 0.2, steps: int = 400):
    rows = []
    for form, equation in LOSS_FORMS.items():
        for lam_probe in lam_probe_values:
            for lam_reg in lam_reg_values:
                for seed in seeds:
                    X_train, y_main_train, y_probe_train, X_val, y_main_val, y_probe_val = _generate_loss_search_data(seed=seed)
                    n, d = X_train.shape
                    rng = np.random.default_rng(seed + 12345)
                    w = rng.normal(scale=0.1, size=d)
                    b_main = 0.0
                    b_probe = 0.0
                    c_main, c_probe, c_reg = _loss_coefficients(form, lam_probe, lam_reg)

                    for _ in range(steps):
                        logits_main = X_train @ w + b_main
                        logits_probe = X_train @ w + b_probe
                        grad_main = (sigmoid(logits_main) - y_main_train) / n
                        grad_probe = (sigmoid(logits_probe) - y_probe_train) / n
                        grad_w = c_main * (X_train.T @ grad_main) + c_probe * (X_train.T @ grad_probe) + c_reg * w
                        grad_b_main = c_main * grad_main.sum() + c_reg * b_main
                        grad_b_probe = c_probe * grad_probe.sum() + c_reg * b_probe
                        w -= lr * grad_w
                        b_main -= lr * grad_b_main
                        b_probe -= lr * grad_b_probe

                    val_main_logits = X_val @ w + b_main
                    val_probe_logits = X_val @ w + b_probe
                    rows.append(
                        {
                            "experiment": "loss_search",
                            "form": form,
                            "equation": equation,
                            "lam_probe": lam_probe,
                            "lam_reg": lam_reg,
                            "seed": seed,
                            "val_main_bce": binary_cross_entropy_from_logits(val_main_logits, y_main_val),
                            "val_probe_bce": binary_cross_entropy_from_logits(val_probe_logits, y_probe_val),
                            "weight_norm": float(np.linalg.norm(w)),
                        }
                    )
    return rows
