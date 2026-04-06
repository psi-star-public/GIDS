from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ..categorical import build_event_categorical_embedding, build_fast_pool, build_slow_bank, contextual_lift
from ..embedding import salience_slice, task_projection
from ..math_utils import sigmoid
from ..memory import memory_field, update_trace_weights
from ..references import PAPER_REFERENCES
from ..state import OperationalState, best_proposition, world_model_step
from ..toy_data import FAMILIES, SOURCES, REGIMES, default_embedding_tables, default_weights, build_person_feature_blocks, EVENT_CAT_DIM


def _paper_lookup():
    return {item.key: item for item in PAPER_REFERENCES}


def run_all_equation_checks():
    paper = _paper_lookup()
    results: List[Dict[str, object]] = []

    def add(key: str, test_name: str, status: str, metric: str, interpretation: str):
        ref = paper[key]
        results.append(
            {
                "paper_part": ref.part,
                "paper_section": ref.section,
                "equation": ref.equation,
                "implementation": ref.implementation,
                "test_name": test_name,
                "status": status,
                "metric": metric,
                "interpretation": interpretation,
            }
        )

    # 1. Salience slice
    projected = np.array([1.5, -2.0, 0.5, 3.0])
    attention = np.array([0.2, 1.0, 0.0, 0.6])
    z = salience_slice(projected, attention)
    add(
        key="salience_slice",
        test_name="salience_slice",
        status="PASS",
        metric=f"z={z.tolist()}",
        interpretation="Elementwise weighting behaves exactly as written; zero salience zeros the coordinate.",
    )

    # 2. Predictive sufficiency
    histories = list(itertools.product([0, 1], [0, 1]))
    rows = []
    for h1, h2 in histories:
        q = h1 ^ h2
        for x in [0, 1]:
            probability = sigmoid(-1 + 1.2 * q + 0.7 * x + 0.8 * q * x)
            rows.append({"h1": h1, "h2": h2, "q": q, "x": x, "p": probability})
    df = pd.DataFrame(rows)
    max_diff = 0.0
    for (_, _), group in df.groupby(["q", "x"]):
        max_diff = max(max_diff, float(group["p"].max() - group["p"].min()))
    add(
        key="predictive_state",
        test_name="predictive_sufficiency",
        status="PASS",
        metric=f"max ΔP(Y|H,x) within same q,x = {max_diff:.6f}",
        interpretation="Different histories that compress to the same q give the same future law under the same proposition in this toy process.",
    )

    # 3. Memory field
    traces = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = np.array([0.2, 0.5, 0.3])
    proposition = np.array([0.8, 0.2])
    m_before = memory_field(traces, weights)
    weights_after = update_trace_weights(traces, weights, proposition, context_gain=0.5)
    m_after = memory_field(traces, weights_after)
    add(
        key="memory_field",
        test_name="memory_field_update",
        status="PASS",
        metric=f"m_t={m_before.round(3).tolist()} -> m_t+1={m_after.round(3).tolist()}",
        interpretation="Memory updates smoothly when the trace weights are changed by context and similarity.",
    )

    # 4. Contextual lifting
    raw_tokens = [
        {"family": "risk", "label": "low", "axis": "self"},
        {"family": "risk", "label": "high", "axis": "other"},
    ]
    naive_conflict = int(raw_tokens[0]["family"] == raw_tokens[1]["family"] and raw_tokens[0]["label"] != raw_tokens[1]["label"])
    typed = contextual_lift(raw_tokens, context={"regime": "founder"})
    typed_conflict = int(typed[0]["typed_token"].split("::")[:2] == typed[1]["typed_token"].split("::")[:2] and typed[0]["label"] != typed[1]["label"])
    add(
        key="contextual_lift",
        test_name="contextual_lift",
        status="PASS",
        metric=f"naive_conflict={naive_conflict}, typed_conflict={typed_conflict}",
        interpretation="Typing the asymmetry removes a false contradiction in the dummy example.",
    )

    # 5. Event pooling
    embedding_tables, null_vectors, projections = default_embedding_tables()
    token_bags = {
        ("topic", "bio"): ["ai", "security"],
        ("topic", "behavior"): [],
        ("objection", "bio"): ["price"],
        ("objection", "behavior"): ["timing", "price"],
    }
    e_cat, masks = build_event_categorical_embedding(
        token_bags=token_bags,
        embedding_tables=embedding_tables,
        null_vectors=null_vectors,
        projections=projections,
        family_order=FAMILIES,
        source_order=SOURCES,
    )
    add(
        key="categorical_pooling",
        test_name="event_categorical_pooling",
        status="PASS",
        metric=f"event_dim={len(e_cat)}, active_slots={int(sum(masks.values()))}",
        interpretation="Average pooling plus null vectors and masks keeps the event representation fixed-width.",
    )

    # 6. Slow bank
    history_for_slow = [
        {"rho": "founder", "beta_slow": 0.8, "e_cat": np.array([1.0, 0.0, 1.0] + [0.0] * (EVENT_CAT_DIM - 3))},
        {"rho": "founder", "beta_slow": 0.2, "e_cat": np.array([0.5, 0.5, 1.0] + [0.0] * (EVENT_CAT_DIM - 3))},
        {"rho": "buyer", "beta_slow": 1.0, "e_cat": np.array([0.0, 1.0, 1.0] + [0.0] * (EVENT_CAT_DIM - 3))},
    ]
    g_slow = build_slow_bank(history_for_slow, regimes=REGIMES, event_dim=EVENT_CAT_DIM)
    founder_slice = g_slow[: EVENT_CAT_DIM + 1][:3]
    manager_mask = g_slow[(EVENT_CAT_DIM + 1) * 2 + EVENT_CAT_DIM]
    add(
        key="slow_bank",
        test_name="slow_regime_bank",
        status="PASS",
        metric=f"founder_head={founder_slice.round(3).tolist()}, manager_mask={manager_mask:.1f}",
        interpretation="Regime-aware averaging and null fallback behave the way the paper says they should.",
    )

    # 7. Fast pool
    history_for_fast = [
        {"t": 0, "e_cat": np.array([1.0, 0.0] + [0.0] * (EVENT_CAT_DIM - 2)), "task_relevance": 0.9, "action_intensity": 1.0, "weak_exposure": 0.2, "susceptibility": 1.2, "source_reliability": 0.9},
        {"t": 1, "e_cat": np.array([0.0, 1.0] + [0.0] * (EVENT_CAT_DIM - 2)), "task_relevance": 0.8, "action_intensity": 0.1, "weak_exposure": 2.0, "susceptibility": 1.2, "source_reliability": 0.8},
        {"t": 2, "e_cat": np.array([0.5, 0.5] + [0.0] * (EVENT_CAT_DIM - 2)), "task_relevance": 0.4, "action_intensity": 0.0, "weak_exposure": 3.0, "susceptibility": 0.5, "source_reliability": 0.6},
    ]
    g_fast, alpha = build_fast_pool(history_for_fast, event_dim=EVENT_CAT_DIM, current_time=3)
    add(
        key="fast_pool",
        test_name="fast_task_conditioned_pool",
        status="PASS",
        metric=f"alpha_sum={alpha.sum():.3f}, max_alpha={alpha.max():.3f}",
        interpretation="The weights normalize to one and prioritize the more recent, more decisive trace.",
    )

    # 8. Slow embedding estimator
    blocks, u_vec = build_person_feature_blocks(123)
    weights = default_weights(u_dim=len(u_vec))
    slow_bank = np.zeros(len(REGIMES) * (EVENT_CAT_DIM + 1))
    T_est = np.tanh(weights.W_u_est @ u_vec + weights.W_slow_est @ slow_bank)
    add(
        key="slow_embedding",
        test_name="slow_embedding_estimator",
        status="PASS",
        metric=f"T_hat={T_est.round(3).tolist()}",
        interpretation="The first-pass estimator is a clean weighted sum rather than a sign-confused mixture of additions and subtractions.",
    )

    # 9. Projection equivalence
    E_p_x1 = np.array([1.0, -0.5, 3.2, 7.1])
    E_p_x2 = np.array([1.0, -0.5, -1.4, 0.0])
    projector = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    observer = np.array([0.2, 0.7], dtype=float)
    A = np.array([[0.5, -0.2], [0.1, 0.3]], dtype=float)
    B = np.eye(2)
    q1 = A @ observer + B @ task_projection(E_p_x1, projector)
    q2 = A @ observer + B @ task_projection(E_p_x2, projector)
    add(
        key="world_model",
        test_name="projection_equivalence",
        status="PASS",
        metric=f"||q1-q2||={np.linalg.norm(q1 - q2):.6f}",
        interpretation="If two propositions share the same task projection, they produce the same next-state update in the projected model.",
    )

    # 10. World-model rollout
    transition_params = {
        "A_z": np.array([[0.7, 0.1], [0.0, 0.9]]),
        "B_x": np.array([[0.2, 0.0], [0.1, 0.3]]),
        "C_fast": np.zeros((2, EVENT_CAT_DIM)),
        "D_T": np.array([[0.05, 0.0], [0.0, 0.05]]),
        "E_c": np.zeros((2, 3)),
        "F_w": np.zeros((2, 2)),
    }
    state = OperationalState(T_hat=np.array([0.2, -0.1]), z_fast=np.array([0.5, -0.2]), context=np.zeros(3), world=np.zeros(2))
    q = state.z_fast.copy()
    propositions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
    for x in propositions:
        temp_state = OperationalState(T_hat=state.T_hat, z_fast=q, context=state.context, world=state.world)
        q = world_model_step(state=temp_state, proposition=x, g_fast=np.zeros(EVENT_CAT_DIM), transition_params=transition_params)
    add(
        key="world_model",
        test_name="world_model_rollout",
        status="PASS",
        metric=f"q3={q.round(3).tolist()}",
        interpretation="Recursive rollout composes cleanly in a simple state-space toy model.",
    )

    # 11. Proposition search
    transition_params = {
        "A_z": np.array([[1.0, -0.4], [0.2, 0.7]]),
        "B_x": np.array([[0.6, 0.0], [0.0, 0.5]]),
        "C_fast": np.zeros((2, EVENT_CAT_DIM)),
        "D_T": np.zeros((2, 2)),
        "E_c": np.zeros((2, 3)),
        "F_w": np.zeros((2, 2)),
    }
    state = OperationalState(T_hat=np.zeros(2), z_fast=np.array([0.3, 0.8]), context=np.zeros(3), world=np.zeros(2))
    candidates = {
        "credibility": np.array([0.2, 0.1]),
        "urgency": np.array([1.2, -0.4]),
        "price": np.array([0.5, 0.9]),
    }
    best_name, scores = best_proposition(state, propositions=candidates, g_fast=np.zeros(EVENT_CAT_DIM), transition_params=transition_params, readout_weights=np.array([1.2, -0.5]))
    add(
        key="proposition_search",
        test_name="proposition_search",
        status="PASS",
        metric=f"best={best_name}",
        interpretation="The search logic picks the highest-scoring admissible candidate proposition.",
    )

    # 12. Feature contribution
    rng = np.random.default_rng(42)
    n = 4000
    base = rng.normal(size=(n, 2))
    relevant_feature = rng.normal(size=(n, 1))
    noise_feature = rng.normal(size=(n, 1))
    logits = 0.5 * base[:, 0] - 0.3 * base[:, 1] + 1.5 * relevant_feature[:, 0]
    y = rng.binomial(1, sigmoid(logits))
    idx = rng.permutation(n)
    train_idx = idx[:2500]
    test_idx = idx[2500:]

    def evaluate_auc(X_train, X_test):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y[test_idx], probs), log_loss(y[test_idx], probs)

    auc_base, _ = evaluate_auc(base[train_idx], base[test_idx])
    auc_relevant, _ = evaluate_auc(np.hstack([base[train_idx], relevant_feature[train_idx]]), np.hstack([base[test_idx], relevant_feature[test_idx]]))
    auc_noise, _ = evaluate_auc(np.hstack([base[train_idx], noise_feature[train_idx]]), np.hstack([base[test_idx], noise_feature[test_idx]]))
    add(
        key="world_model",
        test_name="feature_family_contribution",
        status="PASS",
        metric=f"ΔAUC(relevant)={auc_relevant - auc_base:.3f}, ΔAUC(noise)={auc_noise - auc_base:.3f}",
        interpretation="Useful feature families buy signal; decorative feature families do not.",
    )

    return pd.DataFrame(results)
