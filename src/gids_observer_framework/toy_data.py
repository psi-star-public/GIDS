from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .categorical import build_event_categorical_embedding, build_fast_pool, build_slow_bank
from .math_utils import softmax, sigmoid


FAMILIES = ["topic", "objection"]
SOURCES = ["bio", "behavior"]
REGIMES = ["founder", "buyer", "manager"]
EVENT_CAT_DIM = len(FAMILIES) * len(SOURCES) * (2 + 1)
SLOW_BANK_DIM = len(REGIMES) * (EVENT_CAT_DIM + 1)

REGIME_TO_VECTOR = {
    "founder": np.array([1.0, 0.0, 0.0]),
    "buyer": np.array([0.0, 1.0, 0.0]),
    "manager": np.array([0.0, 0.0, 1.0]),
}

PROPOSITIONS = OrderedDict(
    {
        0: {"name": "credibility", "x": np.array([1.0, 0.2, 0.0])},
        1: {"name": "urgency", "x": np.array([0.1, 1.0, 0.2])},
        2: {"name": "price", "x": np.array([0.3, 0.4, 1.0])},
    }
)

PROPOSITION_TOKENS = {
    0: {("topic", "bio"): ["ai"], ("topic", "behavior"): ["security"], ("objection", "bio"): ["trust"], ("objection", "behavior"): []},
    1: {("topic", "bio"): ["pricing"], ("topic", "behavior"): ["ai"], ("objection", "bio"): ["timing"], ("objection", "behavior"): ["timing"]},
    2: {("topic", "bio"): ["pricing"], ("topic", "behavior"): ["pricing"], ("objection", "bio"): ["price"], ("objection", "behavior"): ["price"]},
}


@dataclass
class ToyWeights:
    W_u_true: np.ndarray
    W_slow_true: np.ndarray
    W_u_est: np.ndarray
    W_slow_est: np.ndarray
    A_z: np.ndarray
    B_x: np.ndarray
    C_fast: np.ndarray
    D_T: np.ndarray
    E_c: np.ndarray
    F_w: np.ndarray
    A_z_est: np.ndarray
    B_x_est: np.ndarray
    C_fast_est: np.ndarray
    D_T_est: np.ndarray
    E_c_est: np.ndarray
    F_w_est: np.ndarray
    y_weights: np.ndarray
    y_x: np.ndarray
    y_c: np.ndarray
    y_w: np.ndarray
    probe_weights: np.ndarray


def default_embedding_tables(seed: int = 7):
    rng = np.random.default_rng(seed)
    token_features = {
        "ai": np.array([1.2, 0.1]),
        "security": np.array([0.2, 1.0]),
        "pricing": np.array([0.9, 0.4]),
        "price": np.array([1.0, -0.3]),
        "timing": np.array([-0.5, 0.8]),
        "trust": np.array([0.3, 1.1]),
    }
    embedding_tables = {}
    for family in FAMILIES:
        for source in SOURCES:
            if family == "topic":
                tokens = ["ai", "security", "pricing"]
            else:
                tokens = ["price", "timing", "trust"]
            embedding_tables[(family, source)] = {
                token: token_features[token] + 0.05 * rng.normal(size=2) for token in tokens
            }
    null_vectors = {key: np.zeros(2, dtype=float) for key in embedding_tables}
    projections = {key: np.eye(2, dtype=float) for key in embedding_tables}
    return embedding_tables, null_vectors, projections


def default_weights(u_dim: int, seed: int = 777) -> ToyWeights:
    rng = np.random.default_rng(seed)
    W_u_true = rng.normal(scale=0.45, size=(2, u_dim))
    W_slow_true = rng.normal(scale=0.18, size=(2, SLOW_BANK_DIM))
    W_u_est = W_u_true + rng.normal(scale=0.05, size=(2, u_dim))
    W_slow_est = W_slow_true + rng.normal(scale=0.05, size=(2, SLOW_BANK_DIM))
    A_z = np.array([[0.78, 0.18], [0.12, 0.82]])
    B_x = rng.normal(scale=0.35, size=(2, 3))
    C_fast = rng.normal(scale=0.14, size=(2, EVENT_CAT_DIM))
    D_T = rng.normal(scale=0.25, size=(2, 2))
    E_c = rng.normal(scale=0.20, size=(2, 3))
    F_w = rng.normal(scale=0.20, size=(2, 2))
    A_z_est = A_z + rng.normal(scale=0.04, size=A_z.shape)
    B_x_est = B_x + rng.normal(scale=0.05, size=B_x.shape)
    C_fast_est = C_fast + rng.normal(scale=0.03, size=C_fast.shape)
    D_T_est = D_T + rng.normal(scale=0.05, size=D_T.shape)
    E_c_est = E_c + rng.normal(scale=0.03, size=E_c.shape)
    F_w_est = F_w + rng.normal(scale=0.03, size=F_w.shape)
    y_weights = np.array([1.4, -1.0])
    y_x = np.array([0.1, 0.1, 0.1])
    y_c = np.array([0.1, 0.1, -0.1])
    y_w = np.array([0.1, -0.1])
    probe_weights = np.array([1.0, 0.8])
    return ToyWeights(
        W_u_true=W_u_true,
        W_slow_true=W_slow_true,
        W_u_est=W_u_est,
        W_slow_est=W_slow_est,
        A_z=A_z,
        B_x=B_x,
        C_fast=C_fast,
        D_T=D_T,
        E_c=E_c,
        F_w=F_w,
        A_z_est=A_z_est,
        B_x_est=B_x_est,
        C_fast_est=C_fast_est,
        D_T_est=D_T_est,
        E_c_est=E_c_est,
        F_w_est=F_w_est,
        y_weights=y_weights,
        y_x=y_x,
        y_c=y_c,
        y_w=y_w,
        probe_weights=probe_weights,
    )


def build_person_feature_blocks(person_seed: int):
    rng = np.random.default_rng(person_seed)
    dims = OrderedDict([("p", 2), ("b", 2), ("ell", 2), ("r", 2), ("h", 2), ("ggrp", 2)])
    blocks = OrderedDict((name, rng.normal(size=dim)) for name, dim in dims.items())
    u_vec = np.concatenate(list(blocks.values()))
    return blocks, u_vec


def choose_regime(local_t: int, rng) -> str:
    base = REGIMES[local_t % len(REGIMES)]
    if rng.random() < 0.15:
        return str(rng.choice(REGIMES))
    return base


def temporal_world(global_t: int) -> np.ndarray:
    return np.array([np.sin(global_t / 17.0), np.cos(global_t / 29.0)], dtype=float)


def build_tokens(action: int, regime: str, rng) -> Mapping[Tuple[str, str], Sequence[str]]:
    base = {key: list(value) for key, value in PROPOSITION_TOKENS[action].items()}
    if regime == "founder" and action == 2 and rng.random() < 0.30:
        base[("objection", "behavior")] = base.get(("objection", "behavior"), []) + ["price"]
    return base


def action_policy_logits(T_hat: np.ndarray, z_est: np.ndarray, c_vec: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            0.4 + 0.6 * T_hat[0] + 0.2 * z_est[0] + 0.3 * c_vec[0] - 0.1 * w_vec[0],
            0.2 + 0.2 * T_hat[1] + 0.5 * z_est[1] + 0.4 * c_vec[1] + 0.1 * w_vec[1],
            0.1 - 0.3 * T_hat[0] + 0.3 * T_hat[1] - 0.2 * z_est[0] + 0.5 * c_vec[2] + 0.2 * w_vec[0],
        ],
        dtype=float,
    )


def estimate_slow_embedding_from_arrays(u_vec: np.ndarray, slow_bank: np.ndarray, weights: ToyWeights) -> np.ndarray:
    return np.tanh(weights.W_u_est @ u_vec + weights.W_slow_est @ slow_bank)


def generate_benchmark_dataset(n_people: int = 120, events_per_person: int = 10, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    embedding_tables, null_vectors, projections = default_embedding_tables(seed=seed)
    example_blocks, example_u = build_person_feature_blocks(seed + 17)
    weights = default_weights(u_dim=len(example_u), seed=seed + 11)

    rows = []
    global_time = 0

    for person_id in range(n_people):
        feature_blocks, u_vec = build_person_feature_blocks(seed + person_id * 17)
        history = []
        z_true = np.tanh(rng.normal(scale=0.2, size=2))
        z_est = np.zeros(2, dtype=float)

        for local_t in range(events_per_person):
            regime = choose_regime(local_t, rng)
            c_vec = REGIME_TO_VECTOR[regime]
            w_vec = temporal_world(global_time)
            slow_bank = build_slow_bank(history, regimes=REGIMES, event_dim=EVENT_CAT_DIM)
            T_true = np.tanh(weights.W_u_true @ u_vec + weights.W_slow_true @ slow_bank)
            T_hat = np.tanh(weights.W_u_est @ u_vec + weights.W_slow_est @ slow_bank)
            g_fast, alpha = build_fast_pool(history, event_dim=EVENT_CAT_DIM, current_time=global_time - 1 if history else None)

            logits = action_policy_logits(T_hat=T_hat, z_est=z_est, c_vec=c_vec, w_vec=w_vec)
            mu = softmax(logits)
            action = int(rng.choice([0, 1, 2], p=mu))
            proposition = PROPOSITIONS[action]["x"]

            token_bags = build_tokens(action=action, regime=regime, rng=rng)
            e_cat, _ = build_event_categorical_embedding(
                token_bags=token_bags,
                embedding_tables=embedding_tables,
                null_vectors=null_vectors,
                projections=projections,
                family_order=FAMILIES,
                source_order=SOURCES,
            )

            q_true = np.tanh(
                weights.A_z @ z_true
                + weights.B_x @ proposition
                + weights.C_fast @ g_fast
                + weights.D_T @ T_true
                + weights.E_c @ c_vec
                + weights.F_w @ w_vec
                + rng.normal(scale=0.05, size=2)
            )
            logit_y = -0.4 + weights.y_weights @ q_true + weights.y_x @ proposition + weights.y_c @ c_vec + weights.y_w @ w_vec
            y = int(rng.random() < sigmoid(logit_y))
            logit_probe = -0.1 + weights.probe_weights @ q_true + 0.2 * proposition[1] - 0.1 * w_vec[0]
            probe = int(rng.random() < sigmoid(logit_probe))

            event_record = {
                "t": global_time,
                "rho": regime,
                "e_cat": e_cat,
                "beta_slow": 0.4 + 0.6 * y + 0.1 * (action == 2),
                "task_relevance": 0.6 + 0.3 * probe,
                "action_intensity": 1.0 if y else 0.2,
                "weak_exposure": 1.0 + float(action == 1) + 0.5 * float(history[-1]["y"] if history else 0.0),
                "susceptibility": 1.0 + 0.2 * T_true[1] + 0.2 * float(regime == "buyer"),
                "source_reliability": 0.8 + 0.1 * float(action != 2),
            }

            touch_count = len(history)
            prior_reply_rate = float(np.mean([item["y"] for item in history])) if history else 0.0
            prior_price_count = int(np.sum([item["action"] == 2 for item in history])) if history else 0
            last_delay = 1 if local_t == 0 else global_time - history[-1]["t"]

            recent = history[-3:]
            sequence_features = []
            for lag in range(3):
                if lag < len(recent):
                    item = recent[-1 - lag]
                    sequence_features.extend([item["action"], item["y"], item["probe"]])
                    sequence_features.extend(item["x_vec"].tolist())
                    sequence_features.extend(item["e_cat"][:4].tolist())
                else:
                    sequence_features.extend([0.0, 0.0, 0.0])
                    sequence_features.extend([0.0, 0.0, 0.0])
                    sequence_features.extend([0.0, 0.0, 0.0, 0.0])

            uniform_fast = np.mean([item["e_cat"] for item in history], axis=0) if history else np.zeros(EVENT_CAT_DIM, dtype=float)
            collapsed_slow = np.mean([item["e_cat"] for item in history], axis=0) if history else np.zeros(EVENT_CAT_DIM, dtype=float)
            T_hat_collapsed = np.tanh(weights.W_u_est @ u_vec)

            q_est = np.tanh(
                weights.A_z_est @ z_est
                + weights.B_x_est @ proposition
                + weights.C_fast_est @ g_fast
                + weights.D_T_est @ T_hat
                + weights.E_c_est @ c_vec
                + weights.F_w_est @ w_vec
            )
            q_est_uniform = np.tanh(
                weights.A_z_est @ z_est
                + weights.B_x_est @ proposition
                + weights.C_fast_est @ uniform_fast
                + weights.D_T_est @ T_hat
                + weights.E_c_est @ c_vec
                + weights.F_w_est @ w_vec
            )
            q_est_no_fast = np.tanh(
                weights.B_x_est @ proposition
                + weights.D_T_est @ T_hat
                + weights.E_c_est @ c_vec
                + weights.F_w_est @ w_vec
            )
            q_est_no_slow = np.tanh(
                weights.A_z_est @ z_est
                + weights.B_x_est @ proposition
                + weights.C_fast_est @ g_fast
                + weights.E_c_est @ c_vec
                + weights.F_w_est @ w_vec
            )

            row = {
                "person_id": person_id,
                "global_time": global_time,
                "local_t": local_t,
                "regime": regime,
                "y": y,
                "probe": probe,
                "prop_action": action,
                "mu_propensity": float(mu[action]),
                "touch_count": touch_count,
                "prior_reply_rate": prior_reply_rate,
                "prior_price_count": prior_price_count,
                "last_delay": last_delay,
                "cold_start": int(local_t == 0),
                "w0": w_vec[0],
                "w1": w_vec[1],
                "c0": c_vec[0],
                "c1": c_vec[1],
                "c2": c_vec[2],
                "x0": proposition[0],
                "x1": proposition[1],
                "x2": proposition[2],
                "T0": T_hat[0],
                "T1": T_hat[1],
                "Tcol0": T_hat_collapsed[0],
                "Tcol1": T_hat_collapsed[1],
                "z0": z_est[0],
                "z1": z_est[1],
                "qest0": q_est[0],
                "qest1": q_est[1],
                "q_nofast0": q_est_no_fast[0],
                "q_nofast1": q_est_no_fast[1],
                "q_noslow0": q_est_no_slow[0],
                "q_noslow1": q_est_no_slow[1],
                "q_uniform0": q_est_uniform[0],
                "q_uniform1": q_est_uniform[1],
            }
            row.update({f"u{i}": value for i, value in enumerate(u_vec)})
            row.update({f"gfast{i}": value for i, value in enumerate(g_fast[:6])})
            row.update({f"slow{i}": value for i, value in enumerate(slow_bank[:6])})
            row.update({f"seq{i}": value for i, value in enumerate(sequence_features)})
            rows.append(row)

            z_true = q_true
            z_est = np.tanh(
                weights.A_z_est @ z_est
                + weights.B_x_est @ proposition
                + weights.C_fast_est @ (g_fast + 0.1 * e_cat)
                + weights.D_T_est @ T_hat
                + weights.E_c_est @ c_vec
                + weights.F_w_est @ w_vec
                + 0.05 * np.array([y, probe], dtype=float)
            )
            history.append({"y": y, "probe": probe, "action": action, "x_vec": proposition, "e_cat": e_cat, **event_record})
            global_time += 1

    df = pd.DataFrame(rows)
    df["u_x_0"] = df["u0"] * df["x0"] + df["u1"] * df["x0"]
    df["u_x_1"] = df["u0"] * df["x1"] + df["u1"] * df["x1"]
    df["u_x_2"] = df["u0"] * df["x2"] + df["u1"] * df["x2"]
    return df


def default_feature_sets(df: pd.DataFrame):
    u_cols = [column for column in df.columns if column.startswith("u") and column[1:].isdigit()]
    seq_cols = [column for column in df.columns if column.startswith("seq")]
    c_cols = ["c0", "c1", "c2"]
    w_cols = ["w0", "w1"]
    x_cols = ["x0", "x1", "x2"]
    history_cols = ["touch_count", "prior_reply_rate", "prior_price_count", "last_delay"]
    return OrderedDict(
        {
            "current_touch": x_cols + c_cols,
            "static": u_cols + c_cols + w_cols + x_cols,
            "shallow_history": u_cols + c_cols + w_cols + x_cols + history_cols,
            "two_tower_like": u_cols + x_cols + ["u_x_0", "u_x_1", "u_x_2"],
            "monolithic_sequence": u_cols + c_cols + w_cols + x_cols + history_cols + seq_cols,
            "latent_full": ["T0", "T1", "qest0", "qest1"] + c_cols + w_cols + x_cols,
            "latent_no_fast": ["T0", "T1", "q_nofast0", "q_nofast1"] + c_cols + w_cols + x_cols,
            "latent_no_slow": ["q_noslow0", "q_noslow1"] + c_cols + w_cols + x_cols,
            "latent_uniform_pool": ["T0", "T1", "q_uniform0", "q_uniform1"] + c_cols + w_cols + x_cols,
            "latent_collapsed_slow": ["Tcol0", "Tcol1", "qest0", "qest1"] + c_cols + w_cols + x_cols,
        }
    )
