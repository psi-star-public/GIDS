from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .toy_data import default_feature_sets


def temporal_split(df: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.20):
    df = df.sort_values("global_time").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()


def fit_binary_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, features: Sequence[str], target: str = "y"):
    if len(features) == 0:
        probs = np.repeat(train_df[target].mean(), len(test_df))
        return None, probs
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[list(features)], train_df[target])
    probs = model.predict_proba(test_df[list(features)])[:, 1]
    return model, probs


def evaluate_probabilities(y_true, probs):
    return {
        "logloss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "auc": float(roc_auc_score(y_true, probs)),
    }


def run_benchmark(df: pd.DataFrame, target: str = "y"):
    feature_sets = default_feature_sets(df)
    train_df, val_df, test_df = temporal_split(df)
    rows = []
    predictions = {}

    _, frequency_probs = fit_binary_classifier(train_df, test_df, [], target=target)
    predictions["frequency"] = frequency_probs
    row = {"model": "frequency", "target": target, **evaluate_probabilities(test_df[target], frequency_probs)}
    rows.append(row)

    for name, features in feature_sets.items():
        model, probs = fit_binary_classifier(train_df, test_df, features, target=target)
        predictions[name] = probs
        row = {"model": name, "target": target, **evaluate_probabilities(test_df[target], probs)}
        rows.append(row)

    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True), predictions, (train_df, val_df, test_df)


def run_cold_start_slice(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "y"):
    feature_sets = default_feature_sets(test_df)
    cold_mask = test_df["cold_start"] == 1
    rows = []

    _, frequency_probs = fit_binary_classifier(train_df, test_df, [], target=target)
    rows.append({"model": "frequency", "target": target, **evaluate_probabilities(test_df.loc[cold_mask, target], frequency_probs[cold_mask])})

    for name, features in feature_sets.items():
        _, probs = fit_binary_classifier(train_df, test_df, features, target=target)
        rows.append({"model": name, "target": target, **evaluate_probabilities(test_df.loc[cold_mask, target], probs[cold_mask])})

    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def run_person_holdout_cold_start(df: pd.DataFrame, target: str = "y", holdout_frac: float = 0.2):
    person_ids = sorted(df["person_id"].unique())
    split = int(len(person_ids) * (1.0 - holdout_frac))
    train_people = set(person_ids[:split])
    test_people = set(person_ids[split:])
    train_df = df[df["person_id"].isin(train_people)].copy()
    test_df = df[df["person_id"].isin(test_people) & (df["cold_start"] == 1)].copy()
    feature_sets = default_feature_sets(df)
    rows = []

    _, frequency_probs = fit_binary_classifier(train_df, test_df, [], target=target)
    rows.append({"model": "frequency", "target": target, **evaluate_probabilities(test_df[target], frequency_probs)})

    for name, features in feature_sets.items():
        _, probs = fit_binary_classifier(train_df, test_df, features, target=target)
        rows.append({"model": name, "target": target, **evaluate_probabilities(test_df[target], probs)})

    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
