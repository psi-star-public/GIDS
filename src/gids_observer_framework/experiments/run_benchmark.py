from __future__ import annotations

import pandas as pd

from ..benchmark import run_benchmark, run_cold_start_slice, run_person_holdout_cold_start
from ..ope import simulate_ips_demo
from ..toy_data import generate_benchmark_dataset


def run_benchmark_suite(seed: int = 2026):
    dataset = generate_benchmark_dataset(seed=seed)
    main_results, predictions, splits = run_benchmark(dataset, target="y")
    probe_results, _, _ = run_benchmark(dataset, target="probe")
    train_df, val_df, test_df = splits
    cold_start_results = run_cold_start_slice(train_df, test_df, target="y")
    person_holdout_results = run_person_holdout_cold_start(dataset, target="y")
    ips_demo = pd.DataFrame([simulate_ips_demo(seed=seed)])
    return {
        "dataset": dataset,
        "main_results": main_results,
        "probe_results": probe_results,
        "cold_start_results": cold_start_results,
        "person_holdout_results": person_holdout_results,
        "ips_results": ips_demo,
    }
