from __future__ import annotations

from pathlib import Path

import pandas as pd

from .run_benchmark import run_benchmark_suite
from .run_candidate_search import run_candidate_search
from .run_toy_equations import run_all_equation_checks


def summarise_candidate_winners(loss_df: pd.DataFrame, slow_df: pd.DataFrame):
    loss_summary = (
        loss_df.groupby(["form", "equation"], as_index=False)[["val_main_bce", "val_probe_bce", "weight_norm"]]
        .mean()
        .sort_values(["val_main_bce", "val_probe_bce", "weight_norm"])
        .reset_index(drop=True)
    )
    slow_summary = (
        slow_df[slow_df["experiment"] == "slow_update_search"]
        .groupby(["form", "equation"], as_index=False)[["rmse", "avg_step", "cp_error"]]
        .mean()
        .sort_values(["rmse", "cp_error"])
        .reset_index(drop=True)
    )
    shock_summary = (
        slow_df[slow_df["experiment"] == "slow_update_shock"]
        .groupby(["form", "equation"], as_index=False)[["shock_deviation", "recovery_error"]]
        .mean()
        .sort_values(["shock_deviation", "recovery_error"])
        .reset_index(drop=True)
    )
    return loss_summary, slow_summary, shock_summary


def run_all(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equation_checks = run_all_equation_checks()
    equation_checks.to_csv(output_dir / "toy_equation_checks.csv", index=False)

    loss_df, slow_df = run_candidate_search()
    loss_df.to_csv(output_dir / "loss_search.csv", index=False)
    slow_df.to_csv(output_dir / "slow_update_search.csv", index=False)

    loss_summary, slow_summary, shock_summary = summarise_candidate_winners(loss_df, slow_df)
    loss_summary.to_csv(output_dir / "loss_search_summary.csv", index=False)
    slow_summary.to_csv(output_dir / "slow_update_summary.csv", index=False)
    shock_summary.to_csv(output_dir / "slow_update_shock_summary.csv", index=False)

    benchmark = run_benchmark_suite()
    benchmark["dataset"].to_csv(output_dir / "toy_benchmark_dataset.csv", index=False)
    benchmark["main_results"].to_csv(output_dir / "benchmark_main_results.csv", index=False)
    benchmark["probe_results"].to_csv(output_dir / "benchmark_probe_results.csv", index=False)
    benchmark["cold_start_results"].to_csv(output_dir / "benchmark_cold_start_results.csv", index=False)
    benchmark["person_holdout_results"].to_csv(output_dir / "benchmark_person_holdout_results.csv", index=False)
    benchmark["ips_results"].to_csv(output_dir / "ips_results.csv", index=False)

    return {
        "equation_checks": equation_checks,
        "loss_search": loss_df,
        "slow_update_search": slow_df,
        "loss_summary": loss_summary,
        "slow_summary": slow_summary,
        "shock_summary": shock_summary,
        **benchmark,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    results_dir = project_root / "results"
    run_all(results_dir)
    print(f"Wrote results to {results_dir}")
