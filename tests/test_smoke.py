from pathlib import Path

from gids_observer_framework.experiments.run_all import run_all


def test_run_all_smoke(tmp_path):
    outputs = run_all(tmp_path)
    assert not outputs["equation_checks"].empty
    assert not outputs["main_results"].empty
