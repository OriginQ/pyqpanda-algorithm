"""Weight invariants, tied optima, retained misses and real CPU replay."""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling import Problem

EXAMPLES = (
    Path(__file__).resolve().parents[2] / "pyqpanda-algorithm/example/LabScheduling"
)
sys.path.insert(0, str(EXAMPLES))
try:
    sensitivity = importlib.import_module("sensitivity")
    demo = importlib.import_module("tradeoff_demo")
finally:
    sys.path.pop(0)


def base_input():
    return json.loads((EXAMPLES / "data/tradeoff.json").read_text())


def test_frozen_inputs_preserve_every_feasible_schedule_and_certify_switch():
    base = base_input()
    untouched = copy.deepcopy(base)
    config = json.loads((EXAMPLES / "data/sensitivity.json").read_text())
    assert config["weights"] == list(range(7))
    assert config["seeds"] == [7, 19, 42]
    assert (
        hashlib.sha256((EXAMPLES / "data/tradeoff.json").read_bytes()).hexdigest()
        == config["base_sha256"]
    )
    reference = None
    for weight in config["weights"]:
        raw = sensitivity.weighted_input(base, weight)
        assert all(t["change_cost"] == weight for t in raw["tasks"])
        reverted = copy.deepcopy(raw)
        for task in reverted["tasks"]:
            task["change_cost"] = 2
        assert reverted == base
        rows = sensitivity.feasible_schedules(Problem.from_dict(raw))
        choices = [r["choices"] for r in rows]
        assert len(choices) == 9
        assert reference is None or choices == reference
        reference = choices
        for row in rows:
            assert (
                row["objective"]
                == row["preference_cost"] + weight * row["changed_tasks"]
            )
        optimum = min(r["objective"] for r in rows)
        assert optimum == min(6 + 3 * weight, 12 + weight)
        optimal = [r["choices"] for r in rows if r["objective"] == optimum]
        assert optimal == (
            [[1, 1, 1]]
            if weight < 3
            else [[2, 0, 0]]
            if weight > 3
            else [[1, 1, 1], [2, 0, 0]]
        )
    assert base == untouched
    for value in (-1, True, 1.5, 1_000_001):
        with pytest.raises(ValueError, match="weight"):
            sensitivity.weighted_input(base, value)


def diagnostic_report(row, optimum):
    # Deliberately synthetic diagnostic inputs; no quantum run is mocked.
    return {
        "status": "feasible",
        "runtime_seconds": 1,
        "absolute_gap": row["objective"] - optimum,
        "quantum": {
            "seed": 7,
            "business_metrics": row,
            "best": {
                "feasible": True,
                "objective": row["objective"],
                "violation_count": 0,
            },
        },
    }


def test_accepts_both_tied_optima_and_retains_nonoptimal_sample():
    rows = sensitivity.feasible_schedules(
        Problem.from_dict(sensitivity.weighted_input(base_input(), 3))
    )
    optimal = [r for r in rows if r["objective"] == 15]
    assert len(optimal) == 2
    for row in optimal:
        metrics = sensitivity.sample_metrics(diagnostic_report(row, 15), rows)
        assert metrics["optimal"] and metrics["absolute_gap"] == 0
        assert metrics["choices"] == row["choices"]
    suboptimal = next(r for r in rows if r["objective"] > 15)
    metrics = sensitivity.sample_metrics(diagnostic_report(suboptimal, 15), rows)
    assert not metrics["optimal"] and metrics["absolute_gap"] > 0
    assert metrics["objective"] == suboptimal["objective"]
    altered = diagnostic_report(optimal[0], 15)
    altered["quantum"]["business_metrics"] = {**optimal[0], "change_cost": 999}
    with pytest.raises(ValueError, match="business audit"):
        sensitivity.sample_metrics(altered, rows)


@pytest.mark.parametrize(
    "best", [None, {"feasible": False, "violation_count": 2, "objective": None}]
)
def test_sampling_failure_keeps_empty_business_fields(best):
    rows = sensitivity.feasible_schedules(Problem.from_dict(base_input()))
    report = {
        "status": "no_feasible_sample",
        "absolute_gap": None,
        "runtime_seconds": 1,
        "quantum": {"seed": 42, "best": best},
    }
    result = sensitivity.sample_metrics(report, rows)
    assert not result["optimal"]
    assert all(
        result[k] is None
        for k in (
            "objective",
            "choices",
            "change_cost",
            "changed_tasks",
            "absolute_gap",
        )
    )


def test_actual_twenty_one_cpu_runs_replay_and_reject_tampering(tmp_path, capsys):
    first = sensitivity.run_sensitivity(tmp_path / "first")
    second = sensitivity.run_sensitivity(
        tmp_path / "second", verify_against=tmp_path / "first/results.json"
    )
    capsys.readouterr()
    assert len(first["runs"]) == 21
    assert json.loads((tmp_path / "second/replay.json").read_text())[
        "identical_except_timing"
    ]
    for row in second["weights"]:
        assert len(row["alternatives"]) == 9
        assert len(row["samples"]) == 3
    for run in first["runs"]:
        for component in run["report"]["quantum"]["components"]:
            key = "".join(map(str, component["best"]["bits"][::-1]))
            assert component["counts"][key] > 0
    # A different tied optimum is accepted by scoring, but is not an exact replay.
    corrupted = copy.deepcopy(first)
    corrupted["weights"][3]["samples"][0]["choices"] = next(
        choice
        for choice in first["weights"][3]["optimal_choices"]
        if choice != first["weights"][3]["samples"][0]["choices"]
    )
    with pytest.raises(ValueError, match="beyond timing"):
        sensitivity.verify_replay(tmp_path / "first/results.json", corrupted)
    corrupted = copy.deepcopy(first)
    corrupted["source_sha256"]["protocol"] = "changed"
    with pytest.raises(ValueError, match="beyond timing"):
        sensitivity.verify_replay(tmp_path / "first/results.json", corrupted)
    assert (tmp_path / "first/summary.csv").read_text().count("\n") == 8
    assert (tmp_path / "first/sensitivity.svg").is_file()


def test_demo_rejects_replay_flag_without_sensitivity(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["tradeoff_demo", "--verify-sensitivity", "missing.json"]
    )
    with pytest.raises(SystemExit) as error:
        demo.main()
    assert error.value.code == 2
    assert "requires --sensitivity" in capsys.readouterr().err
