"""Keep quantum evidence when the independent raw MILP exceeds its budget."""

import json

import pytest
from pyqpanda_alg.LabScheduling.__main__ import main
from pyqpanda_alg.LabScheduling.report import run_experiment


def _problem(raw_count, tasks=1, available=True):
    raw = {
        "name": "raw options versus surviving quantum candidates",
        "horizon": 3 * tasks,
        "resources": [{"id": "r"}],
        "tasks": [
            {
                "id": f"t{i}",
                "group": f"g{i}",
                "duration": 1,
                "options": [
                    {"resource": "r", "start": 3 * i + j, "cost": j} for j in range(2)
                ],
            }
            for i in range(tasks)
        ],
    }
    raw["tasks"][0]["options"].extend(
        {"resource": "r", "start": 100 + i, "cost": 0}
        for i in range(raw_count - 2 * tasks)
    )
    if not available:
        raw["resources"][0]["downtime"] = [[0, raw["horizon"]]]
    return raw


@pytest.mark.parametrize("raw_count", [2048, 2049])
def test_raw_candidate_boundary_retains_quantum_and_exact_certificate(
    tmp_path, raw_count
):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_problem(raw_count)), encoding="utf-8")
    result = run_experiment(path, shots=32, maxiter=6, restarts=1)
    assert result["model"]["qubits"] == 2
    assert len(result["model"]["pruned"]) == raw_count - 2
    assert result["status"] == "feasible"
    assert result["quantum"]["circuit_evaluations"] > 0
    assert result["quantum"]["business_metrics"]["feasible"]
    assert result["quantum"]["best"]["objective"] == 0
    assert result["exact"]["best"]["objective"] == 0
    assert result["absolute_gap"] == 0
    if raw_count == 2048:
        assert result["milp"]["status"] == "optimal"
        assert result["milp"]["best"]["objective"] == 0
    else:
        assert result["milp"]["status"] == "budget_exceeded"
        assert result["milp"]["variables"] == 2049
        assert result["milp"]["max_variables"] == 2048
        assert result["milp"]["best"] is None


def test_no_exact_certificate_means_unknown_gap_not_zero(tmp_path, monkeypatch):
    import pyqpanda_alg.LabScheduling.report as module

    def forbidden_call(*args, **kwargs):
        raise AssertionError("raw MILP must not be constructed above its budget")

    monkeypatch.setattr(module, "solve_milp", forbidden_call)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_problem(2049, tasks=21)), encoding="utf-8")
    result = run_experiment(path, reduce=True, shots=2, maxiter=6, restarts=1)
    assert result["status"] == "feasible"
    assert result["quantum"]["business_metrics"]["feasible"]
    assert len(result["quantum"]["components"]) == 21
    assert all(c["circuit_evaluations"] > 0 for c in result["quantum"]["components"])
    assert result["exact"]["status"] == "budget_exceeded"
    assert result["milp"]["status"] == "budget_exceeded"
    assert all(result["milp"][key] is None for key in ("best", "dual_bound", "gap"))
    assert result["absolute_gap"] is None
    assert result["uniform_one_hot"]["feasible_probability"] is None


def test_empty_domain_remains_infeasible_when_milp_is_skipped(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_problem(2049, available=False)), encoding="utf-8")
    result = run_experiment(path, reduce=True, shots=2, maxiter=6, restarts=1)
    assert result["status"] == "infeasible"
    assert result["exact"]["status"] == "infeasible"
    assert result["milp"]["status"] == "budget_exceeded"
    assert result["quantum"]["circuit_evaluations"] == 0
    assert result["absolute_gap"] is None


def test_cli_saves_report_instead_of_treating_budget_as_bad_input(tmp_path, capsys):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(_problem(2049)), encoding="utf-8")
    before = source.read_bytes()
    status = main(
        [str(source), "--maxiter", "6", "--restarts", "1", "--output", str(output)]
    )
    assert status == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "feasible"
    assert result["milp"]["status"] == "budget_exceeded"
    assert source.read_bytes() == before
