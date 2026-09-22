"""Reject calendar-empty domains before quadratic MILP model construction."""

import importlib
import json
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling import Problem, solve_milp, validate_assignment

milp_module = importlib.import_module("pyqpanda_alg.LabScheduling.milp")


def _problem(start: int, cleanup: int, downtime: list[list[int]]) -> Problem:
    return Problem.from_dict(
        {
            "name": "calendar-domain",
            "horizon": 6,
            "resources": [{"id": "r", "cleanup": cleanup, "downtime": downtime}],
            "tasks": [
                {
                    "id": "blocked",
                    "group": "g0",
                    "duration": 2,
                    "options": [{"resource": "r", "start": start, "cost": 0}],
                },
                {
                    "id": "other",
                    "group": "g1",
                    "duration": 1,
                    "options": [{"resource": "r", "start": 0, "cost": 0}],
                },
            ],
        }
    )


@pytest.mark.parametrize(
    "start, cleanup, downtime",
    [(1, 0, [[2, 3]]), (4, 1, []), (0, 1, [[2, 3]])],
    ids=["downtime-overlap", "cleanup-exceeds-horizon", "cleanup-overlaps-downtime"],
)
def test_calendar_infeasibility_skips_pairs_and_solver(
    monkeypatch: pytest.MonkeyPatch,
    start: int,
    cleanup: int,
    downtime: list[list[int]],
) -> None:
    """A raw calendar witness needs neither pair rows nor a solver invocation."""
    problem = _problem(start, cleanup, downtime)
    assert any(
        row["kind"] == "calendar" and row["task"] == "blocked"
        for row in validate_assignment(problem, [0, 0])["violations"]
    )

    def unnecessary(*args: object, **kwargs: object) -> None:
        pytest.fail("calendar infeasibility must be checked before pairs and MILP")

    monkeypatch.setattr(milp_module, "_incompatible", unnecessary)
    monkeypatch.setattr(milp_module, "milp", unnecessary)
    result = solve_milp(problem)
    assert result["status"] == "infeasible"
    assert result["infeasible_task"] == "blocked"
    assert result["best"] is None and result["dual_bound"] is None
    assert result["gap"] is None and result["constraints"] == 0
    assert "solver_status" not in result


def test_touching_calendar_boundary_still_runs_real_milp() -> None:
    """Half-open downtime and cleanup boundaries must not remove a valid choice."""
    problem = _problem(3, 1, [[2, 3]])
    result = solve_milp(problem)
    assert result["status"] == "optimal" and result["solver_status"] == 0
    assert result["best"]["feasible"] and result["best"]["objective"] == 0


def test_one_remaining_raw_option_preserves_actual_optimization() -> None:
    """An unavailable cheap option must not hide a remaining feasible option."""
    problem = Problem.from_dict(
        {
            "name": "mixed-calendar",
            "horizon": 6,
            "resources": [{"id": "r", "downtime": [[0, 2]]}],
            "tasks": [
                {
                    "id": "task",
                    "group": "group",
                    "duration": 1,
                    "options": [
                        {"resource": "r", "start": 0, "cost": 0},
                        {"resource": "r", "start": 2, "cost": 3},
                    ],
                }
            ],
        }
    )
    result = solve_milp(problem)
    assert result["status"] == "optimal" and result["solver_status"] == 0
    assert result["best"]["choices"] == [1]
    assert result["best"]["objective"] == 3


def test_cli_keeps_large_calendar_infeasibility_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Many raw options with empty calendar domains still produce exit 3 and JSON."""
    from pyqpanda_alg.LabScheduling.__main__ import main

    data = {
        "name": "large-calendar-empty",
        "horizon": 2,
        "resources": [{"id": "r", "downtime": [[0, 2]]}],
        "tasks": [
            {
                "id": f"t{i}",
                "group": f"g{i}",
                "duration": 1,
                "options": [{"resource": "r", "start": 0, "cost": 0}],
            }
            for i in range(512)
        ],
    }
    source, output = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 3
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == result
    assert result["status"] == "infeasible"
    assert result["quantum"]["circuit_evaluations"] == 0
    assert result["milp"]["variables"] == 512
    assert result["milp"]["constraints"] == 0
    assert result["milp"]["infeasible_task"] == "t0"
    assert result["absolute_gap"] is None


@pytest.mark.parametrize("blocked_start", [0, 1])
def test_fixed_zero_candidates_need_no_pair_rows(blocked_start: int) -> None:
    """Retain raw indices and the optimum while omitting bound-redundant rows."""
    problem = Problem.from_dict(
        {
            "name": "fixed-zero-pairs",
            "horizon": 6,
            "resources": [{"id": "r", "downtime": [[0, 2]]}],
            "tasks": [
                {
                    "id": f"t{i}",
                    "group": f"g{i}",
                    "duration": 2,
                    "options": [
                        {"resource": "r", "start": blocked_start, "cost": 0},
                        {"resource": "r", "start": 2, "cost": 3 if i == 0 else 1},
                        {"resource": "r", "start": 4, "cost": 1 if i == 0 else 3},
                    ],
                }
                for i in range(2)
            ],
        }
    )
    result = solve_milp(problem)
    assert result["status"] == "optimal" and result["variables"] == 6
    assert result["best"]["choices"] == [2, 1]
    assert result["best"]["objective"] == 2
    assert validate_assignment(problem, [2, 1])["feasible"]
    # Two exact-one rows and two incompatible pairs of available options.
    assert result["constraints"] == 4
