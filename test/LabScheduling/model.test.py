"""Exhaustive semantic/QUBO equality and input-boundary regression checks."""

import copy
import json
from itertools import product
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling import (
    Problem,
    compile_qubo,
    evaluate,
    load_problem,
    solve_exact,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm/example/LabScheduling/data"
)


def fixture_data():
    return json.loads((DATA / "simple.json").read_text(encoding="utf-8"))


def test_all_binary_energies_equal_independent_constraint_formula():
    model = compile_qubo(load_problem(DATA / "simple.json"))
    feasible_energies, infeasible_energies = [], []
    for bits in product((0, 1), repeat=len(model.variables)):
        checked = evaluate(model, bits)
        assignment_penalty = sum(
            (sum(bits[i] for i in domain) - 1) ** 2 for domain in model.domains
        )
        pair_count = sum(v["kind"] == "conflict" for v in checked["violations"])
        assert model.energy(bits) == checked["raw_cost"] + model.penalty * (
            assignment_penalty + pair_count
        )
        (feasible_energies if checked["feasible"] else infeasible_energies).append(
            model.energy(bits)
        )
    assert min(infeasible_energies) > max(feasible_energies)
    exact = solve_exact(model)
    assert exact["best"]["objective"] == min(feasible_energies) == 5
    assert exact["feasible_count"] == 4


def test_medium_all_4096_states_and_exact_optimum():
    model = compile_qubo(load_problem(DATA / "medium.json"))
    assert len(model.variables) == 12
    assert len(model.pruned) == 2
    for bits in product((0, 1), repeat=12):
        checked = evaluate(model, bits)
        one_hot = sum(
            (sum(bits[i] for i in domain) - 1) ** 2 for domain in model.domains
        )
        pairs = sum(v["kind"] == "conflict" for v in checked["violations"])
        assert model.energy(bits) == checked["raw_cost"] + model.penalty * (
            one_hot + pairs
        )
    exact = solve_exact(model)
    assert exact["best"]["objective"] == 4
    assert exact["best"]["violation_count"] == 0


def test_cleanup_downtime_horizon_and_touching_intervals():
    data = fixture_data()
    # At start=1, occupation [1,4) includes cleanup and touches downtime [0,1).
    data["tasks"] = [data["tasks"][0]]
    data["tasks"][0]["options"].append(
        {"resource": "spectrometer", "start": 7, "cost": 0}
    )
    model = compile_qubo(Problem.from_dict(data))
    assert [v.option.start for v in model.variables] == [1, 3, 6]
    assert [row["reasons"] for row in model.pruned] == [["downtime"], ["horizon"]]
    data["resources"][0]["downtime"] = [[3, 4]]
    model = compile_qubo(Problem.from_dict(data))
    assert 1 not in [v.option.start for v in model.variables]  # cleanup also blocked


def test_pair_multiple_reasons_penalized_once_and_precedence_end_boundary():
    data = fixture_data()
    for task in data["tasks"]:
        task["group"] = "same"
    data["precedence"] = [["sample-A", "sample-B"]]
    model = compile_qubo(Problem.from_dict(data))
    bits = [1, 0, 0, 1, 0, 0]
    result = evaluate(model, bits)
    assert result["violation_count"] == 1
    assert result["violations"][0]["reasons"] == ["resource", "group", "precedence"]
    assert model.energy(bits) == result["raw_cost"] + model.penalty
    # Separate resources: group/precedence allow immediate start at predecessor end.
    data["resources"].append({"id": "other"})
    data["tasks"][1]["options"] = [{"resource": "other", "start": 3, "cost": 0}]
    model = compile_qubo(Problem.from_dict(data))
    assert evaluate(model, [1, 0, 0, 1])["feasible"]


def test_valid_but_infeasible_and_empty_domain_and_cycle():
    model = compile_qubo(load_problem(DATA / "infeasible.json"))
    assert solve_exact(model)["status"] == "infeasible"
    data = fixture_data()
    data["tasks"][0]["options"] = []
    model = compile_qubo(Problem.from_dict(data))
    assert solve_exact(model)["combinations"] == 0
    assert solve_exact(model)["best"] is None
    data = fixture_data()
    data["precedence"] = [["sample-A", "sample-B"], ["sample-B", "sample-A"]]
    assert solve_exact(compile_qubo(Problem.from_dict(data)))["status"] == "infeasible"


def test_empty_problem_and_singleton_zero_cost():
    data = {"name": "empty", "horizon": 1, "resources": [{"id": "r"}], "tasks": []}
    model = compile_qubo(Problem.from_dict(data))
    assert model.energy([]) == 0
    assert solve_exact(model)["best"]["objective"] == 0
    data["tasks"] = [
        {
            "id": "t",
            "group": "g",
            "duration": 1,
            "options": [{"resource": "r", "start": 0, "cost": 0}],
        }
    ]
    model = compile_qubo(Problem.from_dict(data))
    assert model.penalty == 1
    assert model.energy([1]) == 0
    assert model.energy([0]) == 1


def test_limits_penalty_and_decode_errors():
    model = compile_qubo(load_problem(DATA / "simple.json"))
    for penalty in [model.objective_bound, 0, -1, True, 2.5]:
        with pytest.raises(ValueError, match="penalty"):
            compile_qubo(model.problem, penalty=penalty)
    for bits in [[], [0] * 7, [2] * 6]:
        with pytest.raises(ValueError, match="binary"):
            evaluate(model, bits)
    with pytest.raises(ValueError, match="limit"):
        solve_exact(model, max_combinations=1)
    with pytest.raises(ValueError):
        solve_exact(model, max_combinations=False)
    assert compile_qubo(model.problem, penalty=20).penalty == 20


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(horizon=True),
        lambda d: d.update(horizon=0),
        lambda d: d.update(typo=1),
        lambda d: d.update(resources=[]),
        lambda d: d.update(tasks="bad"),
        lambda d: d["tasks"][0].update(duration=-1),
        lambda d: d["tasks"][0].update(duration=1.5),
        lambda d: d["tasks"][0].update(group=" "),
        lambda d: d["tasks"][0]["options"][0].update(cost=-1),
        lambda d: d["tasks"][0]["options"][0].update(cost=float("nan")),
        lambda d: d["tasks"][0]["options"][0].update(resource="missing"),
        lambda d: d["tasks"][0]["options"].append(
            copy.deepcopy(d["tasks"][0]["options"][0])
        ),
        lambda d: d["tasks"].append(copy.deepcopy(d["tasks"][0])),
        lambda d: d["resources"].append(copy.deepcopy(d["resources"][0])),
        lambda d: d["resources"][0].update(downtime=[[2, 2]]),
        lambda d: d["resources"][0].update(downtime=[[0, 20]]),
        lambda d: d["resources"][0].update(downtime=[[1]]),
        lambda d: d.update(precedence=[["missing", "sample-B"]]),
        lambda d: d.update(precedence=[["sample-A", "sample-A"]]),
        lambda d: d.update(precedence=[["sample-A"]]),
        lambda d: d["tasks"][0].pop("original"),
    ],
)
def test_invalid_input(mutation):
    data = fixture_data()
    mutation(data)
    with pytest.raises(ValueError):
        Problem.from_dict(data)


@pytest.mark.parametrize("text", ['{"name":"a","name":"b"}', '{"name":NaN}', "{bad"])
def test_invalid_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_problem(path)


def test_model_compilation_rejects_excessive_candidates_before_pair_expansion():
    data = {
        "name": "large",
        "horizon": 300,
        "resources": [{"id": "r"}],
        "tasks": [
            {
                "id": "t",
                "group": "g",
                "duration": 1,
                "options": [
                    {"resource": "r", "start": i, "cost": 0} for i in range(257)
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="256"):
        compile_qubo(Problem.from_dict(data))
