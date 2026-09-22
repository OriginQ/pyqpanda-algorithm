"""Lossless reduction, independent raw MILP and real circuit equivalence."""

from dataclasses import replace
from itertools import product
from pathlib import Path
from random import Random
from types import SimpleNamespace

import numpy as np
import pytest
from pyqpanda_alg.LabScheduling import (
    Problem,
    circuit_probabilities,
    compile_qubo,
    evaluate,
    load_problem,
    reduce_model,
    solve_exact,
    solve_milp,
    solve_reduced,
    validate_assignment,
)
from pyqpanda_alg.LabScheduling.metrics import (
    circuit_resources,
    distribution_metrics,
    hit_probability,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm/example/LabScheduling/data"
)


def random_problem(seed):
    rng = Random(seed)
    tasks = []
    for ti in range(rng.randint(0, 5)):
        assignments = rng.sample(
            [(r, s) for r in ("r0", "r1") for s in range(6)],
            rng.randint(0 if seed % 9 == 0 else 1, 3),
        )
        tasks.append(
            {
                "id": f"t{ti}",
                "group": f"g{rng.randrange(2)}",
                "duration": rng.randint(1, 2),
                "change_cost": rng.randrange(4),
                "original": {"resource": "r0", "start": ti},
                "options": [
                    {"resource": r, "start": s, "cost": rng.randrange(5)}
                    for r, s in assignments
                ],
            }
        )
    edges = [
        [a["id"], b["id"]]
        for a in tasks
        for b in tasks
        if a != b and rng.random() < 0.07
    ]
    return Problem.from_dict(
        {
            "name": f"case-{seed}",
            "horizon": 7,
            "resources": [
                {"id": "r0", "cleanup": seed % 2, "downtime": [[0, 1]]},
                {"id": "r1"},
            ],
            "tasks": tasks,
            "precedence": edges,
        }
    )


def feasible_vectors(model):
    result = {}
    for choice in product(*model.domains):
        bits = [int(i in choice) for i in range(len(model.variables))]
        row = evaluate(model, bits)
        if row["feasible"]:
            result[tuple(bits)] = row["objective"]
    return result


def test_one_hundred_instances_preserve_every_feasible_schedule_and_raw_milp_optimum():
    for seed in range(100):
        problem = random_problem(seed)
        model = compile_qubo(problem)
        original = feasible_vectors(model)
        reduced = reduce_model(model)
        restored = {}
        if reduced.infeasible_task is None:
            options = [list(feasible_vectors(m)) for m in reduced.components]
            for choices in product(*options):
                row = reduced.restore(choices)
                assert row["feasible"], seed
                restored[tuple(row["bits"])] = row["objective"]
                local_cost = sum(
                    evaluate(m, b)["objective"]
                    for m, b in zip(reduced.components, choices, strict=True)
                )
                assert row["objective"] == local_cost + reduced.audit()["fixed_cost"]
        assert restored == original, seed
        milp = solve_milp(problem)
        assert milp["status"] == ("optimal" if original else "infeasible"), seed
        if original:
            assert milp["best"]["objective"] == min(original.values()), seed


@pytest.mark.parametrize("dataset", ["simple", "medium"])
def test_full_and_reduced_xy_phase_match_for_thirty_real_cpu_parameter_sets(dataset):
    model = compile_qubo(load_problem(DATA / f"{dataset}.json"))
    rng = np.random.default_rng(721)
    for layers in (1, 2, 3):
        for _ in range(10):
            parameters = rng.uniform(-3, 3, 2 * layers).tolist()
            assert circuit_probabilities(model, parameters) == pytest.approx(
                circuit_probabilities(model, parameters, phase_mode="full"), abs=1e-10
            )
    resources = circuit_resources(model, [0.3, 0.7])
    assert resources["auto"]["cnot"] < resources["full"]["cnot"]
    assert resources["auto"]["depth"] < resources["full"]["depth"]
    assert (
        circuit_resources(model, [0.3, 0.7], "x")["auto"]
        == circuit_resources(model, [0.3, 0.7], "x")["full"]
    )


def test_mixed_singleton_domains_and_x_phase_unmodified():
    p = load_problem(DATA / "simple.json")
    p = replace(
        p, tasks=(replace(p.tasks[0], options=(p.tasks[0].options[-1],)), p.tasks[1])
    )
    model = compile_qubo(p)
    for mixer in ("xy", "x"):
        assert circuit_probabilities(
            model, [0.3, -0.9, 1.1, 0.7], mixer
        ) == pytest.approx(
            circuit_probabilities(
                model, [0.3, -0.9, 1.1, 0.7], mixer, phase_mode="full"
            ),
            abs=1e-10,
        )
    with pytest.raises(ValueError, match="phase_mode"):
        circuit_probabilities(model, [0.3, 0.7], phase_mode="bad")
    from pyqpanda_alg.LabScheduling import solve_qaoa

    with pytest.raises(ValueError, match="phase_mode"):
        solve_qaoa(model, phase_mode="bad")


def chain_problem():
    return Problem.from_dict(
        {
            "name": "chain",
            "horizon": 5,
            "resources": [{"id": "r", "downtime": [[0, 1]]}],
            "tasks": [
                {
                    "id": f"t{i}",
                    "group": f"g{i}",
                    "duration": 1,
                    "original": {"resource": "r", "start": i},
                    "change_cost": i + 1,
                    "options": [
                        {"resource": "r", "start": s, "cost": 0} for s in (i, i + 1)
                    ],
                }
                for i in range(3)
            ],
        }
    )


def test_outage_cascades_through_unaffected_bookings_and_restores_cost():
    model = compile_qubo(chain_problem())
    reduced = reduce_model(model)
    assert len(reduced.fixed) == 3
    assert len(reduced.removals) == 2
    assert reduced.components == ()
    assert reduced.restore([])["objective"] == 6
    result = solve_reduced(model)
    assert result["status"] == "classical_propagation"
    assert result["circuit_evaluations"] == 0
    assert result["best"]["objective"] == 6
    assert all(row["changed"] for row in result["best"]["schedule"])
    with pytest.raises(ValueError, match="per component"):
        reduced.restore([[0]])
    with pytest.raises(ValueError, match="seed"):
        solve_reduced(model, seed=-1)


def test_reduction_infeasible_and_actual_component_sampling():
    infeasible = compile_qubo(load_problem(DATA / "infeasible.json"))
    assert solve_reduced(infeasible)["status"] == "infeasible_propagation"
    with pytest.raises(ValueError, match="infeasible"):
        reduce_model(infeasible).restore([])
    model = compile_qubo(load_problem(DATA / "simple.json"))
    result = solve_reduced(model, maxiter=20, restarts=1)
    assert result["status"] == "feasible"
    assert result["best"]["objective"] >= solve_exact(model)["best"]["objective"]
    for run in result["components"]:
        key = "".join(map(str, run["best"]["bits"][::-1]))
        assert run["counts"][key] > 0


def test_disconnected_components_respect_largest_component_budget():
    p = chain_problem()
    tasks = tuple(
        replace(
            p.tasks[1],
            id=f"t{i}",
            group=f"g{i}",
            options=tuple(
                replace(o, start=3 * i + j + 1)
                for j, o in enumerate(p.tasks[1].options)
            ),
        )
        for i in range(9)
    )
    model = compile_qubo(replace(p, horizon=40, tasks=tasks))
    reduced = reduce_model(model)
    assert reduced.audit()["original_qubits"] == 18
    assert reduced.audit()["component_qubits"] == [2] * 9
    result = solve_reduced(model, maxiter=6, restarts=1)
    assert result["best"]["feasible"]
    assert result["total_shots"] == 512 * 9
    task = replace(
        tasks[0],
        options=tuple(replace(tasks[0].options[0], start=i + 1) for i in range(17)),
    )
    big = compile_qubo(replace(p, horizon=40, tasks=(task,)))
    assert solve_reduced(big)["status"] == "resource_limit"


def test_probability_diagnostics_ties_infeasibility_and_validation():
    p = chain_problem()
    model = compile_qubo(replace(p, tasks=(), precedence=()))
    metrics = distribution_metrics(model, np.array([1.0]))
    assert metrics["optimal_count"] == 1
    assert metrics["shots_for_95_percent"] == 1
    model = compile_qubo(load_problem(DATA / "infeasible.json"))
    metrics = distribution_metrics(model, circuit_probabilities(model, [0.0, 0.0]))
    assert metrics["optimal_probability"] == 0
    assert metrics["shots_for_95_percent"] is None
    model = compile_qubo(load_problem(DATA / "simple.json"))
    metrics = distribution_metrics(model, circuit_probabilities(model, [0.0, 0.0]))
    assert metrics["optimal_probability"] == pytest.approx(1 / 9)
    assert metrics["uniform_optimal_probability"] == pytest.approx(1 / 9)
    assert hit_probability(1 / 81, 512) == pytest.approx(0.998271204298638)
    for p in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            hit_probability(p, 8)
    with pytest.raises(ValueError):
        hit_probability(0.2, 0)
    for array in (np.array([1.0]), np.full(64, -1), np.full(64, np.nan), np.ones(64)):
        with pytest.raises(ValueError):
            distribution_metrics(model, array)


def test_raw_validator_catches_unary_error_and_choice_shape():
    problem = chain_problem()
    assert not validate_assignment(problem, [0, 0, 0])["feasible"]
    assert not validate_assignment(problem, [1, 0, 0])["feasible"]
    assert validate_assignment(problem, [1, 1, 1])["changed_tasks"] == 3
    for choices in ([], [99, 0, 0], [True, 0, 0]):
        with pytest.raises(ValueError):
            validate_assignment(problem, choices)
    for limit in (0, -1, True, float("nan")):
        with pytest.raises(ValueError):
            solve_milp(problem, time_limit=limit)


def test_milp_limit_keeps_status_separate_from_optimality(monkeypatch):
    import importlib

    module = importlib.import_module("pyqpanda_alg.LabScheduling.milp")
    monkeypatch.setattr(
        module,
        "milp",
        lambda *a, **kw: SimpleNamespace(
            status=1,
            x=None,
            message="time limit",
            mip_dual_bound=0,
            mip_gap=None,
        ),
    )
    result = solve_milp(chain_problem())
    assert result["status"] == "limit"
    assert result["best"] is None and result["dual_bound"] == 0
    assert result["gap"] is None


def test_tied_optima_and_greedy_dead_end():
    from pyqpanda_alg.LabScheduling.classical import solve_greedy

    p = chain_problem()
    task = replace(p.tasks[1], original=None, change_cost=0)
    model = compile_qubo(replace(p, tasks=(task,)))
    metrics = distribution_metrics(model, np.array([0, 0.5, 0.5, 0]))
    assert metrics["optimal_count"] == 2
    assert metrics["optimal_probability"] == 1
    assert solve_greedy(model)["best"]["objective"] == 0
    infeasible = compile_qubo(load_problem(DATA / "infeasible.json"))
    assert solve_greedy(infeasible)["status"] == "dead_end"


def test_booking_csv_expands_windows_costs_and_preserves_invalid_candidates(tmp_path):
    from pyqpanda_alg.LabScheduling.bookings import COLUMNS, import_bookings

    path = tmp_path / "bookings.csv"
    header = ",".join(COLUMNS) + "\n"
    path.write_text(header + "a,g,1,r,0,r,0,2,1,2,3\n")
    calendar = {
        "name": "csv",
        "horizon": 3,
        "resources": [{"id": "r", "downtime": [[0, 1]]}],
    }
    raw = import_bookings(path, calendar)
    assert [o["cost"] for o in raw["tasks"][0]["options"]] == [0, 3, 6]
    assert len(compile_qubo(Problem.from_dict(raw)).pruned) == 1
    invalid = [
        ("bad-header\n", "headers"),
        (header + "a,g\n", "missing"),
        (header + "a,g,1,r,0,r,0,2,1,2,3,extra\n", "extra"),
        (header + "a,g,1,r,0,r,0,2,0,2,3\n", "step"),
        (header + "a,g,1,r,0,r,3,2,1,2,3\n", "earliest"),
        (header + "a,g,1,r,0,r,0,2,1,-2,3\n", "integer"),
        (header + "a,g,1,r,0,r,0,2,1,1000001,3\n", "1000000"),
        (header + "a,g,1,r,0,r,0,256,1,2,3\n", "256"),
        (header + "a,g,1,r,0,r|r,0,2,1,2,3\n", "duplicate"),
    ]
    for content, match in invalid:
        path.write_text(content)
        with pytest.raises(ValueError, match=match):
            import_bookings(path, calendar)
    with pytest.raises(ValueError, match="calendar"):
        import_bookings(path, {"tasks": []})


def test_reduced_report_and_independent_business_audit(tmp_path):
    import json

    from pyqpanda_alg.LabScheduling.bookings import import_bookings
    from pyqpanda_alg.LabScheduling.report import run_experiment

    calendar = json.loads((DATA / "bookings/outage.json").read_text())
    raw = import_bookings(DATA / "bookings/chain.csv", calendar)
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(raw))
    report = run_experiment(path, reduce=True, maxiter=20, restarts=1)
    assert report["status"] == "feasible"
    assert report["quantum"]["business_metrics"]["changed_tasks"] == 3
    assert report["milp"]["best"]["objective"] == 6
    assert report["quantum"]["reduction"]["max_component_qubits"] == 4


def test_large_exact_budget_is_reported_and_milp_certifies(tmp_path):
    import json

    from pyqpanda_alg.LabScheduling.report import run_experiment

    raw = {
        "name": "20 independent bookings",
        "horizon": 61,
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
            for i in range(20)
        ],
    }
    path = tmp_path / "large.json"
    path.write_text(json.dumps(raw))
    report = run_experiment(path, reduce=True, shots=2, maxiter=6, restarts=1)
    assert report["exact"]["status"] == "budget_exceeded"
    assert report["milp"]["status"] == "optimal"
    assert report["uniform_one_hot"]["feasible_probability"] is None
    assert report["status"] == "feasible"


def test_milp_resource_budget_and_rejects_corrupted_incumbents(monkeypatch):
    import importlib

    module = importlib.import_module("pyqpanda_alg.LabScheduling.milp")
    p = chain_problem()
    one = replace(p, tasks=(replace(p.tasks[1], options=(p.tasks[1].options[0],)),))
    oversized = replace(
        one, tasks=tuple(replace(one.tasks[0], id=f"t{i}") for i in range(2049))
    )
    with pytest.raises(ValueError, match="2048"):
        solve_milp(oversized)
    for x, value, message in (
        (np.array([0.5]), 1, "integral"),
        (np.array([0]), 0, "exact-one"),
        (np.array([1]), 999, "semantic"),
        (None, None, "without an incumbent"),
    ):
        monkeypatch.setattr(
            module,
            "milp",
            lambda *a, x=x, value=value, **kw: SimpleNamespace(
                status=0, x=x, fun=value, message="test"
            ),
        )
        with pytest.raises(RuntimeError, match=message):
            solve_milp(one)
    monkeypatch.setattr(
        module,
        "milp",
        lambda *a, **kw: SimpleNamespace(
            status=1,
            x=np.array([1]),
            fun=0,
            message="limit with incumbent",
            mip_dual_bound=0,
            mip_gap=0,
        ),
    )
    result = solve_milp(one)
    assert result["status"] == "limit" and result["best"]["feasible"]
