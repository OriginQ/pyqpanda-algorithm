"""Raw semantic equivalence, replayable deletion evidence and arc limitations."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
from pyqpanda_alg.LabScheduling import (
    Problem,
    compile_qubo,
    load_problem,
    reduce_model,
    solve_exact,
    solve_reduced,
)
from pyqpanda_alg.LabScheduling.__main__ import main
from pyqpanda_alg.LabScheduling.report import run_experiment

EXAMPLES = (
    Path(__file__).resolve().parents[2] / "pyqpanda-algorithm/example/LabScheduling"
)
sys.path.insert(0, str(EXAMPLES))
try:
    support = importlib.import_module("support_pruning")
finally:
    sys.path.pop(0)


def random_problem(seed):
    rng = Random(seed)
    tasks = []
    for ti in range(rng.randrange(6)):
        tasks.append(
            {
                "id": f"t{ti}",
                "group": f"g{rng.randrange(2)}",
                "duration": rng.randint(1, 2),
                "change_cost": rng.randrange(5),
                "original": {"resource": "r0", "start": ti},
                "options": [
                    {"resource": r, "start": s, "cost": rng.randrange(5)}
                    for r, s in rng.sample(
                        [(r, s) for r in ("r0", "r1") for s in range(6)],
                        rng.randint(0 if seed % 9 == 0 else 1, 4),
                    )
                ],
            }
        )
    return Problem.from_dict(
        {
            "name": f"arc-{seed}",
            "horizon": 7,
            "resources": [
                {"id": "r0", "cleanup": seed % 2, "downtime": [[0, 1]]},
                {"id": "r1"},
            ],
            "tasks": tasks,
            "precedence": [
                [a["id"], b["id"]]
                for a in tasks
                for b in tasks
                if a != b and rng.random() < 0.08
            ],
        }
    )


def slot_problem(tasks, starts, duration=1):
    return Problem.from_dict(
        {
            "name": "shared slots",
            "horizon": 10,
            "resources": [{"id": "r"}],
            "tasks": [
                {
                    "id": f"t{i}",
                    "group": f"g{i}",
                    "duration": duration,
                    "options": [
                        {"resource": "r", "start": s, "cost": 0} for s in starts
                    ],
                }
                for i in range(tasks)
            ],
        }
    )


def test_two_hundred_raw_feasible_sets_costs_and_deletion_witnesses():
    for seed in range(200):
        support.audit_problem(random_problem(seed))


def test_existing_v2_corpus_preserves_every_raw_feasible_schedule():
    paths = sorted((EXAMPLES / "data/corpus-v2").glob("n*.json"))
    assert len(paths) == 36
    for path in paths:
        support.audit_problem(load_problem(path))


def test_support_cascade_crosses_resource_boundary_without_fixing_decisions():
    model = compile_qubo(load_problem(EXAMPLES / "data/support-chain.json"))
    old = reduce_model(model)
    new = reduce_model(model, pruning="arc")
    assert old.audit()["component_qubits"] == [18]
    assert new.audit()["component_qubits"] == [12]
    assert old.fixed == new.fixed == ()
    assert len(new.removals) == 6
    assert all(row["rule"] == "no_support" for row in new.removals)
    # Some witnesses depend on earlier deletions, requiring backward re-queuing.
    assert any(
        len(row["blockers"])
        < len(
            model.domains[
                next(
                    i
                    for i, t in enumerate(model.problem.tasks)
                    if t.id == row["against_task"]
                )
            ]
        )
        for row in new.removals
    )
    support.verify_removals(model, new)
    audit = support.audit_problem(model.problem)
    assert audit["feasible_count"] == 15
    assert audit["exact_objective"] == 3
    assert solve_reduced(model)["status"] == "resource_limit"
    run = solve_reduced(model, pruning="arc", maxiter=10, restarts=1)
    assert run["best"]["feasible"]
    assert run["best"]["objective"] >= 3
    for component in run["components"]:
        key = "".join(map(str, component["best"]["bits"][::-1]))
        assert component["counts"][key] > 0


def test_detects_infeasibility_without_any_initial_singleton():
    model = compile_qubo(slot_problem(2, [0, 1], duration=3))
    assert reduce_model(model).infeasible_task is None
    reduced = reduce_model(model, pruning="arc")
    assert reduced.infeasible_task == "t0"
    support.verify_removals(model, reduced)
    result = solve_reduced(model, pruning="arc")
    assert result["status"] == "infeasible_propagation"
    assert result["circuit_evaluations"] == 0
    assert result["components"] == [] and result["best"] is None
    assert not support.audit_problem(model.problem)["feasible_count"]


def test_pairwise_support_does_not_certify_global_feasibility():
    problem = slot_problem(3, [0, 1])
    model = compile_qubo(problem)
    reduction = reduce_model(model, pruning="arc")
    assert reduction.infeasible_task is None
    assert not reduction.removals and reduction.audit()["component_qubits"] == [6]
    assert solve_exact(model)["status"] == "infeasible"
    run = solve_reduced(model, pruning="arc", maxiter=6, restarts=1)
    assert run["status"] == "no_feasible_sample"
    assert not run["best"]["feasible"]
    support.audit_problem(problem)


def test_empty_independent_and_forced_cases():
    empty = compile_qubo(slot_problem(0, []))
    assert solve_reduced(empty, pruning="arc")["best"]["objective"] == 0
    no_options = compile_qubo(slot_problem(1, []))
    assert (
        solve_reduced(no_options, pruning="arc")["status"] == "infeasible_propagation"
    )
    p = slot_problem(2, [0, 1])
    independent = replace(
        p,
        tasks=(
            p.tasks[0],
            replace(
                p.tasks[1],
                options=tuple(
                    replace(o, start=o.start + 5) for o in p.tasks[1].options
                ),
            ),
        ),
    )
    assert support.audit_problem(independent)["modes"]["arc"]["component_qubits"] == [
        2,
        2,
    ]
    forced = replace(
        p, tasks=(replace(p.tasks[0], options=(p.tasks[0].options[0],)), p.tasks[1])
    )
    assert len(support.audit_problem(forced)["modes"]["arc"]["fixed_variables"]) == 2
    assert (
        solve_reduced(compile_qubo(forced), pruning="arc")["circuit_evaluations"] == 0
    )


def test_rejects_invalid_modes_and_ignored_cli_option(capsys):
    path = EXAMPLES / "data/simple.json"
    model = compile_qubo(load_problem(path))
    with pytest.raises(ValueError, match="pruning"):
        reduce_model(model, pruning="unknown")
    with pytest.raises(ValueError, match="pruning"):
        run_experiment(path, reduce=True, pruning="unknown")
    with pytest.raises(ValueError, match="requires"):
        run_experiment(path, pruning="arc")
    with pytest.raises(SystemExit) as error:
        main([str(path), "--pruning", "arc"])
    assert error.value.code == 2
    assert "requires" in capsys.readouterr().err


def test_arc_cli_report_replay_and_default_backward_compatibility(tmp_path, capsys):
    path = EXAMPLES / "data/simple.json"
    default = run_experiment(path, reduce=True, maxiter=6, restarts=1)
    explicit = run_experiment(
        path, reduce=True, pruning="singleton", maxiter=6, restarts=1
    )
    assert support.without_timing(default) == support.without_timing(explicit)
    out = tmp_path / "report.json"
    assert (
        main(
            [
                str(path),
                "--reduce",
                "--pruning",
                "arc",
                "--maxiter",
                "6",
                "--restarts",
                "1",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == report
    replay = run_experiment(path, reduce=True, pruning="arc", maxiter=6, restarts=1)
    assert support.without_timing(report) == support.without_timing(
        json.loads(json.dumps(replay))
    )
    assert report["quantum"]["business_metrics"]["feasible"]


def test_deletion_verifier_rejects_partial_or_fabricated_support_witness():
    model = compile_qubo(load_problem(EXAMPLES / "data/support-chain.json"))
    reduction = reduce_model(model, pruning="arc")
    row = reduction.removals[0]
    for blockers, match in (
        (row["blockers"][:-1], "entire live domain"),
        ([{**b, "reasons": ["invented"]} for b in row["blockers"]], "forbidden pair"),
    ):
        changed = replace(
            reduction, removals=({**row, "blockers": blockers}, *reduction.removals[1:])
        )
        with pytest.raises(ValueError, match=match):
            support.verify_removals(model, changed)


def test_evidence_command_creates_directories_and_replays_real_cpu_runs(
    tmp_path, monkeypatch, capsys
):
    first, replay = tmp_path / "first", tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["support_pruning", "--output", str(first)])
    support.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "support_pruning",
            "--output",
            str(replay),
            "--verify-against",
            str(first / "results.json"),
        ],
    )
    support.main()
    capsys.readouterr()
    assert json.loads((replay / "replay.json").read_text())["identical_except_timing"]
    result = json.loads((first / "results.json").read_text())
    assert len(result["instances"]) == 37
    assert result["boundary_original"]["before_outage"]["objective"] == 0
    assert not result["boundary_original"]["after_outage"]["feasible"]
    assert result["boundary_default"]["status"] == "resource_limit"
    assert [r["absolute_gap"] for r in result["boundary_arc"]] == [0, 0, 0]
