"""Independent business and corpus invariants for the second evaluation."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling import (
    compile_qubo,
    evaluate,
    load_problem,
    reduce_model,
    solve_milp,
    solve_reduced,
    validate_assignment,
)

EXAMPLES = (
    Path(__file__).resolve().parents[2] / "pyqpanda-algorithm/example/LabScheduling"
)
spec = importlib.util.spec_from_file_location("ablation", EXAMPLES / "ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def test_frozen_heterogeneous_inputs_preserve_controls_and_all_feasible_schedules():
    corpus = EXAMPLES / "data/corpus-v2"
    manifest = json.loads((corpus / "manifest.json").read_text())
    assert len(manifest["instances"]) == 36
    assert (
        hashlib.sha256((EXAMPLES / "make_corpus_v2.py").read_bytes()).hexdigest()
        == manifest["generator_sha256"]
    )
    durations, cleanups, domains = set(), set(), set()
    for record in manifest["instances"]:
        path = corpus / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        problem = load_problem(path)
        control = replace(
            problem, resources=tuple(replace(r, downtime=()) for r in problem.resources)
        )
        witness = validate_assignment(control, [0] * len(problem.tasks))
        assert witness["feasible"] and witness["objective"] == 0
        stages = ablation.audit_stages(problem)
        oracle = solve_milp(problem)
        assert (oracle["status"] == "optimal") == bool(stages["feasible_count"])
        if oracle["best"]:
            assert oracle["best"]["objective"] == stages["exact_objective"]
        durations.update(t.duration for t in problem.tasks)
        cleanups.update(r.cleanup for r in problem.resources)
        domains.update(len(t.options) for t in problem.tasks)
    assert durations == {1, 2, 3}
    assert cleanups == {0, 1, 2}
    assert domains == {2, 3, 4}


def test_tradeoff_retains_all_three_decisions_and_nine_feasible_alternatives():
    model = compile_qubo(load_problem(EXAMPLES / "data/tradeoff.json"))
    reduction = reduce_model(model)
    assert reduction.fixed == ()
    assert [len(m.variables) for m in reduction.components] == [8]
    choices = [evaluate(model, bits) for bits in ablation.feasible_vectors(model)]
    assert len(choices) == 9
    assert min(row["objective"] for row in choices) == 12
    optimum = next(row for row in choices if row["objective"] == 12)
    assert sum(row["changed"] for row in optimum["schedule"]) == 3
    backup = min(
        (
            row
            for row in choices
            if any(r["resource"] == "backup" for r in row["schedule"])
        ),
        key=lambda r: r["objective"],
    )
    assert backup["objective"] == 14
    assert sum(row["changed"] for row in backup["schedule"]) == 1
    stages = ablation.audit_stages(model.problem)["stages"]
    assert stages["raw_input"]["candidate_count"] == 9
    assert (
        stages["components_auto_phase"]["cnot"]
        < stages["components_full_phase"]["cnot"]
    )


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_tradeoff_quantum_choices_are_actual_samples_and_raw_valid(seed):
    model = compile_qubo(load_problem(EXAMPLES / "data/tradeoff.json"))
    result = solve_reduced(model, seed=seed)
    component = result["components"][0]
    assert component["counts"]["".join(map(str, component["best"]["bits"][::-1]))] > 0
    assert result["circuit_evaluations"] > 0
    assert result["best"]["feasible"]
    assert result["best"]["objective"] >= 12
    choices = [
        task.options.index(
            next(
                variable.option
                for variable, bit in zip(
                    model.variables, result["best"]["bits"], strict=True
                )
                if bit and variable.task == ti
            )
        )
        for ti, task in enumerate(model.problem.tasks)
    ]
    raw = validate_assignment(model.problem, choices)
    assert raw["feasible"]
    assert raw["objective"] == result["best"]["objective"]
