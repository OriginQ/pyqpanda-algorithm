"""Audit each preprocessing stage against every raw feasible assignment."""

from dataclasses import replace
from itertools import product
from time import perf_counter
from typing import Any

from pyqpanda_alg.LabScheduling import (
    Model,
    Problem,
    compile_qubo,
    evaluate,
    reduce_model,
    validate_assignment,
)
from pyqpanda_alg.LabScheduling.metrics import circuit_resources


def feasible_vectors(model: Model) -> dict[tuple[int, ...], int]:
    """Enumerate feasible one-hot vectors and their original objective costs."""
    result = {}
    for choices in product(*model.domains):
        bits = tuple(int(i in choices) for i in range(len(model.variables)))
        row = evaluate(model, bits)
        if row["feasible"]:
            result[bits] = row["objective"]
    return result


def resources(models: list[Model], mode: str) -> dict[str, Any]:
    """Count native p=1 gates; distinguish serial depth from component maximum."""
    valid = all(all(m.domains) for m in models)
    counts = [circuit_resources(m, [0.37, 0.61])[mode] for m in models] if valid else []
    return {
        "component_qubits": [len(m.variables) for m in models],
        "total_qubits": sum(len(m.variables) for m in models),
        "max_component_qubits": max((len(m.variables) for m in models), default=0),
        "cnot": sum(c["cnot"] for c in counts) if valid else None,
        "cry": sum(c["cry"] for c in counts) if valid else None,
        "serial_depth_sum": sum(c["depth"] for c in counts) if valid else None,
        "max_component_depth": max((c["depth"] for c in counts), default=0)
        if valid
        else None,
        "circuit_status": "constructed" if valid and models else "no_circuit",
    }


def audit_stages(problem: Problem) -> dict[str, Any]:
    """Verify pruning and restoration bijections before reporting saved resources.

    Intended for the bounded example corpus, not unrestricted production inputs.
    Raw input is enumerated with the independent raw semantic checker. A raw
    invalid option has no circuit representation in this app, so its gate count
    is deliberately absent instead of comparing different constraint problems.
    """
    started = perf_counter()
    model = compile_qubo(problem)
    compiled = perf_counter()
    reduction = reduce_model(model)
    reduced = perf_counter()
    raw_feasible = {}
    for choices in product(*(range(len(t.options)) for t in problem.tasks)):
        row = validate_assignment(problem, choices)
        if row["feasible"]:
            raw_feasible[choices] = row["objective"]
    vectors = feasible_vectors(model)
    mapped = {}
    for bits, cost in vectors.items():
        choices = tuple(
            problem.tasks[ti].options.index(
                model.variables[next(i for i in d if bits[i])].option
            )
            for ti, d in enumerate(model.domains)
        )
        mapped[choices] = cost
    if mapped != raw_feasible:
        raise RuntimeError("calendar pruning changed raw feasible assignments")
    restored = {}
    if reduction.infeasible_task is None:
        local = [feasible_vectors(m) for m in reduction.components]
        for choices in product(*(list(values) for values in local)):
            row = reduction.restore(choices)
            expected = reduction.audit()["fixed_cost"] + sum(
                values[bits] for values, bits in zip(local, choices, strict=True)
            )
            if not row["feasible"] or row["objective"] != expected:
                raise RuntimeError("component restoration changed feasibility or cost")
            restored[tuple(row["bits"])] = row["objective"]
    if restored != vectors:
        raise RuntimeError("reduction changed the complete feasible set")
    proven = perf_counter()
    stage_rows = {
        "raw_input": {
            "candidate_count": sum(len(t.options) for t in problem.tasks),
            "cnot": None,
            "circuit_status": "not_encoded; unary-invalid candidates still present",
        },
        "calendar_full_phase": resources([model], "full"),
    }
    if reduction.infeasible_task is None:
        active = {i for mapping in reduction.mappings for i in mapping}
        tasks = tuple(
            replace(
                task,
                options=tuple(model.variables[i].option for i in domain if i in active),
            )
            for task, domain in zip(problem.tasks, model.domains, strict=True)
            if any(i in active for i in domain)
        )
        ids = {t.id for t in tasks}
        joint = compile_qubo(
            replace(
                problem,
                tasks=tasks,
                precedence=tuple(
                    (a, b) for a, b in problem.precedence if a in ids and b in ids
                ),
            )
        )
        stage_rows.update(
            {
                "propagation_full_phase": resources([joint] if tasks else [], "full"),
                "components_full_phase": resources(list(reduction.components), "full"),
                "components_auto_phase": resources(list(reduction.components), "auto"),
            }
        )
    return {
        "stages": stage_rows,
        "reduction": reduction.audit(),
        "feasible_count": len(vectors),
        "exact_objective": min(vectors.values(), default=None),
        "proof": (
            "all raw feasible choices = pruned choices = restored component choices; "
            "costs equal"
        ),
        "calendar_seconds": compiled - started,
        "reduction_seconds": reduced - compiled,
        "proof_seconds": proven - reduced,
        "resource_audit_seconds": perf_counter() - proven,
    }
