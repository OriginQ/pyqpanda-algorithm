"""Budget conservation, sample provenance and fair-control regression checks."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from pyqpanda_alg.LabScheduling import compile_qubo, load_problem, reduce_model

EXAMPLES = (
    Path(__file__).resolve().parents[2] / "pyqpanda-algorithm/example/LabScheduling"
)
sys.path.insert(0, str(EXAMPLES))
try:
    budget = importlib.import_module("matched_budget")
    summary = importlib.import_module("summarize_matched")
finally:
    sys.path.pop(0)


without_time = summary.without_timing


def test_allocation_conserves_budget_and_rejects_unfunded_components():
    for total in (8, 16, 60, 128, 512):
        for count in (1, 2, 3):
            parts = budget.allocate(total, count)
            assert sum(parts) == total
            assert max(parts) - min(parts) <= 1
    assert budget.allocate(512, 0) == []
    assert budget.allocate(512, 3) == [171, 171, 170]
    for args in ((0, 1), (2, 3), (60, 11, 6), (True, 1), (8, -1)):
        with pytest.raises(ValueError):
            budget.allocate(*args)


def test_component_hit_probability_is_product_of_best_of_samples():
    assert budget.joint_hit([0.2, 0.4], 5) == pytest.approx((1 - 0.8**3) * (1 - 0.6**2))
    assert budget.joint_hit([], 8) == 1
    assert budget.joint_hit([0.0, 1.0], 8) == 0


def test_single_unchanged_component_has_identical_controls_and_replays():
    model = compile_qubo(load_problem(EXAMPLES / "data/tradeoff.json"))
    reduction = reduce_model(model)
    first = budget.compare(model, reduction, 7)
    second = budget.compare(model, reduction, 7)
    assert without_time(first) == without_time(second)
    paths = first["paths"]
    assert without_time(paths["direct"]) == without_time(
        paths["reduced"]["components"][0]
    )
    assert without_time(paths["uniform_direct"]) == without_time(
        paths["uniform_reduced"]["components"][0]
    )
    assert first["exact_objective"] == 12


def test_three_components_share_caps_and_all_selected_results_are_observed():
    model = compile_qubo(load_problem(EXAMPLES / "data/corpus-v2/n5-s0-r0.json"))
    reduction = reduce_model(model)
    assert len(reduction.components) == 3
    row = budget.compare(model, reduction, 7)
    for name in ("reduced", "uniform_reduced"):
        run = row["paths"][name]
        assert run["shots"] == 512
        assert run["circuit_evaluations"] <= 120
        for component, allocation in zip(
            run["components"], run["allocations"], strict=True
        ):
            assert sum(component["counts"].values()) == allocation["shots"]
            key = "".join(map(str, component["best"]["bits"][::-1]))
            assert component["counts"][key] > 0
        restored = reduction.restore([r["best"]["bits"] for r in run["components"]])
        assert restored == run["best"]
    assert (
        sum(
            a["maxiter_per_restart"] * 2 for a in row["paths"]["reduced"]["allocations"]
        )
        == 120
    )
    limited = budget.components(reduction, 7, quantum=True, per_restart=10)
    assert limited["status"] == "budget_insufficient"
    assert limited["circuit_evaluations"] == limited["shots"] == 0


def test_infeasible_domains_do_not_produce_fictitious_samples():
    corpus = EXAMPLES / "data/corpus-v2"
    model = next(
        m
        for m in (compile_qubo(load_problem(p)) for p in sorted(corpus.glob("n*.json")))
        if any(not d for d in m.domains)
    )
    row = budget.compare(model, reduce_model(model), 0)
    assert row["exact_objective"] is None
    assert row["conditional_hit_curve"] == []
    assert row["paths"]["uniform_direct"]["best"] is None
    assert row["paths"]["uniform_direct"]["shots"] == 0
    assert row["paths"]["reduced"]["status"] == "infeasible_propagation"


def test_replay_ignores_only_timing_and_rejects_changed_samples(tmp_path):
    original, replay = tmp_path / "original", tmp_path / "replay"
    for directory, elapsed in ((original, 1), (replay, 2)):
        directory.mkdir()
        (directory / "provenance.json").write_text('{"seed": 7}')
        (directory / "execution.json").write_text('{"paired_records": 1}')
        (directory / "runs.jsonl").write_text(
            json.dumps({"run": {"counts": {"01": 512}, "runtime_seconds": elapsed}})
            + "\n"
        )
    summary.verify_replay(original, replay)
    assert json.loads((original / "replay.json").read_text())["identical_except_timing"]
    (replay / "runs.jsonl").write_text('{"run":{"counts":{"10":512}}}\n')
    with pytest.raises(ValueError, match="beyond timing"):
        summary.verify_replay(original, replay)
