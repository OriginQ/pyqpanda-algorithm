"""Bounded light-cone tests; CPUQVM is a TEST-ONLY independent oracle.

Production estimation is exercised only via the supplied counts callback.
No exact probabilities are supplied to the estimator.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import product
from pathlib import Path
import math
import runpy
import sys
from types import ModuleType

import numpy as np
import pytest
from pyqpanda3.core import CPUQVM, H, QProg, RX, RZ, X, measure
from pyqpanda3.hamiltonian import PauliOperator

from pyqpanda_alg import QAOA as qaoa_package
from pyqpanda_alg.QAOA.qaoa import QAOA


def make_executor(n_qubits, terms, depth):
    return qaoa_package.QAOALightCone(n_qubits, terms, depth)


def plan_for(executor, target):
    return next(p for p in executor.plans if p.target.qubits == target)


def exact_probabilities(program):
    """TEST ONLY: omit final measurements, then inspect the CPU reference."""
    unmeasured = QProg(program.qubits_num())
    for gate in program.gate_operations():
        unmeasured << gate
    reference = CPUQVM()
    reference.run(unmeasured, shots=1)
    return reference.result().get_prob_dict()


def target_marginal(probabilities, qubits):
    marginal: dict[tuple[int, ...], float] = {
        bits: 0.0 for bits in product((0, 1), repeat=len(qubits))}
    for key, probability in probabilities.items():
        marginal[tuple(int(key[-1 - q]) for q in qubits)] += probability
    return marginal


def parity_mean(probabilities, qubits):
    return sum(probability * (-1) ** sum(int(key[-1 - q]) for q in qubits)
               for key, probability in probabilities.items())


def full_legacy_program(n_qubits, terms, gammas, betas):
    """Reference the existing QAOA construction, not the light-cone builder."""
    operator = PauliOperator({"I": 0.0})
    for support, coefficient in terms.items():
        label = " ".join("Z%d" % q for q in support) or "I"
        operator += PauliOperator({label: coefficient})
    legacy = QAOA(operator)
    program = QProg(n_qubits)
    program << legacy._qaoa_circuit(list(range(n_qubits)), gammas, betas)
    return program


def test_public_export_preserves_legacy_api():
    assert hasattr(qaoa_package, "QAOALightCone")
    assert qaoa_package.qaoa.QAOA is QAOA


def test_synchronous_layers_do_not_walk_whole_cost_layer():
    terms = {(0,): 0.5, (0, 1): 1, (1, 2): 2, (2, 3): 3, (3, 4): 4,
             (1,): -0.2, (4,): 0.3}
    one = plan_for(make_executor(5, terms, 1), (0,))
    assert one.global_qubits == (0, 1)
    assert one.layers[0].mixer_qubits == (0,)
    assert {t.qubits for t in one.layers[0].cost_terms} == {(0,), (0, 1)}
    two = plan_for(make_executor(5, terms, 2), (0,))
    assert two.global_qubits == (0, 1, 2)
    assert two.layers[0].mixer_qubits == (0, 1)
    assert two.layers[1] == one.layers[0]
    assert {t.qubits for t in two.layers[0].cost_terms} == {
        (0,), (0, 1), (1,), (1, 2)}
    # This must not depend on the insertion order of commuting cost terms.
    reversed_terms = dict(reversed(tuple(terms.items())))
    assert make_executor(5, terms, 2).plans == make_executor(5, reversed_terms, 2).plans


def test_sparse_disconnected_mapping_and_actual_program_width():
    executor = make_executor(100, {(90, 7): -2, (7,): 0.4, (1, 3): 1}, 1)
    job = next(j for j in executor.build_programs([0.2], [0.3])
               if j.plan.target.qubits == (7,))
    assert job.plan.global_qubits == (7, 90)
    assert dict(job.plan.global_to_local) == {7: 0, 90: 1}
    assert job.local_target == (0,)
    assert job.parity_cbits == (0,)
    assert job.measurement_cbits == (0, 1)
    assert job.program.qubits_num() == 2
    assert job.program.cbits() == [0, 1]
    assert "MEASURE q[0],c[0]" in job.program.originir()
    assert "MEASURE q[1],c[1]" in job.program.originir()
    pair = next(j for j in executor.build_programs([0.2], [0.3])
                if j.plan.target.qubits == (7, 90))
    assert pair.local_target == pair.parity_cbits == (0, 1)


def test_input_copy_and_immutable_plan_and_fresh_programs():
    terms = {(0, 1): 1, (1,): 0.4, (): 2}
    executor = make_executor(3, terms, 2)
    saved = executor.plans
    terms[(0, 1)] = 9
    terms[(1, 2)] = -3
    terms[()] = 99
    assert executor.constant == 2
    assert plan_for(executor, (0, 1)).target.coefficient == 1
    assert executor.plans is saved
    with pytest.raises(FrozenInstanceError):
        executor.depth = 4
    with pytest.raises(FrozenInstanceError):
        saved[0].target.coefficient = 2
    with pytest.raises(FrozenInstanceError):
        saved[0].layers[0].mixer_qubits = ()
    with pytest.raises(TypeError):
        saved[0].global_to_local[0] = 5
    gammas, betas = [0.1, 0.2], [0.3, 0.4]
    jobs = executor.build_programs(gammas, betas)
    original_ir = jobs[0].program.originir()
    gammas[0] = 0.9
    assert jobs[0].program.originir() == original_ir
    newer = executor.build_programs(gammas, betas)
    assert newer[0].program is not jobs[0].program
    assert newer[0].plan is jobs[0].plan
    assert newer[0].program.originir() != original_ir
    jobs[0].program << X(0)
    assert executor.build_programs([0.1, 0.2], betas)[0].program.originir() == original_ir


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("terms,constant", [({}, 0), ({(): -2.5}, -2.5),
                                           ({(): 3, (0,): 0}, 3)])
def test_constant_and_zero_terms_need_no_calls(depth, terms, constant):
    executor = make_executor(1, terms, depth)
    def forbidden(program, shots):
        pytest.fail("a constant must not submit a circuit")
    result = executor.estimate([0.2] * depth, [0.3] * depth, 7, forbidden)
    assert executor.plans == ()
    assert executor.build_programs([0.2] * depth, [0.3] * depth) == ()
    assert result.mean == constant
    assert result.term_means == ()
    assert result.total_shots == 0
    assert result.shots_per_program == 7
    assert make_executor(0, {(): constant}, depth).constant == constant


def test_zero_depth_still_uses_fresh_counts_for_nonconstant_terms():
    executor = make_executor(5, {(1,): 0.5, (1, 4): -2, (): 3}, 0)
    calls = []
    def runner(program, shots):
        calls.append(program)
        assert all(gate in ("QINIT", "CREG", "H", "MEASURE")
                   for gate in (line.split()[0] for line in program.originir().splitlines()))
        return {"0" * program.qubits_num(): shots}
    result = executor.estimate([], [], 9, runner)
    assert result.mean == 1.5
    assert result.term_means == (1, 1)
    assert result.total_shots == 18
    assert len(calls) == 2
    assert [p.global_qubits for p in executor.plans] == [(1,), (1, 4)]
    assert all(p.layers == () for p in executor.plans)


@pytest.mark.parametrize("value", [True, np.bool_(False), -1, 1.5, "2", None,
                                    float("nan"), float("inf")])
@pytest.mark.parametrize("argument", ["n_qubits", "depth"])
def test_invalid_dimensions(argument, value):
    args = dict(n_qubits=2, terms={(0,): 1}, depth=1)
    args[argument] = value
    with pytest.raises((TypeError, ValueError)):
        qaoa_package.QAOALightCone(**args)


@pytest.mark.parametrize("terms", [None, [], [((), 1)], {0: 1}, {"Z0": 1},
    {(0, 1, 2): 1}, {(1, 1): 1}, {(-1,): 1}, {(3,): 1}, {(True,): 1},
    {(np.bool_(False),): 1}, {(1.0,): 1}, {(0, 1): 1, (1, 0): 2}])
def test_invalid_terms(terms):
    with pytest.raises((TypeError, ValueError)):
        make_executor(3, terms, 1)


@pytest.mark.parametrize("coefficient", [True, np.bool_(True), None, "1",
    complex(1, 0), complex(1, 2), float("nan"), float("inf"), -float("inf")])
def test_invalid_coefficients_including_constants(coefficient):
    for support in [(), (0,), (0, 1)]:
        with pytest.raises((TypeError, ValueError)):
            make_executor(2, {support: coefficient}, 1)


def test_numpy_scalars_fractions_and_canonical_order():
    executor = make_executor(np.int64(4), {(np.int64(3), 1): np.float64(0.5),
                             (1,): Fraction(1, 4)}, np.int64(1))
    assert [t.qubits for t in executor.terms] == [(1,), (1, 3)]
    assert [t.coefficient for t in executor.terms] == [0.25, 0.5]
    assert len(executor.build_programs(np.array([0.1]), np.array([0.2]))) == 2


@pytest.mark.parametrize("bad", [[], [0.1, 0.2], [True], [np.bool_(True)],
    [float("nan")], [float("inf")], [1j], ["0.2"], [[0.2]], 0.2, "x", None])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_angles_before_runner_call(bad, position):
    executor = make_executor(1, {(0,): 1}, 1)
    params = [[0.1], [0.2]]
    params[position] = bad
    calls = []
    def runner(program, shots):
        calls.append(program)
        return {"0": shots}
    with pytest.raises((TypeError, ValueError)):
        executor.estimate(*params, 5, runner)
    assert not calls


def test_overflowed_gate_angles_rejected_before_any_runner_call():
    executor = make_executor(2, {(0,): 1, (1,): 1e308}, 1)
    calls = []
    with pytest.raises(ValueError, match="finite"):
        executor.estimate([1e308], [0.2], 5,
                          lambda p, s: calls.append(p) or {"0": s})
    assert not calls


@pytest.mark.parametrize("shots", [True, np.bool_(True), 0, -1, 1.0,
                                     float("nan"), float("inf"), "5", None])
def test_invalid_shots(shots):
    with pytest.raises((TypeError, ValueError)):
        make_executor(1, {(0,): 1}, 0).estimate([], [], shots, lambda p, s: {"0": s})


def test_runner_is_mandatory_even_for_constants():
    for terms in [{}, {(0,): 1}]:
        executor = make_executor(1, terms, 0)
        with pytest.raises(TypeError):
            executor.estimate([], [], 5)
        with pytest.raises(TypeError):
            executor.estimate([], [], 5, None)


@pytest.mark.parametrize("counts", [None, [], {0: 5}, {"": 5}, {"00": 5},
    {"2": 5}, {" 0": 5}, {"0b0": 5}, {"0": True}, {"0": np.bool_(True)},
    {"0": 5.0}, {"0": 5 + 0j}, {"0": float("nan")}, {"0": float("inf")},
    {"0": -1, "1": 6}, {}, {"0": 4}, {"0": 6}, {"0": 0.5, "1": 0.5}])
def test_strict_counts_validation(counts):
    with pytest.raises((TypeError, ValueError)):
        make_executor(1, {(0,): 1}, 0).estimate([], [], 5, lambda p, s: counts)


def test_counts_endianness_parity_total_shots_and_fresh_sampling():
    executor = make_executor(3, {(0,): 1, (0, 1): 2, (1,): 3, (): 4}, 1)
    calls = []
    def runner(program, shots):
        calls.append((program, shots))
        # c1=0, c0=1 in every shot: Z0=-1, Z0Z1=-1, Z1=+1.
        return {"01": np.int64(shots), "00": 0}
    first = executor.estimate([0.2], [0.3], np.int64(7), runner)
    assert first.mean == 4
    assert first.term_means == (-1, -1, 1)
    assert first.shots_per_program == 7
    assert first.total_shots == 21
    assert len(calls) == 3
    plans = executor.plans
    second = executor.estimate([0.4], [-0.1], 7, lambda p, s: {"10": s})
    assert second.mean == 0
    assert second.term_means == (1, -1, -1)
    assert executor.plans is plans
    third = executor.estimate([0.2], [0.3], 7, lambda p, s: {"00": s})
    assert third.mean == 10  # The original parameter values do not hit a result cache.
    assert first.mean == 4
    with pytest.raises(FrozenInstanceError):
        first.mean = 99


def test_runner_errors_are_not_replaced_with_backend_fallback():
    def failing(program, shots):
        raise RuntimeError("device offline")
    with pytest.raises(RuntimeError, match="device offline"):
        make_executor(1, {(0,): 1}, 1).estimate([0.1], [0.2], 5, failing)


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("terms", [
    {(): 0.2, (0,): -0.6, (1,): 0.4, (2,): 0.1,
     (0, 1): 0.7, (1, 2): -0.3, (2, 3): 0.2, (3, 4): -0.9},
    {(0, 2): 0.6, (2,): 0.8, (3, 4): -0.7, (4,): -0.2},
    {(0, 1): 0.7, (0, 2): -0.4, (0, 3): 0.9, (0, 4): 0.2,
     (1, 2): 0.5, (0,): -0.3},
])
def test_local_target_marginals_match_existing_full_qaoa(depth, terms):
    executor = make_executor(5, terms, depth)
    gammas, betas = [0.27, -0.31, 0.19][:depth], [-0.18, 0.43, 0.22][:depth]
    full_probabilities = exact_probabilities(full_legacy_program(5, terms, gammas, betas))
    for job in executor.build_programs(gammas, betas):
        local = exact_probabilities(job.program)
        assert target_marginal(local, job.local_target) == pytest.approx(
            target_marginal(full_probabilities, job.plan.target.qubits), abs=2e-12)


def test_rx_negative_sign_against_one_qubit_independent_formula():
    gamma, beta, weight = 0.37, 0.23, -0.8
    job = make_executor(1, {(0,): weight}, 1).build_programs([gamma], [beta])[0]
    expected = -math.sin(2 * gamma * weight) * math.sin(2 * beta)
    assert abs(expected) > 0.1
    assert parity_mean(exact_probabilities(job.program), (0,)) == pytest.approx(expected)
    assert "RX q[0],(-0.46000000)" in job.program.originir()
    # Distinguish the opposite-sign convention explicitly.
    wrong = QProg(1)
    wrong << H(0) << RZ(0, 2 * gamma * weight) << RX(0, 2 * beta)
    assert parity_mean(exact_probabilities(wrong), (0,)) == pytest.approx(-expected)


def test_cpu_reference_counts_adapter_and_non_palindromic_bit_order():
    """CPUQVM sampling is an explicit TEST runner, never a production default."""
    reference = CPUQVM()
    measured = QProg(2)
    measured << X(0) << measure(0, 0) << measure(1, 1)
    reference.run(measured, shots=8)
    assert reference.result().get_counts() == {"01": 8}
    executor = make_executor(1, {(0,): 1}, 1)
    calls = []
    def counts_only(program, shots):
        calls.append(shots)
        reference.run(program, shots=shots)
        return reference.result().get_counts()
    # These angles prepare |1>; the counts test is deterministic, not a flaky tolerance.
    result = executor.estimate([math.pi / 4], [math.pi / 4], 32, counts_only)
    assert result.mean == -1
    assert result.total_shots == 32
    assert calls == [32]


def test_mixed_counts_use_frequencies_not_distinct_outcome_average():
    executor = make_executor(2, {(): 1.25, (0, 1): -2}, 0)
    result = executor.estimate([], [], 10, lambda p, s: {
        "00": 5, "11": 2, "01": 2, "10": 1})
    assert result.term_means == pytest.approx((0.4,))
    assert result.mean == pytest.approx(0.45)
    assert result.total_shots == 10


def test_zero_edge_does_not_expand_and_isolated_fields_stay_local():
    executor = make_executor(4, {(0,): 0.3, (0, 1): 0, (1, 2): 1, (3,): 0.7}, 3)
    assert plan_for(executor, (0,)).global_qubits == (0,)
    assert plan_for(executor, (3,)).global_qubits == (3,)
    assert len(executor.plans) == 3
    with pytest.raises(ValueError):
        make_executor(0, {(0,): 0}, 0)  # Even a zero term is validated.


def test_target_marginal_not_entire_cone_marginal_is_promised():
    terms = {(0,): 0.4, (0, 1): 0.7, (1,): 0.8}
    executor = make_executor(2, terms, 1)
    job = next(j for j in executor.build_programs([0.3], [0.4])
               if j.plan.target.qubits == (0,))
    full = exact_probabilities(full_legacy_program(2, terms, [0.3], [0.4]))
    local = exact_probabilities(job.program)
    assert target_marginal(local, (0,)) == pytest.approx(target_marginal(full, (0,)))
    assert abs(parity_mean(local, (1,)) - parity_mean(full, (1,))) > 0.1


def test_example_preview_does_not_run_backend(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("preview must not call estimate")
    monkeypatch.setattr(qaoa_package.QAOALightCone, "estimate", forbidden)
    example = Path(__file__).resolve().parents[2] / "examples" / "qaoa_lightcone.py"
    monkeypatch.setattr(sys, "argv", [str(example)])
    runpy.run_path(str(example), run_name="__main__")
    text = capsys.readouterr().out
    assert "global width: 12 local programs: 12" in text
    assert "Preview only: no backend called" in text


def test_example_explicit_runner_submits_fresh_small_programs(monkeypatch, capsys):
    # Test data only: this adapter checks CLI dispatch, not physical execution.
    adapter = ModuleType("lightcone_test_adapter")
    programs = []
    def run_counts(program, shots):
        programs.append(program)
        assert shots == 5
        assert program.qubits_num() <= 4
        return {"0" * program.qubits_num(): shots}
    adapter.run_counts = run_counts
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    example = Path(__file__).resolve().parents[2] / "examples" / "qaoa_lightcone.py"
    monkeypatch.setattr(sys, "argv", [str(example), "--runner",
                                     adapter.__name__ + ":run_counts", "--shots", "5"])
    runpy.run_path(str(example), run_name="__main__")
    assert len(programs) == 24
    assert len({id(program) for program in programs}) == 24
    text = capsys.readouterr().out
    assert text.count("total shots: 60") == 2


def test_objective_finite_cancellation_survives_intermediate_overflow():
    executor = make_executor(2, {(): 1e308, (0,): 1e308, (1,): -1e308}, 0)
    result = executor.estimate([], [], 1, lambda program, shots: {"0": shots})
    assert result.mean == 1e308


def test_objective_cancellation_retains_tiny_residual():
    executor = make_executor(4, {(): 1e308, (0,): 1e308,
                               (1,): -1e308, (2,): -1e308, (3,): 1e-308}, 0)
    result = executor.estimate([], [], 1, lambda program, shots: {"0": shots})
    assert result.mean == 1e-308


def test_objective_true_overflow_is_explicit_after_sampling():
    calls = []
    def counts(program, shots):
        calls.append(shots)
        return {"0": shots}
    executor = make_executor(1, {(): 1e308, (0,): 1e308}, 0)
    with pytest.raises(ValueError, match="finite float range"):
        executor.estimate([], [], 1, counts)
    assert calls == [1]
