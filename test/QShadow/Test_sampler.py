import numpy as np
import pytest

from pyqpanda3.core import H, QCircuit, RX, X

from pyqpanda_alg.QShadow import (
    MeasurementPlan,
    PyQPandaRunner,
    ShadowDataset,
    build_pyqpanda_program,
    run_measurement_plan,
    simulate_pauli_measurements,
    uniform_probabilities,
)


def test_reference_sampler_gets_deterministic_x_and_y_eigenvalues():
    plus = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    plus_i = np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    x_plan = MeasurementPlan(("X",) * 50, uniform_probabilities(1))
    y_plan = MeasurementPlan(("Y",) * 50, uniform_probabilities(1))
    assert set(simulate_pauli_measurements(plus, x_plan, seed=1).outcomes[:, 0]) == {1}
    assert set(simulate_pauli_measurements(plus_i, y_plan, seed=2).outcomes[:, 0]) == {1}


def test_fake_runner_q0_first_and_msb_first_conversion():
    plan = MeasurementPlan(("ZZ",), uniform_probabilities(2))
    q0_first = run_measurement_plan(plan, lambda basis, shots: {"01": 1})
    msb_first = run_measurement_plan(
        plan, lambda basis, shots: {"01": 1}, bit_order="msb_first"
    )
    assert q0_first.bitstrings == ("01",)
    assert msb_first.bitstrings == ("10",)


def test_runner_rejects_count_mismatch_and_non_integer_counts():
    plan = MeasurementPlan(("Z", "Z"), uniform_probabilities(1))
    with pytest.raises(ValueError, match="expected 2"):
        run_measurement_plan(plan, lambda basis, shots: {"0": 1})
    with pytest.raises(TypeError, match="integers"):
        run_measurement_plan(plan, lambda basis, shots: {"0": 1.0, "1": 1.0})


def test_max_jobs_guard_runs_before_any_remote_callback():
    plan = MeasurementPlan(("X", "Y", "Z"), uniform_probabilities(1))
    called = False

    def runner(basis, shots):
        nonlocal called
        called = True
        return {"0": shots}

    with pytest.raises(RuntimeError, match="exceeding max_jobs"):
        run_measurement_plan(plan, runner, max_jobs=2)
    assert not called


def test_grouped_remote_rounds_preserve_each_round_probabilities():
    first_probabilities = np.asarray([[0.2, 0.3, 0.5]])
    second_probabilities = np.asarray([[0.7, 0.2, 0.1]])
    first = run_measurement_plan(
        MeasurementPlan(("Z", "X", "Z"), first_probabilities),
        lambda basis, shots: {"0": shots},
    )
    second = run_measurement_plan(
        MeasurementPlan(("X", "Z", "Z"), second_probabilities),
        lambda basis, shots: {"0": shots},
    )
    combined = ShadowDataset.concatenate(first, second)

    selected_propensities = combined.probabilities[
        np.arange(combined.shots), 0, combined.bases[:, 0]
    ]
    assert combined.basis_strings == ("Z", "Z", "X", "X", "Z", "Z")
    assert np.allclose(selected_propensities, [0.5, 0.5, 0.2, 0.7, 0.1, 0.1])


def test_cpuqvm_runner_converts_pyqpanda_msb_order_to_q0_first():
    plan = MeasurementPlan(("ZZ",) * 20, uniform_probabilities(2))
    runner = PyQPandaRunner(lambda qubits: QCircuit() << X(qubits[0]), 2)
    dataset = run_measurement_plan(plan, runner, max_jobs=1)
    # q0=1 and q1=0.  Native PyQPanda counts are '01'; runner returns q0-first '10'.
    assert set(dataset.bitstrings) == {"10"}


def test_cpuqvm_runner_x_and_y_basis_signs():
    x_plan = MeasurementPlan(("X",) * 30, uniform_probabilities(1))
    y_plan = MeasurementPlan(("Y",) * 30, uniform_probabilities(1))
    x_runner = PyQPandaRunner(lambda qubits: QCircuit() << H(qubits[0]), 1)
    y_runner = PyQPandaRunner(
        lambda qubits: QCircuit() << RX(qubits[0], -np.pi / 2), 1
    )
    assert set(run_measurement_plan(x_plan, x_runner).outcomes[:, 0]) == {1}
    assert set(run_measurement_plan(y_plan, y_runner).outcomes[:, 0]) == {1}


def test_custom_executor_receives_program_and_shots_without_credentials():
    calls = []

    def executor(program, shots):
        calls.append((type(program).__name__, shots))
        return {"0": shots}

    plan = MeasurementPlan(("Z",) * 7, uniform_probabilities(1))
    runner = PyQPandaRunner(
        lambda qubits: None,
        1,
        executor=executor,
        backend_bit_order="msb_first",
    )
    dataset = run_measurement_plan(plan, runner, max_jobs=1)
    assert calls == [("QProg", 7)]
    assert dataset.bitstrings == ("0",) * 7


def test_pyqpanda_builder_validates_basis_width():
    with pytest.raises(ValueError, match="X/Y/Z"):
        build_pyqpanda_program(lambda qubits: None, 2, "X")
