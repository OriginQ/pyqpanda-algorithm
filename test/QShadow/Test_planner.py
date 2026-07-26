import numpy as np
import pytest

from pyqpanda_alg.QShadow import (
    MeasurementPlan,
    PauliObservable,
    optimize_basis_probabilities,
    planning_loss,
    sample_measurement_plan,
    uniform_probabilities,
)


def test_uniform_probabilities_shape_and_values():
    probabilities = uniform_probabilities(4)
    assert probabilities.shape == (4, 3)
    assert np.allclose(probabilities, 1 / 3)


def test_single_axis_optimum_saturates_unused_axes_at_floor():
    observable = PauliObservable.from_terms({"Z": 1.0}, name="z")
    result = optimize_basis_probabilities(observable, probability_floor=0.05)
    assert result.converged
    assert np.allclose(result.probabilities[0], [0.05, 0.05, 0.90])
    assert result.final_loss < result.initial_loss


def test_unused_qubit_stays_uniform():
    observable = PauliObservable.from_terms({"ZI": 1.0}, name="local-z")
    result = optimize_basis_probabilities(observable, probability_floor=0.02)
    assert np.allclose(result.probabilities[1], [1 / 3, 1 / 3, 1 / 3])


def test_optimizer_reduces_multi_observable_proxy_and_respects_floor():
    observables = [
        PauliObservable.from_terms(
            {"ZZI": 1.4, "IZZ": -0.7, "XII": 0.2}, name="energy"
        ),
        PauliObservable.from_terms(
            {"ZIZ": 1.0, "YYY": 0.1}, name="correlator"
        ),
    ]
    uniform = uniform_probabilities(3)
    result = optimize_basis_probabilities(
        observables,
        observable_weights=[2.0, 0.5],
        probability_floor=0.03,
    )
    assert np.all(result.probabilities >= 0.03 - 1e-12)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.final_loss == pytest.approx(
        planning_loss(
            result.probabilities, observables, observable_weights=[2.0, 0.5]
        )
    )
    assert result.final_loss < planning_loss(
        uniform, observables, observable_weights=[2.0, 0.5]
    )
    assert all(
        later <= earlier + 1e-10
        for earlier, later in zip(result.loss_history, result.loss_history[1:])
    )


def test_zero_observable_returns_uniform_without_iteration():
    observable = PauliObservable.from_terms({"II": 0.0}, name="zero")
    result = optimize_basis_probabilities(observable)
    assert result.iterations == 0
    assert result.final_loss == 0.0
    assert np.allclose(result.probabilities, 1 / 3)


def test_sample_plan_is_reproducible_and_grouped_counts_sum_to_shots():
    probabilities = np.asarray([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    first = sample_measurement_plan(probabilities, 200, seed=123)
    second = sample_measurement_plan(probabilities, 200, seed=123)
    assert first.bases == second.bases
    assert first.shots == 200
    assert first.n_qubits == 2
    assert sum(first.grouped_shots.values()) == 200


def test_measurement_plan_rejects_inconsistent_input():
    with pytest.raises(ValueError, match="equal-length"):
        MeasurementPlan(("XZ", "X"), uniform_probabilities(2))
    with pytest.raises(ValueError, match="sum to one"):
        MeasurementPlan(("X",), np.asarray([[0.2, 0.2, 0.2]]))


def test_optimizer_rejects_invalid_floor_and_weights():
    observable = PauliObservable.from_terms({"X": 1}, name="x")
    with pytest.raises(ValueError, match="between 0 and 1/3"):
        optimize_basis_probabilities(observable, probability_floor=1 / 3)
    with pytest.raises(ValueError, match="match"):
        optimize_basis_probabilities(observable, observable_weights=[1, 2])
