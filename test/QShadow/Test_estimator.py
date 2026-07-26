import numpy as np
import pytest

from pyqpanda_alg.QShadow import (
    PauliObservable,
    PauliTerm,
    ShadowBudgetController,
    ShadowDataset,
    empirical_bernstein_radius,
    estimate_observable,
    estimate_observables,
    optimize_basis_probabilities,
    sample_measurement_plan,
    simulate_pauli_measurements,
    term_shadow_samples,
    uniform_probabilities,
)


def test_term_samples_use_matches_signs_and_inverse_propensities():
    probabilities = np.asarray([[0.2, 0.3, 0.5]])
    dataset = ShadowDataset.from_strings(
        ["X", "Y", "Z", "Z"], ["0", "0", "0", "1"], probabilities
    )
    samples = term_shadow_samples(dataset, PauliTerm("Z", 2.0))
    assert np.allclose(samples, [0.0, 0.0, 4.0, -4.0])


def test_identity_observable_is_exact_for_every_dataset():
    plan = sample_measurement_plan(uniform_probabilities(2), 30, seed=1)
    dataset = simulate_pauli_measurements([1, 0, 0, 0], plan, seed=2)
    identity = PauliObservable.from_terms({"II": 2.5}, name="identity")
    estimate = estimate_observable(dataset, identity)
    assert estimate.value == 2.5
    assert estimate.half_width == 0.0
    assert estimate.minimum_term_matches == 30


def test_monte_carlo_estimate_is_unbiased_with_fixed_seed_regression():
    observable = PauliObservable.from_terms(
        {"ZI": 0.8, "IZ": -0.4, "XX": 0.2, "II": 0.1}, name="hamiltonian"
    )
    state = np.asarray([1, 0, 0, 0], dtype=complex)
    probabilities = optimize_basis_probabilities(observable).probabilities
    plan = sample_measurement_plan(probabilities, 12_000, seed=10)
    dataset = simulate_pauli_measurements(state, plan, seed=11)
    estimate = estimate_observable(dataset, observable, confidence=0.95)
    exact = observable.exact_expectation(state)
    assert estimate.value == pytest.approx(exact, abs=0.035)
    assert estimate.lower <= exact <= estimate.upper


def test_concatenated_rounds_preserve_each_round_probabilities():
    observable = PauliObservable.from_terms({"Z": 1.0}, name="z")
    first_plan = sample_measurement_plan(uniform_probabilities(1), 4_000, seed=20)
    second_probabilities = np.asarray([[0.05, 0.05, 0.90]])
    second_plan = sample_measurement_plan(second_probabilities, 4_000, seed=21)
    state = [1, 0]
    combined = ShadowDataset.concatenate(
        simulate_pauli_measurements(state, first_plan, seed=22),
        simulate_pauli_measurements(state, second_plan, seed=23),
    )
    estimate = estimate_observable(combined, observable)
    assert combined.probabilities[0, 0, 2] == pytest.approx(1 / 3)
    assert combined.probabilities[-1, 0, 2] == pytest.approx(0.90)
    assert estimate.value == pytest.approx(1.0, abs=0.04)


def test_many_observables_use_simultaneous_bonferroni_delta():
    plan = sample_measurement_plan(uniform_probabilities(1), 200, seed=30)
    dataset = simulate_pauli_measurements([1, 0], plan, seed=31)
    observables = [
        PauliObservable.from_terms({"Z": 1}, name="z"),
        PauliObservable.from_terms({"X": 1}, name="x"),
    ]
    estimates = estimate_observables(dataset, observables, family_confidence=0.90)
    assert set(estimates) == {"z", "x"}
    assert estimates["z"].failure_probability == pytest.approx(0.05)
    assert estimates["x"].failure_probability == pytest.approx(0.05)


def test_empirical_bernstein_radius_shrinks_with_repeated_data():
    short = empirical_bernstein_radius(
        [1.0, -1.0] * 10, value_range=2.0, failure_probability=0.05
    )
    long = empirical_bernstein_radius(
        [1.0, -1.0] * 1000, value_range=2.0, failure_probability=0.05
    )
    assert long < short


def test_budget_controller_completes_exact_identity_target():
    plan = sample_measurement_plan(uniform_probabilities(1), 100, seed=40)
    dataset = simulate_pauli_measurements([1, 0], plan, seed=41)
    identity = PauliObservable.from_terms({"I": 1}, name="identity")
    report = ShadowBudgetController(
        identity, target_half_width=1e-6, min_shots=100, max_shots=500
    ).evaluate(dataset)
    assert report.complete
    assert not report.exhausted
    assert report.next_batch_shots == 0
    assert report.reason == "target_reached"


def test_budget_controller_never_exceeds_hard_maximum():
    plan = sample_measurement_plan(uniform_probabilities(1), 100, seed=50)
    dataset = simulate_pauli_measurements([1, 0], plan, seed=51)
    observable = PauliObservable.from_terms({"Z": 1}, name="z")
    report = ShadowBudgetController(
        observable,
        target_half_width=1e-9,
        min_shots=50,
        max_shots=150,
        batch_shots=200,
    ).evaluate(dataset)
    assert not report.complete
    assert report.next_batch_shots == 50

    exhausted_data = ShadowDataset.concatenate(
        dataset,
        simulate_pauli_measurements(
            [1, 0], sample_measurement_plan(uniform_probabilities(1), 50, seed=52), seed=53
        ),
    )
    exhausted = ShadowBudgetController(
        observable,
        target_half_width=1e-9,
        min_shots=50,
        max_shots=150,
    ).evaluate(exhausted_data)
    assert exhausted.exhausted
    assert exhausted.next_batch_shots == 0
    assert exhausted.reason == "max_shots_reached"
