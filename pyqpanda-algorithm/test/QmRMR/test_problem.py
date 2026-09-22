import itertools

import numpy as np

from pyqpanda_alg.QmRMR.constrained import QmRMRProblem, cvar_from_probabilities


def test_exact_reference_matches_reference_instance():
    np.random.seed(42)
    n = 6
    relevance = np.random.random(n)
    quadratic = np.random.random((n, n))
    quadratic = (quadratic.T + quadratic) / 2.0
    problem = QmRMRProblem(quadratic, -relevance, 3)

    exact = problem.exact_solution()

    np.testing.assert_allclose(exact.optimal_value, 0.8378619328579009, atol=1e-12)
    assert (0, 1, 1, 1, 0, 0) in exact.optimal_states
    assert len(exact.values) == 20


def test_ising_mapping_matches_binary_objective():
    rng = np.random.default_rng(7)
    quadratic = rng.normal(size=(4, 4))
    linear = rng.normal(size=4)
    problem = QmRMRProblem(quadratic, linear, 2)
    constant, z_terms, zz_terms = problem.ising_coefficients()

    for bits in itertools.product((0, 1), repeat=4):
        z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        mapped = constant + float(z @ z_terms)
        mapped += sum(
            coefficient * z[left] * z[right]
            for (left, right), coefficient in zz_terms.items()
        )
        direct = problem.objective_unconstrained(bits)
        np.testing.assert_allclose(mapped, direct, atol=1e-12)


def test_problem_rejects_infeasible_bits():
    problem = QmRMRProblem(np.eye(3), [0.0, 0.0, 0.0], 1)
    try:
        problem.objective([1, 1, 0])
    except ValueError as exc:
        assert "cardinality" in str(exc)
    else:
        raise AssertionError("infeasible bit vector was accepted")


def test_mrmr_factory_makes_sign_convention_explicit():
    quadratic = np.zeros((3, 3))
    relevance = [0.2, 0.4, 0.6]
    canonical = QmRMRProblem.from_mrmr(
        quadratic,
        relevance,
        1,
        convention="canonical_mrmr",
    )
    plus = QmRMRProblem.from_mrmr(
        quadratic,
        relevance,
        1,
        convention="plus_relevance",
    )
    assert canonical.objective_convention == "canonical_mrmr"
    assert plus.objective_convention == "plus_relevance"
    np.testing.assert_allclose(canonical.objective([0, 0, 1]), -0.6)
    np.testing.assert_allclose(plus.objective([0, 0, 1]), 0.6)


def test_lower_tail_cvar_uses_best_feasible_probability_mass():
    problem = QmRMRProblem.from_mrmr(
        np.zeros((2, 2)),
        [1.0, 2.0],
        1,
    )
    # Raw keys are reversed into feature-index states: raw 10 -> feature 01.
    probabilities = {"10": 0.8, "01": 0.2}
    np.testing.assert_allclose(
        cvar_from_probabilities(problem, probabilities, alpha=0.5),
        -2.0,
    )
    np.testing.assert_allclose(
        cvar_from_probabilities(problem, probabilities, alpha=1.0),
        -1.8,
    )

