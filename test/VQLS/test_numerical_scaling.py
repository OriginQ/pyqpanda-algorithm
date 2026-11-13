"""Regression cases for the finite-scale, full-statevector VQLS objective."""

import numpy as np
import pytest

from pyqpanda_alg.VQLS import VQLS


MATRIX = np.array([[1 + 1j, .5 - .25j], [.25j, 2 - .5j]])
SOLUTION = np.array([.5 + .75j, -.25 + .5j])
VECTOR = MATRIX @ SOLUTION


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200, 1e-310])
@pytest.mark.parametrize("state_scale", [1.0, 1e-200, 1e200, 1e-310])
def test_known_complex_solution_is_scale_invariant(scale, state_scale):
    solver = VQLS(MATRIX * scale, VECTOR * scale,
                  state_fn=lambda _: SOLUTION * state_scale, num_parameters=1)
    state, solution, residual = solver.solution([0])
    # Subnormal inputs lose several decimal places before the solver sees them.
    tolerance = 5e-13 if min(scale, state_scale) < 1e-300 else 2e-15
    np.testing.assert_allclose(solution, SOLUTION, rtol=tolerance, atol=tolerance)
    assert abs(np.linalg.norm(state) - 1) < 5e-16
    assert solver.cost([0]) < 1e-14
    assert residual / scale < tolerance * np.linalg.norm(VECTOR)


@pytest.mark.parametrize("a_scale,b_scale,expected", [
    (1e200, 1e-100, 1e-300),
    (1e-200, 1e100, 1e300),
    (1e-310, 1e-10, 1e300),
    (1e200, 1e-200, 0.0),
])
def test_reconstruction_and_residual_follow_returned_solution(a_scale, b_scale, expected):
    solver = VQLS(np.eye(2) * a_scale, np.array([b_scale, 0.]),
                  state_fn=lambda _: [1., 0.], num_parameters=1)
    _, solution, residual = solver.solution([0])
    np.testing.assert_allclose(solution, [expected, 0.], rtol=1e-13, atol=0.)
    if expected == 0:
        # The formal solution is below the smallest subnormal, so the returned
        # vector is zero. The residual must include the entire RHS in this case.
        assert residual == b_scale
    else:
        assert residual / b_scale < 1e-13


def test_unrepresentable_solution_raises():
    solver = VQLS(np.eye(2) * 1e-200, [1e200, 0.],
                  state_fn=lambda _: [1., 0.], num_parameters=1)
    with pytest.raises(ValueError, match="cannot be represented"):
        solver.solution([0])


def test_large_representable_solution_and_residual():
    # Rebase of the solution must not overflow before applying a negative
    # binary exponent. This caught an intermediate-multiplication failure.
    solver = VQLS(np.eye(2) * .75, [1.2e308, 0.],
                  state_fn=lambda _: [1., 0.], num_parameters=1)
    _, solution, residual = solver.solution([0])
    np.testing.assert_allclose(solution, [1.6e308, 0.], rtol=2e-15, atol=0.)
    assert residual / 1.2e308 < 2e-15


def test_large_solution_along_smaller_matrix_direction():
    # Independent review reproduction: a finite solution must not make the
    # residual NaN just because a scale-ratio intermediate overflows.
    solver = VQLS(np.diag([1.7, .05]), [0., 8e306],
                  state_fn=lambda _: [0., 1.], num_parameters=1)
    _, solution, residual = solver.solution([0])
    np.testing.assert_allclose(solution, [0., 1.6e308], rtol=2e-15, atol=0.)
    assert residual / 8e306 < 2e-15


def test_input_arrays_are_snapshotted():
    matrix = np.eye(2, dtype=complex)
    vector = np.array([1., 0.], dtype=complex)
    solver = VQLS(matrix, vector, state_fn=lambda _: [1., 0.], num_parameters=1)
    matrix[0, 0], vector[0] = 2., 3.
    assert solver.cost([0]) == 0.
    np.testing.assert_array_equal(solver.solution([0])[1], [1., 0.])
    np.testing.assert_array_equal(solver.matrix, np.eye(2))
    np.testing.assert_array_equal(solver.vector, [1., 0.])
    with pytest.raises(ValueError):
        solver.matrix[0, 0] = 2.
    with pytest.raises(ValueError):
        solver.vector[0] = 2.


@pytest.mark.parametrize("bad_state", [[0., 0.], [np.nan, 1.], [np.inf, 1.]])
def test_invalid_states_rejected(bad_state):
    solver = VQLS(np.eye(2), [1., 0.], state_fn=lambda _: bad_state, num_parameters=1)
    with pytest.raises(ValueError, match="zero or non-finite"):
        solver.state([0])


@pytest.mark.parametrize("argument", ["learning_rate", "perturbation", "gradient_clip"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_optimizer_parameters_rejected(argument, value):
    with pytest.raises(ValueError, match="finite and positive"):
        VQLS(np.eye(2), [1., 0.], **{argument: value})


def test_native_pyqpanda_state_factory():
    from pyqpanda3.core import CPUQVM, QProg, RY

    machine = CPUQVM()
    def state_fn(parameters):
        machine.run(QProg(1) << RY(0, float(parameters[0])), 1)
        return np.asarray(machine.result().get_state_vector())

    a = np.array([[1., -1/3], [-1/3, 1.]])
    b = np.array([1., 0.])
    exact = np.linalg.solve(a, b)
    params = [2 * np.arctan2(exact[1], exact[0])]
    for scale in [1., 1e-200, 1e200]:
        solver = VQLS(a * scale, b * scale, state_fn=state_fn, num_parameters=1)
        np.testing.assert_allclose(solver.solution(params)[1], exact, rtol=2e-15)
        assert solver.cost(params) < 1e-14


def test_spsa_optimization_has_same_scale_independent_result():
    a = np.array([[1., -1/3], [-1/3, 1.]])
    b = np.array([1., 0.])
    outputs = [VQLS(a * scale, b * scale, layers=2, max_iterations=400,
                    tolerance=1e-7, learning_rate=.35, perturbation=.15, seed=7).solve()
               for scale in [1., 1e-200, 1e200]]
    baseline = outputs[0]
    assert baseline.cost < .01
    for result in outputs[1:]:
        np.testing.assert_allclose(result.solution, baseline.solution, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.cost_history, baseline.cost_history, atol=1e-14)
