import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm"
    / "pyqpanda_alg"
    / "QWalk"
    / "qwalk.py"
)
spec = importlib.util.spec_from_file_location("qwalk_under_test", MODULE)
qwalk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qwalk
spec.loader.exec_module(qwalk)


def test_reference_zero_steps_is_basis_state():
    p = qwalk.reference_walk_cycle(3, 0, initial_position=5, initial_coin=1)
    expected = np.zeros(8)
    expected[5] = 1.0
    np.testing.assert_allclose(p, expected, atol=0.0, rtol=0.0)


def test_reference_one_step_wraps_left_and_right():
    p = qwalk.reference_walk_cycle(3, 1, initial_position=0, initial_coin=0)
    expected = np.zeros(8)
    expected[7] = 0.5
    expected[1] = 0.5
    np.testing.assert_allclose(p, expected, atol=1e-15, rtol=0.0)


def test_reference_two_steps_has_interference_distribution():
    p = qwalk.reference_walk_cycle(3, 2, initial_position=0, initial_coin=0)
    expected = np.zeros(8)
    expected[6] = 0.25
    expected[0] = 0.50
    expected[2] = 0.25
    np.testing.assert_allclose(p, expected, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("qubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("steps", [0, 1, 2, 3, 7])
def test_reference_is_normalized(qubits, steps):
    p = qwalk.reference_walk_cycle(
        qubits,
        steps,
        initial_position=(2**qubits - 1) // 2,
        initial_coin=steps % 2,
    )
    assert np.all(p >= 0)
    assert np.sum(p) == pytest.approx(1.0, abs=1e-12)


def test_custom_unitary_coin_and_state_contract():
    x_coin = np.array([[0, 1], [1, 0]], dtype=complex)
    state = qwalk.reference_walk_cycle(
        3, 3, initial_position=2, initial_coin=0, coin_matrix=x_coin, return_state=True
    )
    assert state.shape == (2, 8)
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)
    # X alternates the direction: right, left, right from position 2 -> 3.
    assert np.sum(np.abs(state[:, 3]) ** 2) == pytest.approx(1.0)


def test_nonunitary_coin_rejected():
    with pytest.raises(ValueError, match="unitary"):
        qwalk.reference_walk_cycle(2, 1, coin_matrix=np.ones((2, 2)))


def test_principal_displacement_moments_are_explicit_at_wraparound():
    p = np.zeros(8)
    p[7] = 0.5  # -1 from origin 0
    p[1] = 0.5  # +1 from origin 0
    mean, variance = qwalk.principal_displacement_moments(p, origin=0)
    assert mean == pytest.approx(0.0)
    assert variance == pytest.approx(1.0)


def test_convenience_object_matches_functional_reference():
    walk = qwalk.CoinedQuantumWalkCycle(4, 5, initial_position=3, initial_coin=1)
    np.testing.assert_allclose(
        walk.reference(),
        qwalk.reference_walk_cycle(4, 5, initial_position=3, initial_coin=1),
    )
    assert walk.size == 16
    mean, variance = walk.moments()
    assert np.isfinite(mean)
    assert variance >= 0


@pytest.mark.parametrize(
    "args, error",
    [
        ((0, 1), ValueError),
        ((2, -1), ValueError),
        ((2, 1, 4), ValueError),
        ((2, 1, 0, 2), ValueError),
    ],
)
def test_reference_validation(args, error):
    with pytest.raises(error):
        qwalk.reference_walk_cycle(*args)


def test_native_statevector_matches_independent_reference_when_available():
    pytest.importorskip("pyqpanda3")
    for qubits, steps, position, coin in [
        (1, 1, 0, 0),
        (2, 3, 1, 1),
        (3, 4, 7, 0),
    ]:
        walk = qwalk.CoinedQuantumWalkCycle(qubits, steps, position, coin)
        np.testing.assert_allclose(
            walk.run_exact(), walk.reference(), atol=1e-10, rtol=1e-10
        )
