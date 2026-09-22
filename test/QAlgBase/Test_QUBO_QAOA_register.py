"""Preserve declared QUBO variables, including zero-coefficient variables."""

import numpy as np
import pytest
from pyqpanda_alg.QUBO import QUBO_QAOA


@pytest.mark.parametrize(
    "linear, quadratic",
    [
        ([0, 2], None),
        ([2, 0], None),
        ([0, 2, 0], None),
        ([0, 0, 2], None),
        ([0, 0, 0], None),
        ([0, 0, 0], [[0, 0, 0], [0, 0, 3], [0, 0, 0]]),
        ([0, -2], [[0, 0], [0, 2]]),
        ([1, 2, 3], None),
    ],
    ids=[
        "unused-first-variable",
        "unused-last-variable",
        "unused-both-ends",
        "only-last-variable-active",
        "constant-objective",
        "quadratic-only-with-unused-first",
        "cancelled-coefficients",
        "dense-control",
    ],
)
def test_declared_variable_width_and_full_distribution(
    linear: list[int], quadratic: list[list[int]] | None
) -> None:
    """Compare actual CPU output with an independent dense one-layer evolution."""
    width = len(linear)
    constant = 3
    matrix = np.zeros((width, width)) if quadratic is None else np.asarray(quadratic)
    gamma, beta = np.random.RandomState(7).random(2) * np.pi
    states = [format(key, f"0{width}b") for key in range(2**width)]
    costs = []
    for bits in states:
        variables = np.array([int(bit) for bit in bits])
        costs.append(variables @ matrix @ variables + variables @ linear + constant)
    state = np.exp(-1j * gamma * np.asarray(costs)) / np.sqrt(2**width)
    rotation = np.array(
        [[np.cos(beta), 1j * np.sin(beta)], [1j * np.sin(beta), np.cos(beta)]]
    )
    mixer = np.array([[1.0 + 0j]])
    for _ in range(width):
        mixer = np.kron(mixer, rotation)
    expected = np.abs(mixer @ state) ** 2

    model = QUBO_QAOA({"quadratic": quadratic, "linear": linear, "constant": constant})
    random_state = np.random.get_state()
    try:
        np.random.seed(7)
        # Zero optimizer iterations evaluate the seeded initial parameters and
        # still run the real cost circuit, mixer, CPUQVM and result decoding.
        result = model.run(layer=1, optimizer_option={"options": {"maxiter": 0}})
    finally:
        np.random.set_state(random_state)

    assert set(result) == set(states), "Every declared variable needs an output bit"
    actual = np.array([result[bits] for bits in states])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert sum(result.values()) == pytest.approx(1.0)
