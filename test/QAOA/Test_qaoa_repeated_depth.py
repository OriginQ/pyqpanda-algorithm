"""Changing QAOA depth must use every angle on each call, independent of history."""

import numpy as np
import pytest
import sympy as sp
from pyqpanda3.hamiltonian import PauliOperator
from pyqpanda_alg.QAOA.qaoa import QAOA

GAMMAS = [0.31, -0.47, 0.23]
BETAS = [0.17, 0.29, -0.41]
# Basis order is |x1 x0>; compute the cost independently of the library.
ENERGIES = np.array(
    [0.2 + 0.7 * (i & 1) - 1.1 * (i >> 1) + 1.3 * (i & 1) * (i >> 1) for i in range(4)]
)


def _problem(representation: str = "symbolic") -> QAOA:
    """Construct equivalent full-width costs without relying on pending fixes."""
    if representation == "pauli":
        return QAOA(
            PauliOperator({"": 0.325, "Z0": -0.675, "Z1": 0.225, "Z0 Z1": 0.325})
        )
    x0, x1 = sp.symbols("x0 x1")
    return QAOA(0.2 + 0.7 * x0 - 1.1 * x1 + 1.3 * x0 * x1)


def _reference(gammas: list[float], betas: list[float]) -> np.ndarray:
    """Evolve with analytic diagonal costs and dense RX matrices, without SDK."""
    state = np.ones(4, dtype=complex) / 2
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * ENERGIES)
        rx = np.array(
            [[np.cos(beta), 1j * np.sin(beta)], [1j * np.sin(beta), np.cos(beta)]]
        )
        state = np.kron(rx, rx) @ state
    return np.abs(state) ** 2


def _assert_distribution(
    actual: dict[str, float], gammas: list[float], betas: list[float]
) -> None:
    """Compare every probability, including states omitted from SDK output."""
    values = [actual.get(format(i, "02b"), 0.0) for i in range(4)]
    np.testing.assert_allclose(values, _reference(gammas, betas), atol=1e-12, rtol=0)


@pytest.mark.parametrize("representation", ["symbolic", "pauli"])
@pytest.mark.parametrize("depths", [(1, 2, 3, 1), (3, 2, 1, 3), (0, 2, 0, 1)])
def test_depth_sweep_matches_dense_evolution(
    representation: str, depths: tuple[int, ...]
) -> None:
    """Growing, shrinking and zero-layer calls use current angles, not history."""
    problem = _problem(representation)
    for count, depth in enumerate(depths, start=1):
        gammas, betas = GAMMAS[:depth], BETAS[:depth]
        # Exercise existing list, tuple and NumPy input support.
        actual = problem.run_qaoa_circuit(np.array(gammas), tuple(betas), shots=-1)
        _assert_distribution(actual, gammas, betas)
        assert problem.layer == depth
        assert problem.circuit_iter == count


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("lengths", [(1, 2), (2, 1), (0, 1), (1, 0)])
def test_unpaired_angles_rejected_without_changing_depth(
    warmup: bool, lengths: tuple[int, int]
) -> None:
    """Reject unused or missing angles without damaging the next valid run."""
    problem = _problem()
    if warmup:
        problem.run_qaoa_circuit(GAMMAS[:1], BETAS[:1])
    old_layer, old_count = problem.layer, problem.circuit_iter
    with pytest.raises(ValueError, match="same length"):
        problem.run_qaoa_circuit(GAMMAS[: lengths[0]], BETAS[: lengths[1]])
    assert problem.layer == old_layer
    assert problem.circuit_iter == old_count
    actual = problem.run_qaoa_circuit(GAMMAS[:2], BETAS[:2])
    _assert_distribution(actual, GAMMAS[:2], BETAS[:2])


@pytest.mark.parametrize("mode", ["default", "interp"])
def test_optimizer_and_direct_calls_can_share_one_instance(mode: str) -> None:
    """Actual optimizers retain final depth and allow a subsequent direct sweep."""
    problem = _problem()
    problem.run_qaoa_circuit(GAMMAS, BETAS)
    initial = [0.31, 0.17] if mode == "interp" else [0.31, -0.47, 0.17, 0.29]
    distribution, parameters, loss = problem.run(
        layer=2,
        initial_para=initial,
        optimize_type=mode,
        optimizer="SLSQP",
        optimizer_option={"options": {"maxiter": 1}},
        shots=-1,
    )
    gammas, betas = list(parameters[:2]), list(parameters[2:])
    assert problem.layer == 2
    assert len(parameters) == 4
    _assert_distribution(distribution, gammas, betas)
    assert loss == pytest.approx(float(_reference(gammas, betas) @ ENERGIES), abs=1e-12)
    actual = problem.run_qaoa_circuit(GAMMAS[:1], BETAS[:1])
    _assert_distribution(actual, GAMMAS[:1], BETAS[:1])
