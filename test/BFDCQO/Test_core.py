import numpy as np
from scipy.linalg import expm

from pyqpanda_alg.BFDCQO.problems.ising import IsingProblem, qubo_to_ising
from pyqpanda_alg.BFDCQO.core import (
    build_initial_hamiltonian,
    build_problem_hamiltonian,
    first_order_gauge_potential,
    initial_ground_angles,
)
from pyqpanda_alg.BFDCQO.core.pauli import matrix, single, pair
from pyqpanda_alg.BFDCQO.backends.numpy_statevector import apply_pauli_rotation, product_state_from_ry


def test_qubo_to_ising_energy_equivalence():
    Q = np.array([[1.2, -0.7, 0.4], [0.0, -0.5, 1.1], [0.0, 0.0, 0.3]])
    p = qubo_to_ising(Q, convention="upper")
    for idx in range(8):
        x = np.array([(idx >> q) & 1 for q in range(3)], dtype=float)
        f = sum(Q[i, i] * x[i] for i in range(3))
        f += sum(Q[i, j] * x[i] * x[j] for i in range(3) for j in range(i + 1, 3))
        assert np.isclose(f, p.energy_index(idx), atol=1e-12)


def test_first_order_agp_matches_dense_commutator():
    h = np.array([0.4, -0.2, 0.7])
    J = np.array([[0, 0.6, -0.3], [0.6, 0, 0.5], [-0.3, 0.5, 0]])
    p = IsingProblem(h, J)
    bias = np.array([0.2, -0.35, 0.1])
    hi = build_initial_hamiltonian(3, bias)
    hf = build_problem_hamiltonian(p)
    lam = 0.37
    g = first_order_gauge_potential(hi, hf, lam)

    Hi, Hf = matrix(hi), matrix(hf)
    O1 = Hi @ Hf - Hf @ Hi
    Had = (1-lam) * Hi + lam * Hf
    O2 = Had @ O1 - O1 @ Had
    alpha_dense = -np.trace(O1.conj().T @ O1).real / np.trace(O2.conj().T @ O2).real
    assert np.isclose(g.alpha1, alpha_dense, rtol=1e-12, atol=1e-12)


def test_pauli_rotation_matches_matrix_exponential():
    n = 3
    p = pair(n, 0, "Y", 2, "Z")
    rng = np.random.default_rng(2)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    theta = 0.431
    got = apply_pauli_rotation(psi, p, theta)
    P = matrix({p: 1.0})
    want = expm(-1j * theta * P) @ psi
    assert np.allclose(got, want, atol=1e-12)


def test_initial_product_state_is_ground_state():
    bias = np.array([0.0, 0.4, -0.7])
    angles = initial_ground_angles(bias, -1.0)
    psi = product_state_from_ry(angles)
    Hi = matrix(build_initial_hamiltonian(3, bias, -1.0))
    evals = np.linalg.eigvalsh(Hi)
    e = np.vdot(psi, Hi @ psi).real
    assert np.isclose(e, evals.min(), atol=1e-11)


def test_schedule_derivative_matches_finite_difference():
    from pyqpanda_alg.BFDCQO.core import schedule_lambda, schedule_lambda_dot
    T = 0.3
    for t in [0.03, 0.11, 0.19, 0.27]:
        eps = 1e-7
        fd = (schedule_lambda(t + eps, T) - schedule_lambda(t - eps, T)) / (2 * eps)
        assert np.isclose(schedule_lambda_dot(t, T), fd, rtol=1e-7, atol=1e-8)
