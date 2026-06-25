# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the variational Lindblad-Magnus solver and the
bundled open quantum system models."""

import numpy as np
import pytest

from pyqpanda_alg.LindbladMagnus import (HardwareEfficientAnsatz,
                                         LindbladMagnusSolver, fmo_model,
                                         liouvillian, mesolve,
                                         tfim_model, rpm_model)


# ----------------------------------------------------------------------
#  Exact reference solver
# ----------------------------------------------------------------------
def test_liouvillian_trace_preserving():
    """The Lindblad Liouvillian must be trace-preserving: Tr(L(rho)) = 0."""
    H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
    c_ops = [np.array([[0, 0.3], [0, 0]], dtype=complex)]
    L = liouvillian(H, c_ops)
    # Check that the identity's superoperator image is zero
    # (equivalently, Tr(L(I)) = 0 if c_ops are present).
    d = H.shape[0]
    vec_I = np.eye(d, dtype=complex).reshape(-1)
    out = (L @ vec_I).reshape(d, d)
    # Tr(L(I)) should be 0 because Lindblad preserves trace on density matrices.
    np.testing.assert_allclose(np.trace(out), 0, atol=1e-12)


def test_mesolve_closed_system_matches_unitary():
    """``mesolve`` with no collapse operators must reproduce unitary evolution."""
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    tlist = np.linspace(0, 1.0, 11)
    e_ops = [np.array([[1, 0], [0, 0]], dtype=complex)]

    from scipy.linalg import expm
    expect = mesolve(H, psi0, tlist, [], e_ops)
    for i, t in enumerate(tlist):
        psi_t = expm(-1j * H * t) @ psi0
        np.testing.assert_allclose(expect[0, i],
                                   np.vdot(psi_t, e_ops[0] @ psi_t).real,
                                   atol=1e-9)


def test_mesolve_amplitude_damping_analytic():
    """For a single amplitude-damping channel the excited-state population
    must decay as ``exp(-gamma * t)``."""
    gamma = 0.3
    c_op = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    H = np.zeros((2, 2), dtype=complex)
    psi0 = np.array([0, 1.0], dtype=complex)  # excited
    tlist = np.linspace(0, 5, 11)
    e_ops = [np.array([[0, 0], [0, 1]], dtype=complex)]  # |1><1|

    expect = mesolve(H, psi0, tlist, [c_op], e_ops)
    analytic = np.exp(-gamma * tlist)
    np.testing.assert_allclose(expect[0], analytic, atol=1e-6)


# ----------------------------------------------------------------------
#  Bundled models
# ----------------------------------------------------------------------
def test_fmo_model_dimensions():
    """FMO model must be padded to a 3-qubit Hilbert space."""
    H, c_ops, e_ops, psi0, labels = fmo_model()
    assert H.shape == (8, 8)
    assert len(c_ops) == 7
    assert len(e_ops) == 5
    assert len(labels) == 5
    assert psi0.shape == (8,)
    # The initial state is |Site 1> = basis vector 1.
    np.testing.assert_allclose(psi0, np.array([0, 1, 0, 0, 0, 0, 0, 0],
                                              dtype=complex))


def test_tfim_model_dimensions():
    """TFIM model must use a 2-qubit Hilbert space."""
    H, c_ops, e_ops, psi0, labels = tfim_model()
    assert H.shape == (4, 4)
    assert len(c_ops) == 2
    assert len(e_ops) == 3
    assert len(labels) == 3


def test_rpm_model_dimensions():
    """Radical pair model must return consistent shapes."""
    H, c_ops, e_ops, psi0, labels = rpm_model()
    assert H.shape[0] == H.shape[1]
    assert H.shape[0] >= 4
    assert len(c_ops) >= 2
    assert len(e_ops) == len(labels)


# ----------------------------------------------------------------------
#  Solver integration (kept light to stay within the CI budget)
# ----------------------------------------------------------------------
def test_solver_tfim_short_run():
    """A short TFIM run must reproduce the exact Liouvillian dynamics to
    within the Monte-Carlo noise of a handful of trajectories."""
    H, c_ops, e_ops, psi0, labels = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=psi0)
    times = np.linspace(0.0, 1.0, 11)

    exact = mesolve(H, psi0, times, c_ops, e_ops)
    solver = LindbladMagnusSolver(H, c_ops, ansatz,
                                  qsd_type="nonlinear",
                                  magnus_order=1,
                                  integrator="rk4")
    result = solver.solve(psi0, times, e_ops, traj_num=10, seed=0)
    mean, std = result.expect, result.std

    assert mean.shape == (len(e_ops), len(times))
    assert std.shape == mean.shape
    # At t=0 the simulation must reproduce the initial state.
    np.testing.assert_allclose(mean[:, 0], exact[:, 0], atol=1e-9)
    # The short-run error must stay bounded.
    err = float(np.abs(mean - exact).max())
    assert err < 0.2, f"Max error {err} exceeds tolerance"


def test_solver_return_types_and_shapes():
    """The solver's public API must return the documented shapes."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz)
    times = np.linspace(0, 0.5, 6)
    result = solver.solve(psi0, times, e_ops, traj_num=2, seed=1)
    mean = result.expect
    assert mean.shape == (len(e_ops), len(times))
    assert result.std.shape == mean.shape
    # Probabilities must remain non-negative (projector observables).
    assert np.all(mean >= -1e-9)
