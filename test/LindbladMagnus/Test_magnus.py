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

"""Unit tests for the stochastic Magnus expansion module."""

import numpy as np
import pytest

from pyqpanda_alg.LindbladMagnus.magnus import (effective_hamiltonian,
                                                sample_wiener_integrals)


def test_sample_wiener_integrals_shapes():
    """Sampled stochastic integrals must have the documented shapes."""
    integ = sample_wiener_integrals(k=4, dt=0.1,
                                    rng=np.random.RandomState(0))
    assert integ["xis"].shape == (4,)
    assert integ["a0"].shape == (4,)
    assert integ["aij"].shape == (4, 4)
    assert integ["c0"].shape == (4,)
    assert integ["phis"].shape == (4,)
    assert isinstance(integ["alpha_p"], float)


def test_sample_wiener_integrals_reproducible():
    """The same RNG seed must yield identical integrals."""
    i1 = sample_wiener_integrals(3, 0.2, rng=np.random.RandomState(123))
    i2 = sample_wiener_integrals(3, 0.2, rng=np.random.RandomState(123))
    for key in ("xis", "a0", "aij", "c0"):
        np.testing.assert_allclose(i1[key], i2[key])


def test_effective_hamiltonian_matches_reference_order1():
    """H_eff(order=1) must match a hand-coded reference implementation.

    The reference is a direct transcription of the Magnus-1 scheme of
    Huang et al., PRX Quantum 6, 040312 (2025), and was verified against
    the official classical test code of the paper.
    """
    H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
    c1 = np.array([[0, 1.0], [0, 0]], dtype=complex)  # lowering on site 0
    c2 = np.array([[0, 0], [1.0, 0]], dtype=complex)  # raising on site 0
    c_ops = [c1, c2]
    psi = np.array([1.0, 0.0], dtype=complex)
    dt = 0.05
    rng = np.random.RandomState(7)
    integ = sample_wiener_integrals(2, dt, rng=rng)

    H_eff = effective_hamiltonian(H, c_ops, dt, magnus_order=1,
                                  qsd_type="nonlinear", psi=psi,
                                  integrals=integ)

    # Hand-coded reference: rebuild the Magnus-1 exponent directly.
    psi_r = psi / np.linalg.norm(psi)
    rho = np.outer(psi_r.conj(), psi_r)
    expects = np.array([np.trace(rho @ op) for op in c_ops], dtype=complex)
    X0 = -1j * H + sum(
        -0.5 * (op.conj().T + op) @ op + 2.0 * np.real(e) * op
        for op, e in zip(c_ops, expects))
    Omega_ref = X0 * dt + sum(c_ops[i] * np.sqrt(dt) * integ["xis"][i]
                              for i in range(2))
    H_eff_ref = 1j * Omega_ref / dt
    np.testing.assert_allclose(H_eff, H_eff_ref, atol=1e-12)


def test_effective_hamiltonian_higher_orders_run():
    """Schemes II-IV must execute and remain finite."""
    H = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=complex)
    c_ops = [np.array([[0, 0.1], [0, 0]], dtype=complex)]
    psi = np.array([1.0, 0.0], dtype=complex)
    dt = 0.05
    for order in (2, 3, 4):
        H_eff = effective_hamiltonian(H, c_ops, dt, magnus_order=order,
                                      qsd_type="nonlinear", psi=psi,
                                      rng=np.random.RandomState(order))
        assert np.all(np.isfinite(H_eff))
        assert H_eff.shape == H.shape


def test_effective_hamiltonian_linear_vs_nonlinear():
    """The linear QSD drift must drop the state-dependent feedback term."""
    H = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
    c_ops = [np.array([[0, 0.2], [0, 0]], dtype=complex)]
    # Use a *superposition* so that <L> != 0 and the feedback term kicks in.
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    dt = 0.1
    rng = np.random.RandomState(0)
    integ = sample_wiener_integrals(1, dt, rng=rng)
    H_lin = effective_hamiltonian(H, c_ops, dt, magnus_order=1,
                                  qsd_type="linear", integrals=integ)
    H_non = effective_hamiltonian(H, c_ops, dt, magnus_order=1,
                                  qsd_type="nonlinear", psi=psi,
                                  integrals=integ)
    # Sanity check: <L> for the superposition must be non-zero so that the
    # nonlinear feedback term actually differs from the linear one.
    rho = np.outer(psi.conj(), psi)
    expect_L = np.trace(rho @ c_ops[0])
    assert abs(expect_L) > 1e-6
    # The two should differ because the nonlinear version feeds <L> back.
    assert not np.allclose(H_lin, H_non)


def test_effective_hamiltonian_validates_inputs():
    """Invalid ``qsd_type`` and ``magnus_order`` values must raise."""
    H = np.eye(2, dtype=complex)
    c_ops = [np.array([[0, 1], [0, 0]], dtype=complex)]
    with pytest.raises(ValueError):
        effective_hamiltonian(H, c_ops, 0.1, qsd_type="bogus", psi=np.zeros(2))
    with pytest.raises(ValueError):
        effective_hamiltonian(H, c_ops, 0.1, magnus_order=5, psi=np.zeros(2))
