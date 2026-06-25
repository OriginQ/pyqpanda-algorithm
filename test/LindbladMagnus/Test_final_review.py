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

"""Regression tests for issues found in the final deep-review pass.

Covers:
* ``magnus_order=0`` (Euler-Maruyama) must use the *standard* linear-QSD
  drift ``-0.5 L_k^\\dagger L_k`` rather than the Magnus drift
  ``-0.5 (L_k^\\dagger + L_k) L_k`` used by Scheme I-IV.  The two coincide
  for nilpotent collapse operators (e.g. lowering operators where
  ``L**2 == 0``) but differ for Hermitian dephasing operators.
* ``sample_wiener_integrals`` must return the documented set of keys.
* ``_channel_expectations`` must match the trace formula after the ``vdot``
  optimisation.
* No dead code / unused imports.
"""

import numpy as np

from pyqpanda_alg.LindbladMagnus import (effective_hamiltonian,
                                         fmo_model, sample_wiener_integrals,
                                         tfim_model)


def test_euler_maruyama_uses_standard_linear_drift():
    """For Hermitian dephasing operators ``L=L^\\dagger`` we have
    ``(L^\\dagger + L)L = 2 L^\\dagger L``, so the Magnus drift is twice the
    Euler-Maruyama drift.  ``magnus_order=0`` must therefore differ from
    ``magnus_order=1``."""
    H, c_ops, e_ops, psi0, _ = fmo_model()
    dt = 1.0
    integ = sample_wiener_integrals(len(c_ops), dt, rng=np.random.RandomState(7))

    H_em = effective_hamiltonian(H, c_ops, dt, magnus_order=0,
                                 qsd_type="nonlinear", psi=psi0,
                                 integrals=integ)
    H_m1 = effective_hamiltonian(H, c_ops, dt, magnus_order=1,
                                 qsd_type="nonlinear", psi=psi0,
                                 integrals=integ)
    assert not np.allclose(H_em, H_m1), (
        "Euler-Maruyama and Magnus-I H_eff must differ for FMO (dephasing "
        "operators have L^2 != 0).")


def test_euler_maruyama_coincides_with_magnus_for_lowering_ops():
    """For nilpotent lowering operators ``L**2 = 0``, hence
    ``(L^\\dagger + L)L == L^\\dagger L`` and the two drifts coincide."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    # TFIM collapse operators are sigma_- (lowering), L^2 = 0.
    for L in c_ops:
        assert np.allclose(L @ L, 0), "TFIM c_ops should be nilpotent"
    dt = 0.1
    integ = sample_wiener_integrals(len(c_ops), dt, rng=np.random.RandomState(3))
    H_em = effective_hamiltonian(H, c_ops, dt, magnus_order=0,
                                 qsd_type="nonlinear", psi=psi0,
                                 integrals=integ)
    H_m1 = effective_hamiltonian(H, c_ops, dt, magnus_order=1,
                                 qsd_type="nonlinear", psi=psi0,
                                 integrals=integ)
    np.testing.assert_allclose(H_em, H_m1, atol=1e-12)


def test_euler_maruyama_drift_matches_reference_formula():
    """Hand-check the Euler-Maruyama drift against the reference formula
    ``-iH + sum[-0.5 L^\\dagger L + <L>* L]``."""
    H = np.array([[1.0, 0.3], [0.3, -1.0]], dtype=complex)
    L = np.array([[0, 0.2], [0, 0]], dtype=complex)
    c_ops = [L]
    psi = np.array([1.0, 0.0], dtype=complex)
    dt = 0.05
    integ = sample_wiener_integrals(1, dt, rng=np.random.RandomState(0))
    H_em = effective_hamiltonian(H, c_ops, dt, magnus_order=0,
                                 qsd_type="nonlinear", psi=psi,
                                 integrals=integ)
    # Manual reference
    expect_L = np.vdot(L @ psi, psi)
    X_em = -1j * H + (-0.5 * L.conj().T @ L + np.conj(expect_L) * L)
    Omega = X_em * dt + L * np.sqrt(dt) * integ["xis"][0]
    H_em_ref = 1j * Omega / dt
    np.testing.assert_allclose(H_em, H_em_ref, atol=1e-12)


def test_sample_wiener_integrals_returns_documented_keys():
    """The returned dict must contain exactly the documented keys."""
    integ = sample_wiener_integrals(k=3, dt=0.1, rng=np.random.RandomState(0))
    expected = {"xis", "a0", "aij", "c0", "phis", "etas", "alpha_p"}
    assert set(integ.keys()) == expected


def test_channel_expectations_matches_trace_formula():
    """The ``vdot``-based implementation must match the canonical
    ``Tr(|psi><psi| @ L)`` trace formula."""
    from pyqpanda_alg.LindbladMagnus.magnus import _channel_expectations
    psi = np.array([0.6, 0.8j, 0, 0], dtype=complex)
    psi /= np.linalg.norm(psi)
    H, c_ops, *_ = tfim_model()
    expects_new = _channel_expectations(psi, c_ops)
    rho = np.outer(psi.conj(), psi)
    expects_ref = np.array([np.trace(rho @ op) for op in c_ops], dtype=complex)
    np.testing.assert_allclose(expects_new, expects_ref, atol=1e-12)


def test_no_dead_code_param_gate_names():
    """The ``_PARAM_GATE_NAMES`` variable was removed as dead code."""
    import pyqpanda_alg.LindbladMagnus.ansatz as ansatz_mod
    assert not hasattr(ansatz_mod, "_PARAM_GATE_NAMES")


def test_no_unused_odeint_import():
    """The unused ``odeint`` import was removed from ``models``."""
    import pyqpanda_alg.LindbladMagnus.models as models_mod
    assert not hasattr(models_mod, "odeint")
