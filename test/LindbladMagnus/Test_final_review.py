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
                                         fmo_model, mesolve,
                                         sample_wiener_integrals,
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
    ``-iH + sum[-0.5 L^\\dagger L + <L>* L]`` using a state with a *complex*
    ``<L>`` so that the conjugation of the feedback term is actually
    exercised (a real ``<L>`` would hide a sign error)."""
    H = np.array([[1.0, 0.3], [0.3, -1.0]], dtype=complex)
    L = np.array([[0, 0.2], [0, 0]], dtype=complex)
    c_ops = [L]
    # Relative i-phase so that <L> = 0.1j is purely imaginary.
    psi = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
    dt = 0.05
    integ = sample_wiener_integrals(1, dt, rng=np.random.RandomState(0))
    H_em = effective_hamiltonian(H, c_ops, dt, magnus_order=0,
                                 qsd_type="nonlinear", psi=psi,
                                 integrals=integ)
    # Manual reference built from the true <psi|L|psi> = vdot(psi, L @ psi).
    expect_L = np.vdot(psi, L @ psi)
    assert abs(expect_L.imag) > 1e-9  # guard: <L> must be complex
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
    ``Tr(|psi><psi| @ L)`` trace formula, including the complex phase."""
    from pyqpanda_alg.LindbladMagnus.magnus import _channel_expectations
    psi = np.array([0.6, 0.8j, 0, 0], dtype=complex)
    psi /= np.linalg.norm(psi)
    H, c_ops, *_ = tfim_model()
    expects_new = _channel_expectations(psi, c_ops)
    # True density matrix |psi><psi| = outer(psi, psi.conj()).  Using
    # outer(psi.conj(), psi) here would silently compare against the conjugate
    # and mask a sign error in the implementation.
    rho = np.outer(psi, psi.conj())
    expects_ref = np.array([np.trace(rho @ op) for op in c_ops], dtype=complex)
    # The state must yield genuinely complex expectations so that returning the
    # complex conjugate (the historical bug) would be detected.
    assert np.any(np.abs(expects_ref.imag) > 1e-9)
    np.testing.assert_allclose(expects_new, expects_ref, atol=1e-12)


def test_no_dead_code_param_gate_names():
    """The ``_PARAM_GATE_NAMES`` variable was removed as dead code."""
    import pyqpanda_alg.LindbladMagnus.ansatz as ansatz_mod
    assert not hasattr(ansatz_mod, "_PARAM_GATE_NAMES")


def test_no_unused_odeint_import():
    """The unused ``odeint`` import was removed from ``models``."""
    import pyqpanda_alg.LindbladMagnus.models as models_mod
    assert not hasattr(models_mod, "odeint")


# ----------------------------------------------------------------------
#  Unravelling correctness (exact trajectories, no variational ansatz)
# ----------------------------------------------------------------------
def _exact_trajectory_ensemble(H, c_ops, psi0, e_ops, times,
                               magnus_order=1, qsd_type="nonlinear",
                               traj_num=200, seed=0):
    """Propagate raw wavefunctions with exact matrix exponentials of H_eff.

    Bypasses the variational ansatz entirely so that any drift / unravelling
    error in :func:`effective_hamiltonian` shows up directly against
    :func:`mesolve`.  For the *nonlinear* QSD the state is renormalised every
    step (so trajectories stay on the unit sphere and the ensemble average of
    ``<psi|O|psi>`` recovers the open-system observable); for the *linear* QSD
    the decaying norm weights the observable instead.
    """
    from scipy.linalg import expm

    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    psi0 = psi0 / np.linalg.norm(psi0)
    k = len(c_ops)
    acc = np.zeros((len(e_ops), len(times)), dtype=float)
    t0_vals = np.array([float(np.vdot(psi0, op @ psi0).real) for op in e_ops])
    for tr in range(traj_num):
        psi = psi0.copy()
        # t = 0 is identical for every trajectory; accumulate it ``traj_num``
        # times so that the final ``acc / traj_num`` recovers the true value.
        acc[:, 0] += t0_vals
        base = seed + tr * 100003
        for i in range(len(times) - 1):
            dt = float(times[i + 1] - times[i])
            rng = np.random.RandomState(base + i)
            integ = sample_wiener_integrals(k, dt, rng=rng)
            heff = effective_hamiltonian(
                H, c_ops, dt, magnus_order=magnus_order, qsd_type=qsd_type,
                psi=psi if qsd_type == "nonlinear" else None, integrals=integ)
            psi = expm(-1j * heff * dt) @ psi
            if qsd_type == "nonlinear":
                psi = psi / np.linalg.norm(psi)
                nrm = 1.0
            else:
                nrm = float(np.vdot(psi, psi).real)
            for oi, op in enumerate(e_ops):
                acc[oi, i + 1] += nrm * float(np.vdot(psi, op @ psi).real)
    return acc / traj_num


def test_unravelling_reproduces_lindblad_for_hermitian_collapse():
    """The stochastic Magnus unravelling must reproduce the exact Lindblad
    solution for a *Hermitian* collapse operator (pure dephasing).

    This is the case that distinguishes the Magnus drift
    ``-1/2(L^dagger+L)L`` from the textbook ``-1/2 L^dagger L``: for Hermitian
    ``L`` the two differ by a factor of two, so a wrong drift would make the
    coherence decay at twice the physical rate and blow the tolerance.  Raw
    trajectories are propagated with exact matrix exponentials (no ansatz), so
    the bound is set by Monte-Carlo noise only.
    """
    gamma = 0.3
    H = np.array([[0.5, 0.1], [0.1, -0.5]], dtype=complex)
    # Hermitian dephasing operator sqrt(gamma) |1><1|.
    L = np.sqrt(gamma) * np.array([[0, 0], [0, 1]], dtype=complex)
    psi0 = np.array([1.0, 0.0], dtype=complex)
    e_ops = [np.array([[1, 0], [0, 0]], dtype=complex),    # |0><0|
             np.array([[0, 1], [1, 0]], dtype=complex)]    # sigma_x (coherence)
    times = np.linspace(0.0, 8.0, 41)

    exact = mesolve(H, psi0, times, [L], e_ops)
    sim = _exact_trajectory_ensemble(H, [L], psi0, e_ops, times,
                                     magnus_order=1, traj_num=200, seed=0)
    err = float(np.abs(sim - exact).max())
    # A correct unravelling is limited by MC noise (~1e-2 for 200 trajectories).
    # A factor-of-two dephasing-rate error would push this well above 0.1.
    assert err < 0.08, f"Hermitian-collapse unravelling error {err} too large"


def test_unravelling_reproduces_lindblad_for_lowering_ops():
    """The unravelling must also reproduce the exact solution for nilpotent
    lowering operators (the TFIM amplitude-damping channels).  This guards the
    other collapse-operator family at the unravelling level, complementing the
    variational solver test which is restricted to a short horizon."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    times = np.linspace(0.0, 3.0, 31)
    exact = mesolve(H, psi0, times, c_ops, e_ops)
    sim = _exact_trajectory_ensemble(H, c_ops, psi0, e_ops, times,
                                     magnus_order=1, traj_num=200, seed=1)
    err = float(np.abs(sim - exact).max())
    assert err < 0.08, f"Lowering-op unravelling error {err} too large"
