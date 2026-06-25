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

"""Stochastic Magnus expansion for the quantum state diffusion (QSD) equation.

This module implements the high-order stochastic Magnus integrators
(Scheme I-IV) and the Euler-Maruyama scheme (Scheme 0) for one step of the
unravelled Lindblad dynamics.  Given a system Hamiltonian :math:`H`, a list of
collapse (Lindblad) operators :math:`\\{L_k\\}` and a time step ``dt``, the
routines in this module return the (generally non-Hermitian) effective
Hamiltonian :math:`H_{\\mathrm{eff}}` whose exponential :math:`\\exp(-i H_{\\mathrm{eff}}
\\Delta t)` propagates a single wave-function trajectory over ``dt``.

The implementation follows
    J.-C. Huang, H.-E. Li, Y.-C. Wang, G.-Z. Zhang, J. Li, H.-S. Hu,
    "Towards Robust Variational Quantum Simulation of Lindblad Dynamics via
    Stochastic Magnus Expansion", PRX Quantum 6, 040312 (2025).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sample_wiener_integrals",
    "effective_hamiltonian",
]


def sample_wiener_integrals(k: int, dt: float, approx_order: int = 1000,
                            rng: np.random.RandomState | None = None) -> dict:
    """Sample the multiple stochastic integrals required by the Magnus schemes.

    The high-order stochastic Magnus expansion needs not only the Wiener
    increments :math:`\\xi_k` but also the Lévy area :math:`a_{ij}` and the
    higher-order iterated integrals :math:`a_0, c_0`.  All of them are
    approximated by truncating the Brownian bridge Fourier series at
    ``approx_order`` terms following Kloeden & Platen.

    Parameters
    ----------
    k : ``int``\n
        Number of independent Wiener processes, i.e. the number of Lindblad
        collapse operators.
    dt : ``float``\n
        Length of the time step.
    approx_order : ``int``, optional (default=1000)\n
        Truncation order :math:`p` of the Brownian bridge Fourier expansion.
    rng : ``numpy.random.RandomState``, optional\n
        Random number generator.  If ``None`` a fresh generator is created.

    Return
    ----------
    integrals : ``dict``\n
        Dictionary with keys ``"xis"``, ``"mus"``, ``"phis"``, ``"a0"``,
        ``"aij"`` and ``"c0"``, each holding the corresponding stochastic
        multi-integrals.
    """
    if rng is None:
        rng = np.random.RandomState()
    p = int(approx_order)
    # Wiener increments and Brownian-bridge Fourier coefficients.
    xis = rng.normal(loc=0.0, scale=1.0, size=k)
    zetas = rng.normal(loc=0.0, scale=1.0, size=(k, p))
    etas = rng.normal(loc=0.0, scale=1.0, size=(k, p))
    mus = rng.normal(loc=0.0, scale=1.0, size=k)
    phis = rng.normal(loc=0.0, scale=1.0, size=k)

    idx = np.arange(1, p + 1, dtype=float)
    R = np.diag(1.0 / idx)

    # Tails of the zeta(2) and zeta(4) series; both tend to zero as p -> infty.
    rho_p = 1.0 / 12.0 - np.sum(1.0 / idx ** 2) / (2.0 * np.pi ** 2)
    alpha_p = np.pi ** 2 / 180.0 - np.sum(1.0 / idx ** 4) / (2.0 * np.pi ** 2)

    a0 = (-(np.sqrt(2.0 * dt) / np.pi)
          * np.sum(zetas / idx[None, :], axis=1)
          - 2.0 * np.sqrt(dt * rho_p) * mus)
    aij = (zetas @ R @ etas.T - etas @ R @ zetas.T) / (2.0 * np.pi)
    c0 = ((np.sqrt(2.0 * dt) / (8.0 * np.pi ** 3))
          * np.sum(zetas / idx[None, :] ** 3, axis=1))

    return {"xis": xis, "a0": a0, "aij": aij, "c0": c0, "phis": phis,
            "etas": etas, "alpha_p": alpha_p}


def _drift_operator(H: np.ndarray, c_ops: list[np.ndarray],
                    channel_expects: np.ndarray) -> np.ndarray:
    """Build the deterministic drift :math:`X_0` of the QSD unravelling.

    For the *nonlinear* unravelling ``channel_expects`` should hold the
    state-dependent expectations :math:`\\langle L_k \\rangle`, while for the
    *linear* unravelling they are zero.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian.
    c_ops : ``list`` of ``ndarray``\n
        List of collapse operators.
    channel_expects : ``ndarray``\n
        Complex array of length ``len(c_ops)`` with the expectation values of
        each collapse operator (zero for the linear QSD).

    Return
    ----------
    X_0 : ``ndarray``\n
        The drift operator :math:`X_0 = -iH + \\sum_k [-\\tfrac{1}{2}(L_k^\\dagger
        + L_k)L_k + 2\\mathrm{Re}(\\langle L_k \\rangle) L_k]`.
    """
    X_0 = -1j * H
    for op, e_op in zip(c_ops, channel_expects):
        X_0 = X_0 + (-0.5 * (op.conj().T + op) @ op + 2.0 * np.real(e_op) * op)
    return X_0


def effective_hamiltonian(H: np.ndarray, c_ops: list[np.ndarray], dt: float,
                          magnus_order: int = 1,
                          qsd_type: str = "nonlinear",
                          nonlinear_corr: bool = False,
                          psi: np.ndarray | None = None,
                          psi_p: np.ndarray | None = None,
                          rng: np.random.RandomState | None = None,
                          integrals: dict | None = None) -> np.ndarray:
    """Construct :math:`H_{\\mathrm{eff}}` for a single stochastic Magnus step.

    The propagator over ``dt`` is :math:`\\exp(\\Omega) = \\exp(-i H_{\\mathrm{eff}}
    \\Delta t)` where :math:`\\Omega` is the Magnus exponent.  Schemes of order
    1-4 are supported together with the zeroth-order Euler-Maruyama scheme.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian.
    c_ops : ``list`` of ``ndarray``\n
        List of collapse operators.
    dt : ``float``\n
        Time step.
    magnus_order : ``int``, optional (default=1)\n
        Order of the stochastic Magnus expansion.  ``0`` selects the
        Euler-Maruyama scheme.
    qsd_type : ``{'nonlinear', 'linear'}``, optional (default='nonlinear')\n
        Unravelling of the Lindblad master equation.
    nonlinear_corr : ``bool``, optional (default=False)\n
        If ``True`` a predictor-corrector is used: the drift operator is
        re-evaluated using the predicted state ``psi_p`` and averaged with the
        drift computed from ``psi``.
    psi, psi_p : ``ndarray``, optional\n
        Current and predicted wave-functions.  Used to evaluate the
        state-dependent expectations for the nonlinear QSD.
    rng : ``numpy.random.RandomState``, optional\n
        Random number generator used to sample the Wiener integrals.  Ignored
        when ``integrals`` is provided.
    integrals : ``dict``, optional\n
        Pre-sampled stochastic integrals as returned by
        :func:`sample_wiener_integrals`.  When ``None`` the integrals are
        sampled internally using ``rng``.

    Return
    ----------
    H_eff : ``ndarray``\n
        Effective Hamiltonian such that :math:`\\exp(-iH_{\\mathrm{eff}}\\Delta t)`
        propagates the wave-function over one step.
    """
    if qsd_type not in ("nonlinear", "linear"):
        raise ValueError(f"qsd_type must be 'nonlinear' or 'linear', got {qsd_type!r}")
    if magnus_order < 0 or magnus_order > 4:
        raise ValueError(f"magnus_order must be in [0, 4], got {magnus_order}")

    k = len(c_ops)
    if integrals is None:
        integrals = sample_wiener_integrals(k, dt, rng=rng)
    xis = integrals["xis"]
    a0 = integrals["a0"]
    aij = integrals["aij"]
    c0 = integrals["c0"]
    phis = integrals["phis"]
    etas = integrals["etas"]
    alpha_p = integrals["alpha_p"]

    # Expectation values <L_k> for the nonlinear unravelling (zero for linear).
    expects = np.zeros(k, dtype=complex)
    expects_p = np.zeros(k, dtype=complex)
    if qsd_type == "nonlinear":
        if psi is None:
            raise ValueError("psi must be provided for the nonlinear QSD.")
        expects = _channel_expectations(psi, c_ops)
        if nonlinear_corr:
            if psi_p is None:
                raise ValueError("psi_p must be provided when nonlinear_corr=True.")
            expects_p = _channel_expectations(psi_p, c_ops)
            expects_used = 0.5 * (expects + expects_p)
        else:
            expects_used = expects
    else:
        expects_used = expects

    X_0 = _drift_operator(H, c_ops, expects_used)
    Omega = X_0 * dt

    if magnus_order > 0:
        # Order >= 1: stochastic kicks + nested commutators.
        sqrt_dt = np.sqrt(dt)
        for i in range(k):
            Omega = Omega + c_ops[i] * sqrt_dt * xis[i]
            if magnus_order > 1:
                comm = X_0 @ c_ops[i] - c_ops[i] @ X_0
                Omega = Omega + comm * a0[i] * dt / 2.0
                for j in range(i + 1, k):
                    comm_ij = c_ops[i] @ c_ops[j] - c_ops[j] @ c_ops[i]
                    if np.abs(comm_ij).sum() != 0:
                        Omega = Omega + 0.5 * comm_ij * (
                            (a0[j] * xis[i] - a0[i] * xis[j]) * sqrt_dt
                            + 2.0 * dt * aij[j, i]
                        )
                if magnus_order > 2:
                    comm = X_0 @ comm - comm @ X_0
                    levy = (np.sqrt(dt * alpha_p) * phis[i]
                            + np.sqrt(dt / 2.0)
                            * np.sum(etas[i] / np.arange(1, etas.shape[1] + 1) ** 2)
                            / np.pi)
                    Omega = Omega + comm * dt ** 2 * levy / (2.0 * np.pi)
                    if magnus_order > 3:
                        comm = X_0 @ comm - comm @ X_0
                        Omega = Omega + comm * dt ** 3 * c0[i]
        H_eff = 1j * Omega / dt
    else:
        # Order 0: Euler-Maruyama (matches the reference implementation:
        # only the c_ops[i]*sqrt(dt)*xi noise is added on top of the drift).
        for i in range(k):
            Omega = Omega + c_ops[i] * np.sqrt(dt) * xis[i]
        H_eff = 1j * Omega / dt
    return H_eff


def _channel_expectations(psi: np.ndarray,
                          c_ops: list[np.ndarray]) -> np.ndarray:
    """Return :math:`\\langle\\psi|L_k|\\psi\\rangle` for every collapse operator.

    Parameters
    ----------
    psi : ``ndarray``\n
        Normalised wave-function.
    c_ops : ``list`` of ``ndarray``\n
        List of collapse operators.

    Return
    ----------
    expects : ``ndarray``\n
        Complex array of expectation values.
    """
    psi = np.asarray(psi).reshape(-1)
    rho = np.outer(psi.conj(), psi)
    return np.array([np.trace(rho @ op) for op in c_ops], dtype=complex)
