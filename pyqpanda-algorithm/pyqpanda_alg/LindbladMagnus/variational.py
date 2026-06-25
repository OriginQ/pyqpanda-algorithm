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

"""McLachlan variational principle for (non-Hermitian) time evolution.

Given a parameterised ansatz :math:`|\\psi(\\boldsymbol\\theta)\\rangle` and an
effective (possibly non-Hermitian) generator :math:`H_{\\mathrm{eff}}`, this
module integrates the McLachlan variational equations

.. math::
    \\sum_j M_{ij}\\dot\\theta_j = V_i, \\qquad
    M_{ij} = \\mathrm{Re}\\!\\left[
        \\langle\\partial_i\\psi|\\partial_j\\psi\\rangle -
        \\langle\\partial_i\\psi|\\psi\\rangle\\langle\\psi|\\partial_j\\psi\\rangle
    \\right],

where the right-hand side :math:`V_i` aggregates the real-time (Hermitian) and
the imaginary-time (non-Hermitian) contributions of :math:`H_{\\mathrm{eff}}`.

Both first-order (forward Euler) and fourth-order Runge-Kutta integrators are
provided.  The module is agnostic to the choice of ansatz as long as it exposes
:meth:`get_statevector` and :meth:`get_jacobian`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mclachlan_system", "variational_step_euler", "variational_step_rk4"]


def mclachlan_system(ansatz, theta: np.ndarray, H_eff: np.ndarray,
                     eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the Fubini-Study metric :math:`M` and the RHS :math:`V`.

    The general (possibly non-Hermitian) generator ``H_eff`` is handled through
    the *least-squares* McLachlan principle.  Decomposing the projected quantum
    geometric tensor as :math:`A = A_R + i A_I` and defining
    :math:`B_i = \\langle\\partial_i\\psi|(I-|\\psi\\rangle\\langle\\psi|)H_{\\mathrm{eff}}
    |\\psi\\rangle`, the variational equation :math:`A\\dot{\\boldsymbol\\theta}
    = -iB` (with real :math:`\\dot{\\boldsymbol\\theta}`) splits into two real
    equations

    .. math::
        A_R \\dot{\\boldsymbol\\theta} = \\mathrm{Im}(B), \\qquad
        A_I \\dot{\\boldsymbol\\theta} = -\\mathrm{Re}(B).

    For Hermitian :math:`H_{\\mathrm{eff}}` the imaginary-time equation is
    identically satisfied and the routine reduces to the standard real-time
    McLachlan principle.  For non-Hermitian generators (Lindblad QSD) the two
    equations are stacked and solved in the least-squares sense.

    Parameters
    ----------
    ansatz : object with ``get_statevector`` and ``get_jacobian``\n
        The parameterised ansatz describing the variational manifold.
    theta : ``ndarray`` of shape ``(n_parameters,)``\n
        Current variational parameters.
    H_eff : ``ndarray``\n
        Effective (possibly non-Hermitian) Hamiltonian.
    eps : ``float``, optional (default=1e-12)\n
        Regularisation added to the diagonal of the metric before solving.

    Return
    ----------
    dtheta : ``ndarray`` of shape ``(n_parameters,)``\n
        Time derivative :math:`\\dot{\\boldsymbol\\theta}` of the parameters.
    M : ``ndarray`` of shape ``(n_parameters, n_parameters)``\n
        Real part of the Fubini-Study metric (returned for diagnostics).
    imag_norm_rate : ``float``\n
        Instantaneous rate :math:`\\mathrm{Im}\\langle H_{\\mathrm{eff}}\\rangle`
        driving the wave-function norm evolution of the underlying trajectory.
        The norm obeys :math:`\\dot N = 2N\\cdot` ``imag_norm_rate``.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    psi = ansatz.get_statevector(theta)
    psi = psi / np.linalg.norm(psi)
    jac = ansatz.get_jacobian(theta)  # (dim, n_params)

    # Project out the gauge component so that <psi|d_i psi> is absorbed.
    f = jac.conj().T @ psi                      # <d_i psi | psi>
    jac_perp = jac - np.outer(psi, f.conj())    # (I - |psi><psi|)|d_i psi>

    A = jac_perp.conj().T @ jac_perp            # projected metric (complex)
    A_R = np.real(A)
    A_R = 0.5 * (A_R + A_R.T)                   # enforce numerical symmetry
    A_I = np.imag(A)

    Hpsi = H_eff @ psi
    B = jac_perp.conj().T @ Hpsi
    B_R = np.real(B)
    B_I = np.imag(B)

    # Stack the real and imaginary McLachlan equations and solve the
    # resulting real over-determined system in the least-squares sense.
    M_full = np.vstack([A_R, A_I])
    V_full = np.concatenate([B_I, -B_R])
    # Tikhonov regularisation through the normal equations to stay close to the
    # reference implementation's pinv approach.
    G = M_full.T @ M_full + eps * np.eye(M_full.shape[1])
    dtheta = np.linalg.solve(G, M_full.T @ V_full)

    # Norm evolution rate: d(ln N)/dt = 2 Im(<H_eff>)
    norm_rate = float(np.imag(np.vdot(psi, Hpsi)))

    return dtheta, A_R, norm_rate


def variational_step_euler(ansatz, theta: np.ndarray, H_eff: np.ndarray,
                           dt: float, psi_norm: float,
                           eps: float = 1e-12
                           ) -> tuple[np.ndarray, float]:
    """Single forward-Euler variational step.

    Parameters
    ----------
    ansatz : ansatz object\n
        Parameterised variational ansatz.
    theta : ``ndarray``\n
        Current variational parameters.
    H_eff : ``ndarray``\n
        Effective Hamiltonian for this step.
    dt : ``float``\n
        Time step.
    psi_norm : ``float``\n
        Current wave-function norm :math:`N` tracked for the linear QSD.
    eps : ``float``, optional\n
        Regularisation passed to :func:`mclachlan_system`.

    Return
    ----------
    theta_new : ``ndarray``\n
        Updated variational parameters.
    psi_norm_new : ``float``\n
        Updated wave-function norm.
    """
    dtheta, _, rate = mclachlan_system(ansatz, theta, H_eff, eps=eps)
    theta_new = theta + dt * dtheta
    psi_norm_new = psi_norm * np.exp(2.0 * rate * dt)
    return theta_new, psi_norm_new


def variational_step_rk4(ansatz, theta: np.ndarray, H_eff: np.ndarray,
                         dt: float, psi_norm: float,
                         eps: float = 1e-12
                         ) -> tuple[np.ndarray, float]:
    """Classical fourth-order Runge-Kutta variational step.

    The same effective Hamiltonian ``H_eff`` is reused for all four stages,
    which matches the reference implementation of the paper.

    Parameters
    ----------
    ansatz : ansatz object\n
        Parameterised variational ansatz.
    theta : ``ndarray``\n
        Current variational parameters.
    H_eff : ``ndarray``\n
        Effective Hamiltonian for this step (kept constant across the RK4
        sub-steps).
    dt : ``float``\n
        Time step.
    psi_norm : ``float``\n
        Current wave-function norm.
    eps : ``float``, optional\n
        Regularisation passed to :func:`mclachlan_system`.

    Return
    ----------
    theta_new : ``ndarray``\n
        Updated variational parameters.
    psi_norm_new : ``float``\n
        Updated wave-function norm.
    """
    # Stage 1 at theta.
    k1, _, r1 = mclachlan_system(ansatz, theta, H_eff, eps=eps)
    # Stage 2 at theta + dt*k1/2.
    k2, _, r2 = mclachlan_system(ansatz, theta + 0.5 * dt * k1, H_eff, eps=eps)
    # Stage 3 at theta + dt*k2/2.
    k3, _, r3 = mclachlan_system(ansatz, theta + 0.5 * dt * k2, H_eff, eps=eps)
    # Stage 4 at theta + dt*k3.
    k4, _, r4 = mclachlan_system(ansatz, theta + dt * k3, H_eff, eps=eps)

    theta_new = theta + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    # Integrate d(ln N)/dt = 2 Im(<H_eff>) with classical RK4.
    rs = np.array([r1, r2, r3, r4])
    ks_log = 2.0 * rs
    ln_n = (ks_log[0] + 2.0 * ks_log[1] + 2.0 * ks_log[2] + ks_log[3]) * dt / 6.0
    psi_norm_new = psi_norm * np.exp(ln_n)
    return theta_new, psi_norm_new
