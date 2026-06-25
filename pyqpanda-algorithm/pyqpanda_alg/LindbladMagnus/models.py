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

"""Open-quantum-system models bundled with the Lindblad-Magnus solver.

The :func:`fmo_model`, :func:`tfim_model` and :func:`rpm_model` helpers return
the Hamiltonian, collapse operators, observables and initial state for the
three systems studied in

    Huang et al., "Towards Robust Variational Quantum Simulation of Lindblad
    Dynamics via Stochastic Magnus Expansion", PRX Quantum 6, 040312 (2025).

A reference exact solver :func:`mesolve` is provided as well.  It vectorises
the Lindblad master equation through the Liouvillian super-operator and
integrates it with :mod:`scipy.integrate`, removing any external dependency on
``qutip`` while keeping the same numerical content.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint, solve_ivp

__all__ = [
    "fmo_model",
    "tfim_model",
    "rpm_model",
    "mesolve",
    "liouvillian",
]


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def _pad_to_power_of_two(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero-pad ``M`` (square) to the nearest :math:`2^n \\times 2^n` matrix.

    Returns the padded matrix together with the number of qubits.
    """
    n = M.shape[0]
    n_qubits = int(np.ceil(np.log2(max(n, 2))))
    dim = 1 << n_qubits
    if dim == n:
        return M.astype(complex), n_qubits
    padded = np.zeros((dim, dim), dtype=complex)
    padded[:n, :n] = M
    return padded, n_qubits


def liouvillian(H: np.ndarray, c_ops: list[np.ndarray]) -> np.ndarray:
    """Return the Lindblad Liouvillian super-operator.

    For a density matrix :math:`\\rho` the master equation reads
    :math:`\\dot\\rho = \\mathcal{L}\\rho = -i[H, \\rho] + \\sum_k
    \\left(L_k\\rho L_k^\\dagger - \\tfrac{1}{2}\\{L_k^\\dagger L_k, \\rho\\}
    \\right)`.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian.
    c_ops : ``list`` of ``ndarray``\n
        Collapse operators.

    Return
    ----------
    L : ``ndarray``\n
        The Liouvillian super-operator with shape ``(d**2, d**2)``.
    """
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for op in c_ops:
        op = np.asarray(op, dtype=complex)
        op_d = op.conj().T
        op_dag_op = op_d @ op
        L = L + np.kron(op, op.conj()) - 0.5 * (
            np.kron(op_dag_op, I) + np.kron(I, op_dag_op.T))
    return L


def mesolve(H: np.ndarray, psi0: np.ndarray, tlist: np.ndarray,
            c_ops: list[np.ndarray], e_ops: list[np.ndarray],
            method: str = "RK45") -> np.ndarray:
    """Solve the Lindblad master equation exactly by Liouvillian vectorisation.

    The routine is equivalent to ``qutip.mesolve`` for the supported inputs
    but only relies on :mod:`numpy` and :mod:`scipy`, removing any external
    dependency from the package.

    Parameters
    ----------
    H : ``ndarray``\n
        System Hamiltonian.
    psi0 : ``ndarray``\n
        Initial pure state.  Mixed-state inputs are also accepted as long as
        they are square density matrices.
    tlist : ``ndarray``\n
        Time grid.
    c_ops : ``list`` of ``ndarray``\n
        Collapse operators.
    e_ops : ``list`` of ``ndarray``\n
        Observables whose expectation values are returned.
    method : ``str``, optional (default='RK45')\n
        Integrator passed to :func:`scipy.integrate.solve_ivp`.

    Return
    ----------
    expect : ``ndarray`` of shape ``(len(e_ops), len(tlist))``\n
        Expectation values of ``e_ops``.
    """
    H = np.asarray(H, dtype=complex)
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    d = H.shape[0]
    if psi0.size == d:
        rho0 = np.outer(psi0, psi0.conj())
    elif psi0.size == d * d:
        rho0 = psi0.reshape(d, d)
    else:
        raise ValueError(f"psi0 has shape inconsistent with H of dim {d}")

    L = liouvillian(H, list(c_ops))
    vec0 = rho0.reshape(-1)

    def _rhs(t, y):
        return L @ y

    sol = solve_ivp(_rhs, (tlist[0], tlist[-1]), vec0, t_eval=tlist,
                    method=method, rtol=1e-9, atol=1e-11)
    traj = sol.y.T  # (len(tlist), d**2)

    expect = np.empty((len(e_ops), len(tlist)), dtype=float)
    for k, op in enumerate(e_ops):
        op = np.asarray(op, dtype=complex)
        for i in range(len(tlist)):
            rho = traj[i].reshape(d, d)
            expect[k, i] = float(np.real(np.trace(op @ rho)))
    return expect


# ----------------------------------------------------------------------
#  Fenna-Matthews-Olson (FMO) complex
# ----------------------------------------------------------------------
def fmo_model() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray],
                          np.ndarray, list[str]]:
    """Return the 5-site FMO sub-network used in the paper.

    The Hamiltonian describes the electronic exciton dynamics in the
    Fenna-Matthews-Olson pigment-protein complex restricted to the three-site
    sub-network coupled to a sink and to the electronic ground state.  The
    returned matrices are padded to the nearest power of two so that they fit
    in a 3-qubit Hilbert space.

    The Lindblad operators describe pure dephasing on the three sites,
    radiative decay to the ground state and irreversible transfer to the sink.

    Parameters
    ----------

    Return
    ----------
    H : ``ndarray`` of shape ``(8, 8)``\n
        System Hamiltonian in atomic units.
    c_ops : ``list`` of ``ndarray``\n
        Seven collapse operators: three dephasing, three decay and one sink.
    e_ops : ``list`` of ``ndarray``\n
        Five population observables ``|site><site|`` for site 1, 2, 3, the
        sink and the ground state.
    psi0 : ``ndarray``\n
        Initial state ``|site 1>`` encoded in an 8-dimensional Hilbert space.
    labels : ``list`` of ``str``\n
        Human-readable names of the observables in ``e_ops``.
    """
    # Hamiltonian in eV-like units then converted to angular-frequency units
    # (1 eV / hbar) using the value of ``hbar`` consistent with the reference.
    H = np.array([[0, 0, 0, 0, 0],
                  [0, 0.0267, -0.0129, 0.000632, 0],
                  [0, -0.0129, 0.0273, 0.00404, 0],
                  [0, 0.000632, 0.00404, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=complex) * 1.6022 / 1.05457266
    H, n_qubits = _pad_to_power_of_two(H)
    dim = 1 << n_qubits

    # Local basis states.
    def basis(k: int) -> np.ndarray:
        v = np.zeros((dim, 1), dtype=complex)
        v[k, 0] = 1.0
        return v

    ground = basis(0)
    site_1 = basis(1)
    site_2 = basis(2)
    site_3 = basis(3)
    sink = basis(4)

    # Dissipation parameters (in the same units as H).
    a_deph = 3.0e-3
    b_decay = 5.0e-7
    g_sink = 6.28e-3

    L_deph_1 = np.sqrt(a_deph) * (site_1 @ site_1.conj().T)
    L_deph_2 = np.sqrt(a_deph) * (site_2 @ site_2.conj().T)
    L_deph_3 = np.sqrt(a_deph) * (site_3 @ site_3.conj().T)
    L_diss_1 = np.sqrt(b_decay) * (ground @ site_1.conj().T)
    L_diss_2 = np.sqrt(b_decay) * (ground @ site_2.conj().T)
    L_diss_3 = np.sqrt(b_decay) * (ground @ site_3.conj().T)
    L_sink = np.sqrt(g_sink) * (sink @ site_3.conj().T)
    c_ops = [L_deph_1, L_deph_2, L_deph_3,
             L_diss_1, L_diss_2, L_diss_3, L_sink]

    e_ops = [site_1 @ site_1.conj().T,
             site_2 @ site_2.conj().T,
             site_3 @ site_3.conj().T,
             sink @ sink.conj().T,
             ground @ ground.conj().T]
    labels = ["Site 1", "Site 2", "Site 3", "Sink", "Ground"]

    psi0 = site_1.reshape(-1)
    return H, c_ops, e_ops, psi0, labels


# ----------------------------------------------------------------------
#  Transverse-field Ising model with damping
# ----------------------------------------------------------------------
def tfim_model() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray],
                          np.ndarray, list[str]]:
    """Return the two-spin transverse-field Ising model with amplitude damping.

    Following the reference implementation, the Hamiltonian reads
    :math:`H = Z_0 Z_1 - \\tfrac{1}{2}(X_0 + X_1)` (units where :math:`g=1`)
    and each spin undergoes amplitude damping with rate ``g=0.1``.

    Parameters
    ----------

    Return
    ----------
    H : ``ndarray``\n
        System Hamiltonian (4x4).
    c_ops : ``list`` of ``ndarray``\n
        Two amplitude-damping operators.
    e_ops : ``list`` of ``ndarray``\n
        Projectors on ``|00>, |11>`` and ``|01>``.
    psi0 : ``ndarray``\n
        Initial state ``|11>``.
    labels : ``list`` of ``str``\n
        Names of the observables.
    """
    PX = np.array([[0, 1], [1, 0]], dtype=complex)
    PZ = np.array([[1, 0], [0, -1]], dtype=complex)
    PL = np.array([[0, 1], [0, 0]], dtype=complex)  # lowering |1><0|
    PJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
    PJ1 = np.array([[0, 0], [0, 1]], dtype=complex)
    ID = np.eye(2, dtype=complex)

    H = np.kron(PZ, PZ) - 0.5 * (np.kron(PX, ID) + np.kron(ID, PX))
    g_damp = 0.1
    c_ops = [np.sqrt(g_damp) * np.kron(PL, ID),
             np.sqrt(g_damp) * np.kron(ID, PL)]
    e_ops = [np.kron(PJ0, PJ0), np.kron(PJ1, PJ1), np.kron(PJ0, PJ1)]
    labels = ["|00>", "|11>", "|01>"]
    psi0 = np.array([0, 0, 0, 1], dtype=complex)
    return H, c_ops, e_ops, psi0, labels


# ----------------------------------------------------------------------
#  Radical pair model
# ----------------------------------------------------------------------
def rpm_model(k_recombine: float = 0.1,
              k_escape: float = 0.001,
              omega: float = 1.0) -> tuple[
        np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, list[str]]:
    """Return a *normalised* radical pair model for the avian compass.

    The radical pair model (RPM) describes the singlet-triplet spin dynamics
    that is believed to underlie the magnetic compass of migratory birds.
    The full physical model of the paper lives in a 3-qubit spin space and
    uses hyperfine couplings of order :math:`10^7` (rad/s), which require
    dedicated time-step choices.  For convenience we expose here a
    **normalised** version in which the S–T0 mixing frequency ``omega`` is
    set to ``1.0``; this lets the model be used directly with the default
    solver settings (``dt`` of order 0.1–1.0).  Rescale ``omega`` together
    with the rates and the time grid if absolute physical units are needed.

    Parameters
    ----------
    k_recombine : ``float``, optional (default=0.1)\n
        Recombination rate (S/T decay into the product states).
    k_escape : ``float``, optional (default=0.001)\n
        Slow escape rate from every radical-pair state.
    omega : ``float``, optional (default=1.0)\n
        Singlet-triplet mixing frequency.  Set to ``1.0`` for normalised
        units; the physical hyperfine value is around :math:`2\\pi\\cdot10^7`.

    Return
    ----------
    H : ``ndarray``\n
        System Hamiltonian padded to ``8 x 8`` (3 qubits) so it can be used
        directly with :class:`~pyqpanda_alg.LindbladMagnus.lindblad.LindbladMagnusSolver`.
    c_ops : ``list`` of ``ndarray``\n
        Collapse operators for singlet and triplet products plus escape,
        padded to the same ``8 x 8`` shape.
    e_ops : ``list`` of ``ndarray``\n
        Projectors on the singlet and triplet product states.
    psi0 : ``ndarray``\n
        Initial singlet radical pair state, length 8.
    labels : ``list`` of ``str``\n
        Observable names.
    """
    # Two spin-1/2 particles: S, T0, T+, T- (S+T0 share the m=0 subspace,
    # T+ and T- are the polarised triplets).  We label the basis as
    # (|S>, |T0>, |T+>, |T->) and add two product states |PS>, |PT>.
    basis_states = ["S", "T0", "T+", "T-", "PS", "PT"]
    dim_raw = len(basis_states)
    H = np.zeros((dim_raw, dim_raw), dtype=complex)
    # S <-> T0 mixing driven by the hyperfine frequency.
    H[0, 1] = omega
    H[1, 0] = omega

    def basis(k: int, dim: int) -> np.ndarray:
        v = np.zeros((dim, 1), dtype=complex)
        v[k, 0] = 1.0
        return v

    PS = basis(4, dim_raw)
    PT = basis(5, dim_raw)
    S = basis(0, dim_raw)
    T0 = basis(1, dim_raw)
    Tp = basis(2, dim_raw)
    Tm = basis(3, dim_raw)

    # Collapse operators: singlet / triplet recombination into the products
    # and a slow escape from every radical-pair state.
    c_ops = [
        np.sqrt(k_recombine) * (PS @ S.conj().T),
        np.sqrt(k_recombine) * (PT @ (T0 + Tp + Tm).conj().T / np.sqrt(3)),
    ]
    for st in (S, T0, Tp, Tm):
        c_ops.append(np.sqrt(k_escape) * (PS @ st.conj().T))

    e_ops = [PS @ PS.conj().T, PT @ PT.conj().T]
    labels = ["Singlet product", "Triplet product"]
    psi0 = S.reshape(-1)

    # Pad everything to the nearest power of two so that the model can be
    # used directly with the variational solver (whose Hilbert space must be
    # a tensor product of qubits).
    H, n_qubits = _pad_to_power_of_two(H)
    dim = 1 << n_qubits
    c_ops = [_pad_to_power_of_two(op)[0] for op in c_ops]
    e_ops = [_pad_to_power_of_two(op)[0] for op in e_ops]
    psi_full = np.zeros(dim, dtype=complex)
    psi_full[:psi0.size] = psi0
    psi0 = psi_full
    return H, c_ops, e_ops, psi0, labels
