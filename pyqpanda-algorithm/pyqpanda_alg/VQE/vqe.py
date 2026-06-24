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

r"""
Core Variational Quantum Eigensolver (VQE) implementation.

The :class:`VQE` class finds the ground-state energy of a qubit Hamiltonian
:math:`H` by minimizing the energy expectation

.. math::
    E(\vec\theta) = \bra{0} U^\dagger(\vec\theta)\, H\, U(\vec\theta) \ket{0}

over the parameters :math:`\vec\theta` of an ansatz circuit :math:`U(\vec\theta)`,
using a classical optimizer. Gradients, when required, are obtained with the
exact **parameter-shift rule**

.. math::
    \frac{\partial E}{\partial\theta_k} = \frac{1}{2}\big[
        E(\theta_k + \pi/2) - E(\theta_k - \pi/2)\big],

which is hardware-measurable and exact for the gate set used here.

Low-lying **excited states** can be estimated with
:meth:`VQE.excited_state_energies`, which builds a small Hamiltonian matrix in a
subspace spanned by the ground state and a few electron-repulsion operators
(Quantum Subspace Expansion, McClean 2017).
"""

import numpy as np
from scipy.optimize import minimize
from pyqpanda3.core import CPUQVM, QCircuit, QProg, expval_pauli_operator
from pyqpanda3.hamiltonian import PauliOperator

from . import ansatz as ansatz_mod
from . import hamiltonian as ham_mod


# scipy.optimize.minimize methods that actually consume a gradient (jac).
# Used to validate the ``gradient`` option of :meth:`VQE.run`.
_GRADIENT_METHODS = {
    "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP",
    "dogleg", "trust-ncg", "trust-exact", "trust-krylov", "trust-constr",
}


class VQE:
    r"""Variational Quantum Eigensolver.

    Given a Hamiltonian (as a :class:`~pyqpanda3.hamiltonian.PauliOperator`) and
    an ansatz, :meth:`run` minimizes the variational energy and returns the
    ground-state energy estimate together with the optimal parameters and a
    convergence trace.

    .. note::

        Energies are evaluated as **exact state-vector expectation values**
        :math:`\langle\psi|H|\psi\rangle` on the CPU simulator (no shot noise).
        Gradients, when requested, are obtained with the hardware-measurable
        **parameter-shift rule** rather than autodiff, so the same code path
        would run on real quantum hardware.

    .. note::

        The result quality is bounded by the *expressivity* of the ansatz: a
        single hardware-efficient layer is enough for ``H2`` but under-fits
        larger systems (e.g. a 3-site Ising chain needs ~3 layers). Increase
        ``layers`` or switch to a problem-tailored ansatz if VQE stalls above
        the exact energy :func:`~pyqpanda_alg.VQE.hamiltonian.exact_ground_energy`.

    Parameters
        hamiltonian : ``PauliOperator``\n
            The qubit Hamiltonian whose ground state is sought. It can be built
            with :mod:`pyqpanda_alg.VQE.hamiltonian` or supplied directly.
        ansatz : ``callable``, optional\n
            Function ``(n_qubits, params) -> QCircuit`` producing the trial
            circuit. If ``None``, a single-layer hardware-efficient ansatz is
            used (see :func:`~pyqpanda_alg.VQE.ansatz.hardware_efficient_ansatz`).
        n_qubits : ``int``, optional\n
            Number of qubits. If ``None`` it is inferred from ``hamiltonian``
            (``max_qbit_idx + 1``).
        n_params : ``int``, optional\n
            Number of variational parameters expected by ``ansatz``. Required
            when a custom ``ansatz`` is supplied (there is no reliable way to
            infer it automatically); ignored for the default ansatz.

    Attributes
        energy_history : ``list`` of ``float``\n
            Energy value recorded after every optimizer step (useful for
            plotting convergence).
        optimal_params : ``numpy.ndarray``\n
            Parameters at the end of the optimization.
        optimal_energy : ``float``\n
            Energy value at ``optimal_params``.
        circuit_evals : ``int``\n
            Total number of energy evaluations performed.

    Methods
        run : Run the variational optimization and return the ground-state energy.

        expectation : Evaluate the energy for a given parameter vector.

        get_statevector : Return the state vector produced by the ansatz.

        excited_state_energies : Estimate low-lying excited-state energies (QSE).

    Reference
        [1] PERUZZO A, MCCLEAN J, SHADBOLT P, et al. A variational eigenvalue
        solver on a photonic quantum processor[J]. Nature Communications, 2014,
        5: 4213. DOI: 10.1038/ncomms5213.\n
        [2] MITTAL R, BRACCARDI M, HOGAN P, et al. (Qiskit contributors)
        Real-time quantum dynamics and VQE. Qiskit Textbook.
    """

    def __init__(self, hamiltonian, ansatz=None, n_qubits=None, n_params=None):
        if not isinstance(hamiltonian, PauliOperator):
            raise TypeError("hamiltonian must be a pyqpanda3 PauliOperator")
        self.hamiltonian = hamiltonian
        if n_qubits is None:
            n_qubits = max(hamiltonian.max_qbit_idx() + 1, 1)
        self.n_qubits = n_qubits

        # default ansatz: single-layer hardware-efficient (RY,RZ)+CNOT
        if ansatz is None:
            def _default(n, params):
                return ansatz_mod.hardware_efficient_ansatz(
                    n, params, layers=1, rotations=("RY", "RZ"), entangler="CNOT")
            self.ansatz = _default
            self.n_params = ansatz_mod.hardware_efficient_n_params(
                n_qubits, layers=1, rotations=("RY", "RZ"), entangler="CNOT")
        else:
            if n_params is None:
                raise ValueError(
                    "n_params must be provided when a custom ansatz is given "
                    "(the parameter count cannot be inferred reliably).")
            self.ansatz = ansatz
            self.n_params = n_params

        self._qvm = CPUQVM()
        self.energy_history = []
        self.optimal_params = None
        self.optimal_energy = None
        self.circuit_evals = 0

    # ------------------------------------------------------------------ #
    #  energy evaluation
    # ------------------------------------------------------------------ #
    def _build_prog(self, params):
        """Build the full QProg: ansatz circuit applied to |0>."""
        prog = QProg(self.n_qubits)
        prog << self.ansatz(self.n_qubits, np.asarray(params))
        return prog

    def expectation(self, params):
        r"""Evaluate the energy expectation value :math:`E(\vec\theta)`.

        Parameters
            params : ``array-like``\n
                Variational parameters.

        Return
            ``float``\n

        Examples
            >>> from pyqpanda_alg.VQE import vqe, hamiltonian
            >>> solver = vqe.VQE(hamiltonian.h2_hamiltonian())
            >>> e = solver.expectation([0.0, 0.0, 0.0, 0.0])  # |00> reference
            >>> print(round(e, 4))
                -1.0637
        """
        return self._eval_energy(params, count=True)

    def _eval_energy(self, params, count=True):
        """Core energy evaluation: build the circuit, run it, measure <H>.

        ``count`` controls whether the call is added to ``circuit_evals``
        (disabled for the redundant re-evaluations avoided in :meth:`run`).
        """
        prog = self._build_prog(params)
        self._qvm.run(prog, self.n_qubits)
        val = expval_pauli_operator(prog, self.hamiltonian)
        if count:
            self.circuit_evals += 1
        return float(val.real)

    def param_shift_gradient(self, params):
        r"""Compute the energy gradient with the parameter-shift rule.

        .. math::
            \partial E/\partial\theta_k =
                \tfrac{1}{2}[E(\theta_k{+}\pi/2) - E(\theta_k{-}\pi/2)].

        Parameters
            params : ``array-like``\n

        Return
            ``numpy.ndarray``\n
                Gradient vector of the same length as ``params``.
        """
        params = np.asarray(params, dtype=float)
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for k in range(len(params)):
            plus = params.copy(); plus[k] += shift
            minus = params.copy(); minus[k] -= shift
            grad[k] = 0.5 * (self._eval_energy(plus)
                             - self._eval_energy(minus))
        return grad

    def get_statevector(self, params):
        r"""Return the state vector :math:`U(\vec\theta)\ket{0}` as a numpy array."""
        prog = self._build_prog(params)
        self._qvm.run(prog, self.n_qubits)
        return np.array(self._qvm.result().get_state_vector(), dtype=complex)

    # ------------------------------------------------------------------ #
    #  optimization
    # ------------------------------------------------------------------ #
    def run(self, initial_para=None, optimizer="COBYLA",
            max_iter=200, tol=1e-6, gradient=False, verbose=False):
        r"""Run the variational optimization.

        Parameters
            initial_para : ``array-like``, optional\n
                Starting parameters. If ``None``, a uniform random vector in
                :math:`[0, 2\pi)` is drawn.
            optimizer : ``str``, optional\n
                Any ``scipy.optimize.minimize`` method name (e.g. ``"COBYLA"``,
                ``"Powell"``, ``"Nelder-Mead"``, ``"L-BFGS-B"``). Default is
                ``"COBYLA"``.
            max_iter : ``int``, optional\n
                Maximum number of optimizer iterations. Default is 200.
            tol : ``float``, optional\n
                Optimizer tolerance. Default is ``1e-6``.
            gradient : ``bool``, optional\n
                If ``True``, supply the parameter-shift gradient to the optimizer
                (only effective for gradient-aware methods such as
                ``"L-BFGS-B"``). Default is ``False``.
            verbose : ``bool``, optional\n
                Print the energy every 25 iterations. Default is ``False``.

        Return
            ``tuple`` ``(energy, params, history)``\n
                - ``energy`` : optimal (ground-state) energy estimate.\n
                - ``params`` : optimal parameter vector.\n
                - ``history`` : list of energies recorded each iteration.

        Examples
            >>> from pyqpanda_alg.VQE import vqe, hamiltonian
            >>> solver = vqe.VQE(hamiltonian.h2_hamiltonian())
            >>> energy, params, _ = solver.run(optimizer='L-BFGS-B',
            ...                                gradient=True, max_iter=200)
            >>> print('VQE energy :', round(energy, 5))
            >>> print('exact      :', round(hamiltonian.exact_ground_energy(
            ...     hamiltonian.h2_hamiltonian()), 5))
                VQE energy : -1.85727
                exact      : -1.85727
        """
        if initial_para is None:
            initial_para = np.random.uniform(0.0, 2 * np.pi, self.n_params)
        else:
            initial_para = np.asarray(initial_para, dtype=float)
            if len(initial_para) != self.n_params:
                raise ValueError(
                    "initial_para has length %d but the ansatz expects %d "
                    "parameter(s)." % (len(initial_para), self.n_params))

        # validate gradient/optimizer compatibility up-front
        if gradient and optimizer not in _GRADIENT_METHODS:
            import warnings
            warnings.warn(
                "optimizer %r is gradient-free; the parameter-shift gradient "
                "will be ignored. Use a gradient-aware method (e.g. 'L-BFGS-B', "
                "'CG', 'SLSQP') to make use of it." % optimizer, stacklevel=2)

        self.circuit_evals = 0
        self.energy_history = []

        step = {"i": 0}
        last = {"e": None}  # cache the most recent cost value for the callback

        def _cost(params):
            e = self._eval_energy(params, count=True)
            last["e"] = e
            return e

        def _callback(xk):
            # reuse the energy already computed at this iterate -- do NOT
            # re-evaluate, which would double the circuit cost and inflate
            # ``circuit_evals``.
            e = last["e"]
            if e is None:
                e = self._eval_energy(xk, count=True)
            self.energy_history.append(e)
            step["i"] += 1
            if verbose and (step["i"] % 25 == 0 or step["i"] == 1):
                print("  iter %4d   energy = %.8f" % (step["i"], e))

        # only hand the gradient to optimizers that actually use it, otherwise
        # scipy emits its own (confusing) "does not use gradient" warning.
        jac = self.param_shift_gradient if (gradient and optimizer in _GRADIENT_METHODS) else None
        res = minimize(_cost, initial_para, method=optimizer,
                       jac=jac, tol=tol,
                       options={"maxiter": max_iter, "disp": False},
                       callback=_callback)

        self.optimal_params = res.x
        self.optimal_energy = float(res.fun.real)
        if not self.energy_history:
            self.energy_history.append(self.optimal_energy)
        return self.optimal_energy, self.optimal_params, self.energy_history

    # ------------------------------------------------------------------ #
    #  excited states (Quantum Subspace Expansion)
    # ------------------------------------------------------------------ #
    def excited_state_energies(self, params, n_states=3,
                               excitations=None, return_vectors=False):
        r"""Estimate low-lying excited-state energies via Quantum Subspace
        Expansion (QSE).

        Given a (ground-state-like) reference :math:`\ket{\psi}` produced by
        ``params``, a small subspace is spanned by

        .. math::
            \{\ket{\psi},\; a^\dagger_i a_j \ket{\psi}\}

        (the reference plus single fermionic excitations). The Hamiltonian and
        overlap matrices in this subspace are evaluated from quantum
        expectation values and diagonalized classically, yielding a set of
        energy eigenvalues that includes the ground state and the lowest
        excitations.

        Parameters
            params : ``array-like``\n
                Reference parameters (typically the VQE ground-state optimum).
            n_states : ``int``, optional\n
                Number of lowest eigenvalues to return. Default is 3.
            excitations : ``list`` of ``(int, int)``, optional\n
                Fermionic excitation pairs ``(i, j)`` used to build the subspace.
                If ``None`` all unique pairs are used.
            return_vectors : ``bool``, optional\n
                Also return the subspace eigenvectors. Default is ``False``.

        Return
            ``numpy.ndarray``\n
                ``n_states`` lowest energy eigenvalues (ascending). If
                ``return_vectors`` is True, returns ``(energies, vectors)``.

        References
            [1] McCLEAN J R, KIMCHI-SCHWARTZ M E, CARTER J, et al. Hybrid
            quantum-classical hierarchy for mitigation of decoherence and a
            faithful representation of electronic excited states. Physical
            Review A, 2017, 95(4): 042308.
        """
        n = self.n_qubits
        if excitations is None:
            excitations = [(i, j) for i in range(n) for j in range(i + 1, n)]
        ops = [PauliOperator({"": 1.0})]  # identity -> reference state
        for (i, j) in excitations:
            ops.append(ham_mod.jw_create(n, i) * ham_mod.jw_annihilate(n, j))

        # Evaluate the ansatz circuit ONCE and reuse the state vector for every
        # matrix element: <psi|O|psi> = psi^dagger . O . psi. This avoids
        # rebuilding/rerunning the circuit for each of the dim**2 elements.
        psi = self.get_statevector(params)
        h_mat = _full_matrix(self.hamiltonian, n)
        op_mats = [_full_matrix(op, n) for op in ops]

        def _expect(op_mat):
            return complex(np.vdot(psi, op_mat @ psi))

        dim = len(ops)
        H_sub = np.zeros((dim, dim), dtype=complex)
        S_sub = np.zeros((dim, dim), dtype=complex)
        for a in range(dim):
            for b in range(dim):
                # Hamiltonian matrix element <psi| A†_a H A_b |psi>.
                # A_0 = I, so the (0,0) block reduces to the bare <psi|H|psi>.
                adag_h_b = op_mats[a].conj().T @ h_mat @ op_mats[b]
                H_sub[a, b] = _expect(adag_h_b)
                # overlap matrix element <psi| A†_a A_b |psi>
                S_sub[a, b] = _expect(op_mats[a].conj().T @ op_mats[b])

        # solve the generalized eigenvalue problem H c = E S c
        S_sub = 0.5 * (S_sub + S_sub.conj().T)  # symmetrize against numerical noise
        H_sub = 0.5 * (H_sub + H_sub.conj().T)
        # regularize S for numerical stability
        s_evals, s_evecs = np.linalg.eigh(S_sub)
        keep = s_evals > 1e-9
        sqrt_inv = (s_evecs[:, keep]
                    * (1.0 / np.sqrt(s_evals[keep])))
        Hp = sqrt_inv.T.conj() @ H_sub @ sqrt_inv
        eigvals, eigvecs_p = np.linalg.eigh(Hp)
        order = np.argsort(eigvals.real)[:n_states]
        energies = eigvals[order].real
        if return_vectors:
            vectors = s_evecs[:, keep] @ sqrt_inv @ eigvecs_p[:, order]
            return energies, vectors
        return energies


def _full_matrix(op, n_qubits):
    """Return ``op`` as a dense ``2**n_qubits x 2**n_qubits`` matrix.

    :meth:`PauliOperator.matrix` only spans the qubits an operator actually
    touches (size ``2**(max_qbit_idx+1)``), so single-qubit / identity terms
    must be padded with identities on the unused high-index qubits to live in
    the full ``n_qubits`` Hilbert space. The padding uses
    ``kron(I, op_matrix)`` which matches pyqpanda3's qubit ordering.
    """
    mat = np.array(op.matrix(), dtype=complex)
    pad = n_qubits - (op.max_qbit_idx() + 1)
    if pad > 0:
        mat = np.kron(np.eye(2 ** pad, dtype=complex), mat)
    return mat
