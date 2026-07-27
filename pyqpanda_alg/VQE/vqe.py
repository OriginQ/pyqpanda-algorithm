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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, RY, RZ, CNOT, H, RX, CZ, measure
from pyqpanda3.hamiltonian import PauliOperator, Hamiltonian
import numpy as np
from scipy.optimize import minimize

from ..QAOA import spsa
from ..plugin import *


# ─────────────────────────────────────────────────────────────
#  Built-in Ansatz Circuits
# ─────────────────────────────────────────────────────────────

def hardware_efficient_circuit(qlist, params, depth=2, entangle='linear'):
    """
    Build a hardware-efficient ansatz circuit.

    Each layer consists of:
        1. RY and RZ rotations on every qubit
        2. CNOT entangling gates between adjacent qubits

    Parameters
    ----------
    qlist : list
        Qubit list.
    params : array-like
        Flattened parameter vector with length ``2 * n_qubits * depth``.
    depth : int, optional
        Number of variational layers (default 2).
    entangle : str, optional
        Entangling scheme: ``'linear'`` (default) or ``'full'``.

    Returns
    -------
    QCircuit
        Hardware-efficient parameterized circuit.

    Examples
    --------
    >>> from pyqpanda_alg.VQE.vqe import hardware_efficient_circuit
    >>> from pyqpanda3.core import QProg
    >>> q = QProg(3).qubits()
    >>> params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    >>> cir = hardware_efficient_circuit(q, params, depth=2)
    """
    n = len(qlist)
    circuit = QCircuit()
    param_idx = 0

    for _ in range(depth):
        # Rotation layer: RY + RZ on each qubit
        for j in range(n):
            circuit << RY(qlist[j], params[param_idx])
            param_idx += 1
        for j in range(n):
            circuit << RZ(qlist[j], params[param_idx])
            param_idx += 1

        # Entanglement layer
        if entangle == 'linear':
            for j in range(n - 1):
                circuit << CNOT(qlist[j], qlist[j + 1])
            # Close the ring
            if n > 2:
                circuit << CNOT(qlist[-1], qlist[0])
        elif entangle == 'full':
            for j in range(n):
                for k in range(j + 1, n):
                    circuit << CNOT(qlist[j], qlist[k])
        else:
            raise ValueError(f"Unknown entangle scheme: {entangle}")

    return circuit


# ─────────────────────────────────────────────────────────────
#  VQE Main Class
# ─────────────────────────────────────────────────────────────

class VQE:
    """
    Variational Quantum Eigensolver (VQE).

    This class implements a hybrid quantum-classical algorithm to find
    the ground-state energy of a given Hamiltonian.  It uses a
    parameterized quantum circuit (ansatz) to prepare trial states,
    measures the expectation value of the Hamiltonian, and iteratively
    improves the parameters via a classical optimizer.

    Parameters
    ----------
    hamiltonian : PauliOperator or Hamiltonian
        The Hamiltonian whose ground-state energy is sought.
    n_qubits : int, optional
        Number of qubits.  If not given it is inferred from the
        Hamiltonian.
    ansatz_circuit : callable, optional
        Function ``f(qlist, params) -> QCircuit`` that builds the
        parameterized trial-state circuit.  Defaults to
        :func:`hardware_efficient_circuit` with depth=2.
    ansatz_depth : int, optional
        Depth (number of layers) for the built-in hardware-efficient
        ansatz.  Ignored when a custom ``ansatz_circuit`` is supplied.
        Default 2.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    n_params : int
        Number of variational parameters.
    energy_history : list[float]
        Energy value at each optimizer iteration.
    optimal_params : ndarray
        Best parameters found by the optimizer.

    Methods
    -------
    run(optimizer='SLSQP', maxiter=200, initial_params=None, shots=-1, **opt_kwargs)
        Run the VQE optimization and return the ground-state energy.

    calculate_energy(params, shots=-1)
        Run the ansatz circuit with given parameters and compute
        the expectation value ⟨ψ|H|ψ⟩.

    References
    ----------
    .. [1] Peruzzo A, McClean J, Shadbolt P, et al.
           A variational eigenvalue solver on a photonic quantum processor.
           Nature Communications, 2014, 5: 4213.
           https://doi.org/10.1038/ncomms5213

    Examples
    --------
    Construct a simple Hamiltonian and find its ground-state energy:

    >>> from pyqpanda3.hamiltonian import PauliOperator
    >>> from pyqpanda_alg.VQE import VQE
    >>> import numpy as np
    >>> H = (PauliOperator({"Z0": 1.0})
    ...      + PauliOperator({"X0": 0.5}))
    >>> solver = VQE(H, n_qubits=1, ansatz_depth=2)
    >>> energy = solver.run(optimizer='SLSQP', maxiter=100)
    >>> print(f"Ground-state energy: {energy:.6f}")
    """

    # ── constructor ──────────────────────────────────────────

    def __init__(self,
                 hamiltonian,
                 n_qubits=None,
                 ansatz_circuit=None,
                 ansatz_depth=2):
        # --- Hamiltonian ---
        if isinstance(hamiltonian, Hamiltonian):
            self._operator = hamiltonian.pauli_operator()
        elif isinstance(hamiltonian, PauliOperator):
            self._operator = hamiltonian
        else:
            raise TypeError(
                "hamiltonian must be a PauliOperator or Hamiltonian, "
                f"got {type(hamiltonian)}"
            )

        # --- Number of qubits ---
        if n_qubits is None:
            qubits = set()
            for term in self._operator.terms():
                for p in term.paulis():
                    if p.is_Z() or p.is_X() or p.is_Y():
                        qubits.add(p.qbit())
            self.n_qubits = max(qubits) + 1 if qubits else 1
        else:
            self.n_qubits = n_qubits

        # --- Ansatz ---
        if ansatz_circuit is not None:
            self._ansatz = ansatz_circuit
            self.ansatz_depth = None  # user-defined
        else:
            self._ansatz = lambda q, p: hardware_efficient_circuit(
                q, p, depth=ansatz_depth
            )
            self.ansatz_depth = ansatz_depth

        # Number of parameters: 2 * n_qubits * depth
        if self.ansatz_depth is not None:
            self.n_params = 2 * self.n_qubits * self.ansatz_depth
        else:
            self.n_params = None  # will be inferred later

        # --- State ---
        self.energy_history = []
        self.optimal_params = None
        self._circuit_iter = 0

    # ── core: run circuit → energy ───────────────────────────

    def calculate_energy(self, params, shots=-1):
        """
        Run the ansatz with *params* and return ⟨ψ(θ)|H|ψ(θ)⟩.

        Parameters
        ----------
        params : array-like
            Variational parameters.
        shots : int, optional
            Number of measurement shots.  ``-1`` uses the state-vector
            simulator (theoretical / infinite-shots result).  Default
            is ``-1``.

        Returns
        -------
        float
            Expectation value of the Hamiltonian.
        """
        qvm = CPUQVM()
        prog = QProg(self.n_qubits)
        qlist = prog.qubits()

        # Build and insert ansatz
        if self.n_params is None:
            # First call: infer parameter count from the user circuit
            self.n_params = len(params)
        cir = self._ansatz(qlist, params)
        prog << cir

        # --- State-vector (theoretical) mode ---
        if shots == -1:
            qvm.run(prog, shots=1)
            prob_dict = qvm.result().get_prob_dict(qlist)
            prob_dict = parse_quantum_result_dict(
                prob_dict, qlist, select_max=-1
            )

            energy = 0.0
            for bitstring, prob in prob_dict.items():
                # bitstring is e.g. '010'  (lowest qubit on the right)
                config = [int(c) for c in bitstring[::-1]]
                e_state = self._compute_energy_of_config(config)
                energy += prob * e_state
            return energy

        # --- Sampling mode ---
        elif shots > 0:
            prog << measure_all(qlist, qlist)
            qvm.run(prog, shots=shots)
            counts = qvm.result().get_counts()

            energy = 0.0
            total = sum(counts.values())
            for bitstring, cnt in counts.items():
                config = [int(c) for c in bitstring[::-1]]
                e_state = self._compute_energy_of_config(config)
                energy += (cnt / total) * e_state
            return energy

        else:
            raise ValueError(f"Invalid shots number: {shots}")

    def _compute_energy_of_config(self, config):
        """
        Compute the energy of a single computational-basis
        configuration.

        Parameters
        ----------
        config : list[int]
            Binary list, e.g. ``[1, 0, 1]`` for q0=1, q1=0, q2=1.

        Returns
        -------
        float
        """
        energy = 0.0
        for term in self._operator.terms():
            coef = term.coef()
            coef_real = coef.real if isinstance(coef, complex) else coef

            term_contrib = coef_real
            for pauli in term.paulis():
                idx = pauli.qbit()
                p_char = pauli.pauli_char()

                if idx >= len(config):
                    # qubit index outside config — treat as 0
                    continue

                if p_char == 'I':
                    continue
                elif p_char == 'Z':
                    # Z|b⟩ = (-1)^b |b⟩
                    if config[idx] == 1:
                        term_contrib *= -1
                elif p_char == 'X':
                    # X|0⟩ = |1⟩, X|1⟩ = |0⟩ — off-diagonal,
                    # expectation is zero for computational basis
                    term_contrib = 0
                    break
                elif p_char == 'Y':
                    # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩ — off-diagonal
                    term_contrib = 0
                    break
                else:
                    term_contrib = 0
                    break

            energy += term_contrib

        return energy

    # ── loss function for the optimizer ──────────────────────

    def _loss_function(self, params, shots=-1):
        """
        Loss function called by the classical optimizer.

        Parameters
        ----------
        params : ndarray
            Current parameter vector.
        shots : int

        Returns
        -------
        float
            Energy expectation value.
        """
        energy = self.calculate_energy(params, shots=shots)
        self.energy_history.append(energy)
        self._circuit_iter += 1
        return energy

    # ── main entry point ─────────────────────────────────────

    def run(self,
            optimizer='SLSQP',
            maxiter=200,
            initial_params=None,
            shots=-1,
            **opt_kwargs):
        """
        Run the VQE optimization.

        Parameters
        ----------
        optimizer : str, optional
            Classical optimizer.  ``'SLSQP'`` (default), ``'SPSA'``,
            ``'COBYLA'``, or any method accepted by
            :func:`scipy.optimize.minimize`.
        maxiter : int, optional
            Maximum number of optimizer iterations (default 200).
        initial_params : ndarray, optional
            Initial parameter vector.  If not given, random values
            in ``[0, 2π)`` are used.
        shots : int, optional
            Measurement shots.  ``-1`` uses state-vector simulation
            (default).
        **opt_kwargs
            Additional keyword arguments passed to the optimizer.

        Returns
        -------
        float
            Ground-state energy estimate.

        Examples
        --------
        >>> from pyqpanda3.hamiltonian import PauliOperator
        >>> from pyqpanda_alg.VQE import VQE
        >>> H = PauliOperator({"Z0": 1.0}) + PauliOperator({"Z1": 1.0})
        >>> v = VQE(H, n_qubits=2, ansatz_depth=1)
        >>> e = v.run(optimizer='SLSQP', maxiter=100)
        >>> print(e)
        """
        # --- Initial parameters ---
        if initial_params is None:
            if self.n_params is None:
                raise ValueError(
                    "n_params is unknown; please provide initial_params "
                    "or use the built-in ansatz."
                )
            np.random.seed(42)
            initial_params = np.random.random(self.n_params) * 2 * np.pi

        initial_params = np.asarray(initial_params, dtype=float)

        if self.n_params is None:
            self.n_params = len(initial_params)

        # --- Reset state ---
        self.energy_history = []
        self._circuit_iter = 0

        # --- Set default options ---
        if 'options' not in opt_kwargs:
            opt_kwargs['options'] = {'maxiter': maxiter}
        else:
            opt_kwargs['options'].setdefault('maxiter', maxiter)

        # --- Optimize ---
        if optimizer.upper() == 'SPSA':
            final = spsa.spsa_minimize(
                lambda p: self._loss_function(p, shots=shots),
                initial_params,
                **opt_kwargs
            )
            self.optimal_params = final
        else:
            result = minimize(
                lambda p: self._loss_function(p, shots=shots),
                initial_params,
                method=optimizer,
                **opt_kwargs
            )
            self.optimal_params = result.x

        # --- Final energy ---
        final_energy = self.calculate_energy(self.optimal_params, shots=shots)
        self.energy_history.append(final_energy)

        return final_energy
