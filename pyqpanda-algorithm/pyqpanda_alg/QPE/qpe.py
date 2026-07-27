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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, measure
import numpy as np

from ..plugin import *


class QPE:
    """
    Quantum Phase Estimation (QPE).

    Estimates the phase :math:`\\varphi` of the eigenvalue
    :math:`e^{2\\pi i \\varphi}` of a unitary operator :math:`U`,
    given an eigenstate :math:`|\\psi\\rangle` such that
    :math:`U|\\psi\\rangle = e^{2\\pi i \\varphi}|\\psi\\rangle`.

    Parameters
    ----------
    unitary : callable
        Function ``f(target_qubits) -> QCircuit`` that returns the
        unitary circuit :math:`U` acting on the target qubits.
        The circuit should not include measurement gates.
    n_count : int, optional
        Number of counting (estimation) qubits.  Larger values give
        higher precision.  Default 4.
    n_target : int, optional
        Number of target (eigenstate) qubits.  Default 1.
    state_prep : callable, optional
        Function ``f(target_qubits) -> QCircuit`` that prepares the
        eigenstate :math:`|\\psi\\rangle` on the target qubits.
        If not given, the target qubits start in :math:`|0\\rangle`.

    Attributes
    ----------
    n_count : int
        Number of counting qubits.
    n_target : int
        Number of target qubits.
    n_qubits : int
        Total number of qubits (= n_count + n_target).

    Methods
    -------
    build_circuit()
        Build the full QPE circuit (no measurement).

    run(shots=-1)
        Run QPE and return the estimated phase :math:`\\varphi`
        as a float in :math:`[0, 1)`.

    run_dict(shots=-1)
        Run QPE and return a probability dictionary over the
        counting-qubit bitstrings.

    References
    ----------
    .. [1] Kitaev A Y. Quantum measurements and the Abelian
           Stabilizer Problem. 1995.
           https://arxiv.org/abs/quant-ph/9511026
    .. [2] Nielsen M A, Chuang I L. Quantum Computation and
           Quantum Information. Cambridge University Press, 2010.

    Examples
    --------
    Estimate the phase of the Pauli-Z operator:

    >>> from pyqpanda3.core import Z
    >>> from pyqpanda_alg.QPE import QPE
    >>> def u_z(tq):
    ...     cir = QCircuit()
    ...     cir << Z(tq[0])
    ...     return cir
    >>> qpe = QPE(unitary=u_z, n_count=4, n_target=1)
    >>> phi = qpe.run()
    >>> print(f"Phase estimate: {phi:.4f}")
    """

    # ── constructor ──────────────────────────────────────────

    def __init__(self,
                 unitary,
                 n_count=4,
                 n_target=1,
                 state_prep=None):
        if unitary is None:
            raise ValueError("unitary must be provided")
        self._unitary = unitary
        self.n_count = n_count
        self.n_target = n_target
        self.n_qubits = n_count + n_target
        self._state_prep = state_prep
        self._circuit = None  # cached circuit

    # ── helper: controlled-U^(2^k) ───────────────────────────

    def _controlled_u_power(self, q_count, q_targets, power):
        """
        Build controlled-U^(2^power) acting on target qubits,
        controlled by a single counting qubit.

        Parameters
        ----------
        q_count : Qubit
            The control (counting) qubit.
        q_targets : list[Qubit]
            The target qubits.
        power : int
            The exponent k, so that the operation is U^(2^k).

        Returns
        -------
        QCircuit
        """
        repetitions = 1 << power          # 2^power
        if repetitions == 0:
            repetitions = 1

        cir = QCircuit()
        base_u = self._unitary(q_targets)
        # Repeat U 2^power times, each controlled by q_count
        for _ in range(repetitions):
            cir << base_u.control([q_count])
        return cir

    # ── build full QPE circuit ───────────────────────────────

    def build_circuit(self):
        """
        Build the full QPE circuit (no measurement).

        The circuit consists of:
            1. Hadamard gates on all counting qubits.
            2. Controlled-U^(2^k) gates for k = 0, ..., n_count-1.
            3. Inverse QFT on the counting qubits.

        Returns
        -------
        QCircuit
        """
        q_all = QProg(self.n_qubits).qubits()
        q_count = q_all[:self.n_count]      # counting register
        q_target = q_all[self.n_count:]      # target register

        cir = QCircuit()

        # Step 0: prepare eigenstate (if provided)
        if self._state_prep is not None:
            cir << self._state_prep(q_target)

        # Step 1: Hadamard on all counting qubits
        cir << hadamard_circuit(q_count)

        # Step 2: controlled-U^(2^k) for each counting qubit
        # q_count[0] controls U^1, q_count[1] controls U^2, ...
        for k in range(self.n_count):
            cir << self._controlled_u_power(
                q_count[k], q_target, k
            )

        # Step 3: inverse QFT on counting qubits
        cir << QFT(q_count).dagger()

        self._circuit = cir
        return cir

    # ── run and return phase ─────────────────────────────────

    def run(self, shots=-1):
        """
        Run QPE and return the estimated phase as a float.

        Parameters
        ----------
        shots : int, optional
            Number of measurement shots.  ``-1`` (default) uses the
            state-vector simulator.  Positive values use sampling.

        Returns
        -------
        float
            Phase estimate :math:`\\varphi \\in [0, 1)`.
        """
        qvm = CPUQVM()
        cir = self.build_circuit()
        prog = QProg(self.n_qubits)
        q_all = prog.qubits()
        q_count = q_all[:self.n_count]

        prog << cir

        if shots == -1:
            qvm.run(prog, shots=1)
            prob_dict = qvm.result().get_prob_dict(q_count)
            prob_dict = parse_quantum_result_dict(
                prob_dict, q_count, select_max=1
            )
            best_bitstring = list(prob_dict.keys())[0]
        elif shots > 0:
            prog << measure_all(q_count, q_count)
            qvm.run(prog, shots=shots)
            counts = qvm.result().get_counts()
            best_bitstring = max(counts, key=counts.get)
        else:
            raise ValueError(f"Invalid shots: {shots}")

        # Convert bitstring to phase φ = integer / 2^n_count
        phase_int = int(best_bitstring, 2)
        phi = phase_int / (1 << self.n_count)
        return phi

    # ── run and return raw probability dict ──────────────────

    def run_dict(self, shots=-1):
        """
        Run QPE and return the full probability distribution over
        the counting-register bitstrings.

        Parameters
        ----------
        shots : int, optional
            ``-1`` for state-vector, positive for sampling.

        Returns
        -------
        dict[str, float]
            Mapping from n_count-bit binary strings to probabilities.
        """
        qvm = CPUQVM()
        cir = self.build_circuit()
        prog = QProg(self.n_qubits)
        q_all = prog.qubits()
        q_count = q_all[:self.n_count]

        prog << cir

        if shots == -1:
            qvm.run(prog, shots=1)
            prob_dict = qvm.result().get_prob_dict(q_count)
            return parse_quantum_result_dict(
                prob_dict, q_count, select_max=-1
            )
        elif shots > 0:
            prog << measure_all(q_count, q_count)
            qvm.run(prog, shots=shots)
            counts = qvm.result().get_counts()
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}
        else:
            raise ValueError(f"Invalid shots: {shots}")
