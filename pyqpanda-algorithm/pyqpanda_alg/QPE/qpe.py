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

from pyqpanda3.core import CPUQVM, QCircuit, QProg

from ..plugin import QFT, hadamard_circuit, measure_all
from .readout import (PhaseEstimationResult, decode_phase, phase_to_eigenvalue,
                      normalize_distribution, positive_integer)

# This engine retains the API and sampled execution of the earlier QPE
# contribution by 1446170411, OriginQ/pyqpanda-algorithm PR #34.
# The optional efficient-power hook is preserved from tokenjunkielabs PR #81.


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
        The factory must deterministically describe the same fixed U, including
        global phase, and must not include measurement gates. It is called once
        per build and the controlled circuit is reused for repeated powers.
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
                 state_prep=None,
                 *,
                 unitary_power=None,
                 machine=None):
        if unitary is None:
            raise ValueError("unitary must be provided")
        if not callable(unitary):
            raise TypeError("unitary must be callable")
        if state_prep is not None and not callable(state_prep):
            raise TypeError("state_prep must be callable")
        if unitary_power is not None and not callable(unitary_power):
            raise TypeError("unitary_power must be callable")
        self._unitary = unitary
        self.n_count = positive_integer(n_count, "n_count")
        self.n_target = positive_integer(n_target, "n_target")
        self.n_qubits = self.n_count + self.n_target
        self._state_prep = state_prep
        self._unitary_power = unitary_power
        self._machine = machine
        self._circuit = None  # cached circuit

    # ── helper: controlled-U^(2^k) ───────────────────────────

    def _controlled_u_power(self, q_count, q_targets, power, base_u=None):
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
        if self._unitary_power is not None:
            powered = self._unitary_power(q_targets, repetitions)
            if not isinstance(powered, QCircuit):
                raise TypeError("unitary_power must return QCircuit")
            return powered.control([q_count])

        cir = QCircuit()
        if base_u is None:
            base_u = self._unitary(q_targets)
        if not isinstance(base_u, QCircuit):
            raise TypeError("unitary must return QCircuit")
        controlled = base_u.control([q_count])
        # Repeat U 2^power times, each controlled by q_count
        for _ in range(repetitions):
            cir << controlled
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

        return self._circuit_for_registers(q_count, q_target)

    def _circuit_for_registers(self, q_count, q_target):
        """Shared circuit engine; the compatibility facade selects its layout."""

        cir = QCircuit()

        # Step 0: prepare eigenstate (if provided)
        if self._state_prep is not None:
            preparation = self._state_prep(q_target)
            if not isinstance(preparation, QCircuit):
                raise TypeError("state_prep must return QCircuit")
            cir << preparation

        # Step 1: Hadamard on all counting qubits
        cir << hadamard_circuit(q_count)

        # Step 2: controlled-U^(2^k) for each counting qubit
        # q_count[0] controls U^1, q_count[1] controls U^2, ...
        base_u = None
        if self._unitary_power is None:
            base_u = self._unitary(q_target)
            if not isinstance(base_u, QCircuit):
                raise TypeError("unitary must return QCircuit")
        for k in range(self.n_count):
            cir << self._controlled_u_power(
                q_count[k], q_target, k, base_u
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
        probabilities = self.run_dict(shots=shots)
        # Preserve the earlier exact-mode tie rule: lowest bitstring first.
        if shots == -1:
            best = min(probabilities, key=lambda key: (-probabilities[key], key))
        else:
            best = max(probabilities, key=probabilities.get)
        return int(best, 2) / (1 << self.n_count)

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
        self._validate_shots(shots)
        prog = QProg(self.n_qubits)
        q_count = prog.qubits()[:self.n_count]
        prog << self.build_circuit()
        return self._execute_distribution(prog, q_count, shots)

    @staticmethod
    def _validate_shots(shots):
        if isinstance(shots, bool) or shots != -1:
            return positive_integer(shots, "shots")
        # Exclude -1.0 as well as booleans from integer shot parameters.
        from numbers import Integral
        if not isinstance(shots, Integral):
            raise ValueError("shots must be -1 or a positive integer")
        return -1

    def _execute_distribution(self, program, counting_register, shots):
        """Shared native exact/readout path, with compact classical-bit order."""
        shots = self._validate_shots(shots)
        qvm = self._machine if self._machine is not None else CPUQVM()
        if shots == -1:
            qvm.run(program, 1)
            weights = qvm.result().get_prob_dict(counting_register)
        else:
            program << measure_all(counting_register, list(range(self.n_count)))
            qvm.run(program, shots)
            weights = qvm.result().get_counts()
        return normalize_distribution(weights, self.n_count)
