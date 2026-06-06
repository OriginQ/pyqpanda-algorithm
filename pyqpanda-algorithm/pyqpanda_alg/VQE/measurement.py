"""
Expectation value measurement for Pauli Hamiltonians.

To measure ⟨ψ|H|ψ⟩, we decompose H into Pauli terms and compute each
term's expectation value via basis rotation + probability readout:
    - Z: measure directly in computational basis
    - X: apply H gate before measurement (rotates X-basis to Z-basis)
    - Y: apply Sdg then H before measurement (rotates Y-basis to Z-basis)
    - I: contributes +1 always, no measurement needed

Author: Bai
"""

import numpy as np
from pyqpanda3.core import QCircuit, QProg, CPUQVM, H as HGate, RZ


def _basis_rotation_circuit(qubits, pauli_str):
    """
    Build circuit that rotates measurement basis for a Pauli string.

    For each qubit position:
        'Z' or 'I' -> no gate needed
        'X' -> H gate (maps X-eigenstates to Z-eigenstates)
        'Y' -> RZ(-π/2) + H (maps Y-eigenstates to Z-eigenstates)

    Parameters
        qubits : list[int]
            Qubit indices.
        pauli_str : str
            Pauli string, e.g. "IXYZ".

    Returns
        QCircuit : basis rotation circuit.
    """
    cir = QCircuit()
    for i, pauli in enumerate(pauli_str):
        if pauli == 'X':
            cir << HGate(qubits[i])
        elif pauli == 'Y':
            cir << RZ(qubits[i], -np.pi / 2)
            cir << HGate(qubits[i])
        # 'Z' and 'I' need no rotation
    return cir


def measure_expectation(circuit, qubits, pauli_str, shots=1024):
    """
    Measure expectation value of a single Pauli string.

    Uses probability distribution from the simulator. The parity of
    measured bits (on non-I positions) determines the ±1 eigenvalue:
        eigenvalue(b) = (-1)^(count of '1' bits at non-I positions in b)

    Parameters
        circuit : QCircuit
            The state preparation circuit (ansatz with parameters applied).
        qubits : list[int]
            Qubit indices used in the circuit.
        pauli_str : str
            Pauli operator string, e.g. "XZ", "IY".
        shots : int, default=1024
            Number of samples for adding statistical noise.
            If shots <= 0, uses exact probabilities (no noise).

    Returns
        float : estimated expectation value in range [-1, +1].

    Examples:
        .. code-block:: python

            expectation = measure_expectation(ansatz_circuit, [0,1], "ZZ")
    """
    n = len(qubits)
    if len(pauli_str) != n:
        raise ValueError(
            f"Pauli string length ({len(pauli_str)}) must match "
            f"qubit count ({n})."
        )

    # positions where we actually need to check parity
    active_positions = [i for i, p in enumerate(pauli_str) if p != 'I']
    if not active_positions:
        return 1.0

    # build full program: ansatz + basis rotation
    prog = QProg(n)
    qvec = prog.qubits()

    prog << circuit
    prog << _basis_rotation_circuit(qvec, pauli_str)

    # run on CPUQVM simulator to get probability distribution
    machine = CPUQVM()
    machine.run(prog, shots)
    prob_list = machine.result().get_prob_list(qvec)

    # compute expectation from probability distribution
    expectation = 0.0
    for state_idx, prob in enumerate(prob_list):
        if prob < 1e-15:
            continue
        # extract bits and compute parity on active positions
        parity = 0
        for pos in active_positions:
            # bit at position pos in the binary repr of state_idx
            # qubit ordering: pos=0 is most significant bit
            bit_val = (state_idx >> (n - 1 - pos)) & 1
            parity += bit_val
        sign = (-1) ** (parity % 2)
        expectation += sign * prob

    return expectation


def compute_energy(circuit, qubits, hamiltonian, shots=1024):
    """
    Compute total energy ⟨ψ|H|ψ⟩ for a full Pauli Hamiltonian.

    Iterates over all Pauli terms, measures each, and sums weighted results:
        E = Σ ci * ⟨Pi⟩

    Parameters
        circuit : QCircuit
            State preparation circuit (ansatz).
        qubits : list[int]
            Qubit indices.
        hamiltonian : PauliHamiltonian
            The Hamiltonian to evaluate.
        shots : int, default=1024
            Measurement shots per Pauli term.

    Returns
        float : estimated total energy.

    Examples:
        .. code-block:: python

            from pyqpanda_alg.VQE import PauliHamiltonian, compute_energy

            H = PauliHamiltonian()
            H.add_term(-1.05, "II")
            H.add_term(0.39, "ZZ")

            energy = compute_energy(ansatz_circuit, [0, 1], H)
    """
    energy = 0.0

    for coeff, pauli_str in hamiltonian.terms:
        # all-I terms don't need quantum measurement
        if all(ch == 'I' for ch in pauli_str):
            energy += coeff
        else:
            exp_val = measure_expectation(circuit, qubits, pauli_str, shots)
            energy += coeff * exp_val

    return energy
