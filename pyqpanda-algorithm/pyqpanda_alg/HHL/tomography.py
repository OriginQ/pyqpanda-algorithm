"""Quantum state tomography for the HHL data register.

Runtime backends report measurement counts, not statevectors, so the
post-selected solution state of the data register must be reconstructed
from sampling.  The measurement plan is the standard Pauli-basis
tomography declared by
:class:`~pyqpanda_alg.HHL.resources.HHLResourceEstimate`:
``3 ** data_qubits`` circuits, one per X/Y/Z string over the data
register, each measuring the success ancilla and every data qubit in
the string's basis.  These functions are pure and backend-independent:
circuit rotation, parity expectation from counts, and density-matrix
reconstruction with dominant-eigenvector extraction.

Basis rotations
---------------
Measuring in the X basis is a Hadamard before the Z-basis measurement;
measuring in the Y basis is ``S^dagger`` (``P(-pi/2)``) then a
Hadamard.  With these rotations the outcome bit ``0`` always maps to
the eigenvalue ``+1`` and ``1`` to ``-1`` of the measured Pauli, so a
single parity rule serves every basis.

Reconstruction
--------------
``reconstruct_density_matrix`` estimates every Pauli coefficient
``<P>`` from the counts of the one canonical basis compatible with
``P`` (identity positions are measured in Z and marginalized), then
forms ``rho = (1/2**d) * sum_P <P> P``.  The dominant eigenvector of
``rho`` is the reconstructed solution state; its eigenvalue is the
state fidelity of ``rho`` against that state, so ``1 - fidelity`` is
the reconstruction uncertainty — a self-consistency measure of the
reconstructed density matrix, not a statistical error bar.  Because
the linear inversion imposes no positivity constraint, shot noise can
push the estimated Pauli norm above 1 and the dominant eigenvalue
outside ``[0, 1]`` (e.g. ``<X> = <Y> = <Z> = -0.8`` gives
``lambda_max = (1 + 0.8 * sqrt(3)) / 2 ~ 1.19``); the HHL solver
clamps the reported fidelity and uncertainty into ``[0, 1]`` at the
metadata construction site.
"""

import itertools

import numpy as np
from pyqpanda3.core import H, P

#: Single-qubit Pauli matrices in the ``{I, X, Y, Z}`` basis.
_PAULI_MATRICES = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1j], [1j, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


def pauli_basis_strings(data_qubits: int) -> list[str]:
    """All ``3**data_qubits`` X/Y/Z strings over the data register.

    The strings are generated in lexicographic order and one string
    names the measurement basis of each data qubit (position ``j``
    names data qubit ``j``).  This is exactly the circuit count
    declared by ``HHLResourceEstimate.tomography_circuits``.
    """
    return [
        "".join(letters)
        for letters in itertools.product("XYZ", repeat=data_qubits)
    ]


def apply_basis_rotation(program, basis: str, data_qubits) -> None:
    """Append the rotations measuring each data qubit in ``basis``.

    ``data_qubits`` is the tuple of qubit indices, so ``basis[j]``
    names the basis of ``data_qubits[j]``: ``X`` appends a Hadamard,
    ``Y`` appends ``S^dagger`` then a Hadamard, and ``Z`` appends
    nothing.  The program is extended in place and can carry its own
    gates before the rotation.
    """
    for index, letter in enumerate(basis):
        qubit = data_qubits[index]
        if letter == "X":
            program << H(qubit)
        elif letter == "Y":
            program << P(qubit, -np.pi / 2) << H(qubit)
        elif letter != "Z":
            raise ValueError(
                f"unknown basis letter {letter!r}; expected X, Y, or Z"
            )


def pauli_expectation(pauli: str, counts: dict[str, int]) -> float:
    """Estimate ``<pauli>`` from counts measured in a compatible basis.

    ``counts`` maps outcome bitstrings to shot counts; ``key[j]`` is
    the outcome of data qubit ``j`` with ``0`` meaning the eigenvalue
    ``+1`` and ``1`` the eigenvalue ``-1`` of the measured Pauli, so
    the expectation is the parity-weighted outcome average.  Positions
    where ``pauli`` is ``I`` contribute no sign — those outcomes are
    marginalized.  The measurement basis must be compatible with
    ``pauli``: every nontrivial position measured in the matching
    X/Y/Z basis, identity positions in any basis.
    """
    if not pauli or not all(letter in "IXYZ" for letter in pauli):
        raise ValueError(f"pauli {pauli!r} must use only I, X, Y, Z")
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for key, count in counts.items():
        sign = 1.0
        for index, letter in enumerate(pauli):
            if letter != "I" and key[index] == "1":
                sign = -sign
        value += sign * count
    return value / total


def reconstruct_density_matrix(
    basis_counts: dict[str, dict[str, int]], data_qubits: int
) -> np.ndarray:
    """Reconstruct the data-register density matrix from basis counts.

    ``basis_counts`` maps each X/Y/Z basis string to the counts of its
    measurement circuit (outcome ``key[j]`` is data qubit ``j``).  Each
    Pauli coefficient ``<P>`` is estimated from the canonical basis
    matching ``P`` (identity positions measured in Z), and the density
    matrix is ``rho = (1/2**data_qubits) * sum_P <P> P``.  A missing
    basis raises ``KeyError``, so incomplete tomography fails loudly.
    """
    dimension = 1 << data_qubits
    density = np.zeros((dimension, dimension), dtype=np.complex128)
    for pauli in itertools.product("IXYZ", repeat=data_qubits):
        pauli = "".join(pauli)
        expectation = pauli_expectation(pauli, basis_counts[_canonical_basis(pauli)])
        density += expectation * _pauli_matrix(pauli)
    return density / dimension


def dominant_eigenvector(rho: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(fidelity, vector)`` of the dominant eigenvector.

    The reconstructed density matrix is Hermitian by construction, so
    its largest eigenvalue and eigenvector come from a Hermitian
    eigensolve.  The eigenvalue is the state fidelity of ``rho``
    against the dominant eigenvector; ``1 - fidelity`` is the
    reconstruction uncertainty — a self-consistency measure, not a
    statistical error bar.  Linear inversion has no positivity
    constraint, so shot noise can push the raw eigenvalue outside
    ``[0, 1]``; callers that report it as a fidelity should clamp it
    (the HHL solver clamps the reported value at the metadata
    construction site).
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(
            f"density matrix must be square, got shape {rho.shape}"
        )
    values, vectors = np.linalg.eigh(rho)
    index = int(np.argmax(values))
    return float(values[index]), np.asarray(vectors[:, index])


def _canonical_basis(pauli: str) -> str:
    """Return the canonical X/Y/Z basis compatible with ``pauli``.

    Nontrivial letters keep their basis and identity positions measure
    in Z, so every Pauli string maps to exactly one of the ``3**d``
    basis strings.
    """
    return "".join(letter if letter != "I" else "Z" for letter in pauli)


def _pauli_matrix(pauli: str) -> np.ndarray:
    """Tensor product of the single-qubit Pauli matrices of ``pauli``."""
    matrix = np.array([[1.0]])
    for letter in pauli:
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix
