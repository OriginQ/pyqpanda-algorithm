# Quantum Fourier Transform (QFT)

pyqpanda_alg.QFT provides a dedicated reusable Quantum Fourier Transform
module for circuit construction and backend-independent reference semantics.

## Circuit API

    from pyqpanda3.core import QProg
    from pyqpanda_alg.QFT import qft_circuit

    program = QProg(4)
    qubits = program.qubits()
    program << qft_circuit(qubits)

The qubit sequence is interpreted from least-significant to most-significant,
matching the existing pyqpanda-algorithm QFT helper convention. By default the
builder includes the final bit-reversal swaps, so output follows ordinary
computational-basis index order.

Use inverse=True for the exact circuit dagger:

    inverse = qft_circuit(qubits, inverse=True)

Use do_swaps=False only when the consumer intentionally accepts bit-reversed
output. bit_reversed_indices(n) exposes that permutation explicitly instead of
leaving ordering implicit.

## Mathematical reference

reference_qft(state) evaluates the normalized transform directly in NumPy:

    y[k] = sum_j x[j] exp(+2 pi i j k / N) / sqrt(N)

The inverse uses the negative phase. The helper accepts any finite 1-D complex
vector with power-of-two length and does not silently renormalize it. This
provides a simulator-independent definition of the package convention.

## Gate cost

For n qubits, the forward circuit contains n Hadamard gates, n(n-1)/2
controlled phase rotations, and, when standard ordering is requested,
floor(n/2) swaps.

The QuantumFourierTransform class stores a validated qubit configuration when
callers need to build both forward and inverse circuits repeatedly.
