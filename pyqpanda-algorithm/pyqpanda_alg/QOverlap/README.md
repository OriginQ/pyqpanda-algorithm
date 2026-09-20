# QOverlap

QOverlap provides a reusable multi-qubit SWAP Test for estimating state overlap.

For pure states, measuring the ancilla in zero with probability P0 gives

    |<psi|phi>|^2 = 2 * P0 - 1.

The implementation accepts caller-supplied state-preparation circuits, so it can
compare computational-basis states, variational states, encoded feature states,
or other equal-width preparations without duplicating their construction logic.

Example:

    from pyqpanda3.core import QCircuit, H, X
    from pyqpanda_alg.QOverlap import SwapTest

    def plus_state(qubits):
        circuit = QCircuit()
        circuit << H(qubits[0])
        return circuit

    def one_state(qubits):
        circuit = QCircuit()
        circuit << X(qubits[0])
        return circuit

    result = SwapTest(
        qubits_per_state=1,
        prepare_left=plus_state,
        prepare_right=one_state,
        shots=2048,
    ).run()

    print(result.overlap_squared)

The result object also reports the measured ancilla-zero probability, overlap
magnitude, dominant ancilla bitstring, and shot count.
