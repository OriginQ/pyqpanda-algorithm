"""Minimal Quantum Phase Estimation example."""

from math import pi

from pyqpanda3.core import QCircuit, U1, X

from pyqpanda_alg.QPE import StructuredQPE as QPE


TARGET_PHASE = 3.0 / 8.0


def prepare_one(qubits):
    circuit = QCircuit()
    circuit << X(qubits[0])
    return circuit


def phase_unitary(qubits):
    circuit = QCircuit()
    circuit << U1(qubits[0], 2.0 * pi * TARGET_PHASE)
    return circuit


result = QPE(
    unitary=phase_unitary,
    system_qubits=1,
    precision_qubits=5,
    eigenstate_preparation=prepare_one,
).run()

print("phase:", result.phase)
print("bitstring:", result.bitstring)
print("probability:", result.probability)
