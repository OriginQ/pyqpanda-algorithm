"""Example: estimate a complex expectation with the Hadamard test.

For |+> and RX(theta), <+|RX(theta)|+> = exp(-i theta / 2).
"""

import numpy as np

from pyqpanda3.core import H, RX, QCircuit
from pyqpanda_alg.HadamardTest import HadamardTest, reference_expectation


THETA = np.pi / 3


def prepare_plus(qubits):
    circuit = QCircuit()
    circuit << H(qubits[0])
    return circuit


def rotate_x(qubits):
    circuit = QCircuit()
    circuit << RX(qubits[0], THETA)
    return circuit


def main():
    estimator = HadamardTest(
        state_preparation=prepare_plus,
        unitary=rotate_x,
        qnumber=1,
    )
    sampled = estimator.estimate_complex(shots=8192)

    state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    c = np.cos(THETA / 2)
    s = np.sin(THETA / 2)
    rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    exact = reference_expectation(state, rx)

    print("sampled:", sampled)
    print("reference:", exact)


if __name__ == "__main__":
    main()
