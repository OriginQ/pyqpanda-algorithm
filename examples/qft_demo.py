"""Build a reusable QFT circuit with pyqpanda-algorithm."""

from pyqpanda3.core import QProg, X

from pyqpanda_alg.QFT import qft_circuit


program = QProg(3)
qubits = program.qubits()

# Prepare |001> under the package's low-to-high convention, then apply QFT.
program << X(qubits[0])
program << qft_circuit(qubits)

print(program)
