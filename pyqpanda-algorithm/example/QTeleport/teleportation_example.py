"""Quantum teleportation example using QTeleport."""

import numpy as np

from pyqpanda_alg.QTeleport import (
    coherent_teleportation_circuit,
    reference_teleportation,
    teleportation_gate_counts,
)


logical = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
reference = reference_teleportation(logical)

print("input:", reference["logical_state"])
print("Bob density matrix:")
print(reference["bob_density_matrix"])
print("fidelity:", reference["fidelity"])
print("logical gate counts:", teleportation_gate_counts())

# q0 carries the message; q1/q2 begin in |0>.
print(coherent_teleportation_circuit([0, 1, 2]))
