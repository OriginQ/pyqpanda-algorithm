# QTeleport: coherent quantum teleportation

pyqpanda_alg.QTeleport adds a reusable three-qubit quantum teleportation
building block for PyQPanda3 plus an independent NumPy reference model.

## Protocol

The qubit contract is:

- q0: message qubit containing an arbitrary state alpha|0> + beta|1>;
- q1: Alice's half of the Bell pair, initially |0>;
- q2: Bob's half of the Bell pair / output, initially |0>.

The circuit prepares a Bell pair between q1 and q2, performs the Bell-basis
transform on q0/q1, then applies the usual X and Z corrections coherently:

    H(q1)
    CNOT(q1, q2)
    CNOT(q0, q1)
    H(q0)
    CNOT(q1, q2)       # X correction controlled by Bell bit q1
    CZ(q0, q2)         # Z correction controlled by Bell bit q0

This is the deferred-measurement form of standard teleportation. It has the
same transfer semantics as measuring q0/q1 and feeding the two classical bits
forward, while avoiding backend-specific mid-circuit classical-control APIs.
After the correction gates, Bob's qubit is separable from the first two qubits
and carries the original message state.

## Circuit use

    from pyqpanda_alg.QTeleport import coherent_teleportation_circuit

    # q0 already contains the state to teleport; q1 and q2 start in |0>.
    circuit = coherent_teleportation_circuit([0, 1, 2])
    print(circuit)

The builder imports PyQPanda3 lazily. Importing QTeleport or using the
reference helpers does not initialize a quantum backend.

## Independent reference

reference_teleportation() performs the same six logical gates with a compact
NumPy state-vector implementation and traces out q0/q1 to obtain Bob's reduced
density matrix:

    import numpy as np
    from pyqpanda_alg.QTeleport import reference_teleportation

    state = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    result = reference_teleportation(state)

    print(result["fidelity"])
    print(result["bob_density_matrix"])

The helper accepts arbitrary non-zero complex amplitudes and normalizes them.
Set return_statevector=True when the full three-qubit reference state is useful
for demonstrations or downstream diagnostics.

## Scope

This module demonstrates ideal logical teleportation. It does not claim a
network transport layer, noisy-channel correction, entanglement purification,
or hardware calibration. Those concerns belong above or below this algorithmic
building block.
