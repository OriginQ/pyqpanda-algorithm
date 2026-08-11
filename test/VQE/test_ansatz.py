"""Structure and concrete-circuit tests for the hardware-efficient ansatz.

The ansatz is the built-in variational circuit of the VQE package:
alternating RY/RZ rotation layers with linear CNOT entanglement.  The
parameter ordering (layer-major, qubit-major, RY before RZ per qubit)
is a stable API contract and is pinned by the ordering test below.
"""

import math

import pytest
from pyqpanda3.hamiltonian import Hamiltonian

from pyqpanda_alg.VQE.ansatz import hardware_efficient_ansatz
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def test_hardware_efficient_ansatz_parameter_count():
    ansatz = hardware_efficient_ansatz(num_qubits=2, layers=2)
    assert ansatz.mutable_parameter_total() == 8
    assert ansatz.qubits() == [0, 1]


def test_ansatz_parameter_ordering_is_layer_major_qubit_major():
    parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ansatz = hardware_efficient_ansatz(num_qubits=2, layers=2)
    gates = ansatz(parameters).circuits()[0].gate_operations()
    rotations = [
        (gate.name(), gate.target_qubits(), gate.parameters())
        for gate in gates
        if gate.name() in ("RY", "RZ")
    ]
    assert rotations == [
        ("RY", [0], [0.1]),
        ("RZ", [0], [0.2]),
        ("RY", [1], [0.3]),
        ("RZ", [1], [0.4]),
        ("RY", [0], [0.5]),
        ("RZ", [0], [0.6]),
        ("RY", [1], [0.7]),
        ("RZ", [1], [0.8]),
    ]


def test_ansatz_entanglement_is_linear_chain_per_layer():
    ansatz = hardware_efficient_ansatz(num_qubits=3, layers=1)
    gates = ansatz([0.1] * 6).circuits()[0].gate_operations()
    assert [(gate.name(), gate.target_qubits()) for gate in gates] == [
        ("RY", [0]),
        ("RZ", [0]),
        ("RY", [1]),
        ("RZ", [1]),
        ("RY", [2]),
        ("RZ", [2]),
        ("CNOT", [0, 1]),
        ("CNOT", [1, 2]),
    ]


def test_ansatz_materializes_expected_gate_count():
    ansatz = hardware_efficient_ansatz(num_qubits=2, layers=2)
    circuit = ansatz([0.0] * 8).circuits()[0]
    assert circuit.count_ops() == {"RY": 4, "RZ": 4, "CNOT": 2}


def test_ansatz_binds_and_executes_locally():
    # Zero angles reduce every rotation to identity, so the two CNOTs
    # cancel and the ZZ expectation value of the |00> state is exactly
    # one -- a closed-form check that the bound circuit runs through
    # the local CPU backend.
    ansatz = hardware_efficient_ansatz(num_qubits=2, layers=2)
    bound = ansatz([0.0] * 8)
    backend = LocalBackend()
    task = backend.submit_estimate(
        (bound.circuits()[0], Hamiltonian({"Z0 Z1": 1.0})),
        options=ExecutionOptions(),
    )
    value = task.result().single_value()
    assert math.isfinite(value)
    assert abs(value - 1.0) < 1e-9


def test_ansatz_rejects_insufficient_qubits():
    with pytest.raises(ValueError, match="num_qubits"):
        hardware_efficient_ansatz(num_qubits=1, layers=1)


def test_ansatz_rejects_invalid_layer_count():
    with pytest.raises(ValueError, match="layers"):
        hardware_efficient_ansatz(num_qubits=2, layers=0)
