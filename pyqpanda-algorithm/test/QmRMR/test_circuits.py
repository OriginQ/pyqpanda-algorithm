import numpy as np

from pyqpanda3.core import CPUQVM, QProg, X

from pyqpanda_alg.QmRMR.constrained import QmRMRProblem
from pyqpanda_alg.QmRMR.constrained.circuits import (
    append_circuit_to_program,
    build_dicke_state,
    constrained_qaoa_circuit,
    exchange_ansatz_circuit,
    parity_xy_mixer,
)


def _probabilities(circuit, n):
    program = QProg(n)
    program.append(circuit)
    machine = CPUQVM()
    machine.run(program, 1)
    return machine.result().get_prob_dict(program.qubits())


def test_dicke_state_is_uniform_and_fixed_weight():
    circuit = build_dicke_state(list(range(4)), 2, mode="amplitude")
    probabilities = _probabilities(circuit, 4)
    feasible = {
        key[::-1]: value
        for key, value in probabilities.items()
        if key[::-1].count("1") == 2
    }
    assert len(feasible) == 6
    np.testing.assert_allclose(sum(feasible.values()), 1.0, atol=1e-12)
    for probability in feasible.values():
        np.testing.assert_allclose(probability, 1.0 / 6.0, atol=1e-12)


def test_relevance_biased_dicke_state_is_normalized_and_fixed_weight():
    circuit = build_dicke_state(
        list(range(4)),
        2,
        mode="relevance",
        weights=[0.0, 0.0, 0.0, 1.0],
        strength=2.0,
    )
    probabilities = _probabilities(circuit, 4)
    feasible = {
        key[::-1]: value
        for key, value in probabilities.items()
        if key[::-1].count("1") == 2
    }
    np.testing.assert_allclose(sum(feasible.values()), 1.0, atol=1e-12)
    assert len(feasible) == 6
    assert feasible["1001"] > feasible["1100"]


def test_xy_mixer_preserves_hamming_weight():
    initial = build_dicke_state(list(range(6)), 3, mode="amplitude")
    mixer = parity_xy_mixer(list(range(6)), 0.37)
    program = QProg(6)
    program.append(initial)
    program.append(mixer)
    machine = CPUQVM()
    machine.run(program, 1)
    probabilities = machine.result().get_prob_dict(program.qubits())
    feasible_mass = sum(
        probability
        for key, probability in probabilities.items()
        if key.count("1") == 3
    )
    np.testing.assert_allclose(feasible_mass, 1.0, atol=1e-12)


def test_exchange_circuit_preserves_hamming_weight_and_reports_parameters():
    problem = QmRMRProblem(np.zeros((6, 6)), np.zeros(6), 3)
    parameters = np.linspace(-0.2, 0.2, 18)
    circuit, active = exchange_ansatz_circuit(problem, parameters)
    assert active == 15
    probabilities = _probabilities(circuit, 6)
    np.testing.assert_allclose(
        sum(value for key, value in probabilities.items() if key.count("1") == 3),
        1.0,
        atol=1e-12,
    )


def test_qaoa_circuit_has_expected_parameter_pairs():
    problem = QmRMRProblem(np.eye(4), np.zeros(4), 2)
    circuit = constrained_qaoa_circuit(
        problem,
        [0.1, 0.2, -0.1, 0.3],
        initializer="amplitude",
    )
    assert circuit.size() > 0
    assert circuit.depth() < 100


def test_statevector_qubit_guard():
    problem = QmRMRProblem(np.zeros((21, 21)), np.zeros(21), 1)
    with np.testing.assert_raises(ValueError):
        constrained_qaoa_circuit(problem, [0.1, 0.2], initializer="amplitude")

