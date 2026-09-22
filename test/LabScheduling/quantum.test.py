"""Real CPUQVM tests for bit ordering, Hamiltonians, invariants and sampling."""

from pathlib import Path

import numpy as np
import pytest
from pyqpanda3.core import CPUQVM, QProg, X
from pyqpanda_alg.LabScheduling import (
    Problem,
    build_circuit,
    circuit_probabilities,
    compile_qubo,
    load_problem,
    solve_qaoa,
    to_ising,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm/example/LabScheduling/data"
)


def small_model():
    return compile_qubo(load_problem(DATA / "simple.json"))


def test_simulator_little_endian_contract():
    machine = CPUQVM()
    machine.run(QProg(3) << X(0) << X(2), shots=1)
    probabilities = np.abs(machine.result().get_state_vector()) ** 2
    assert probabilities.argmax() == 5
    assert machine.result().get_prob_dict()["101"] == pytest.approx(1)
    machine.run(QProg(3) << X(0), shots=1)
    assert np.abs(machine.result().get_state_vector()).argmax() == 1


def test_ising_diagonal_matches_qubo_all_states_including_indices_above_9():
    model = compile_qubo(load_problem(DATA / "medium.json"))
    terms = to_ising(model).terms()
    for index in range(1 << len(model.variables)):
        bits = [(index >> i) & 1 for i in range(len(model.variables))]
        energy = 0.0
        for term in terms:
            sign = 1
            for pauli in term.paulis():
                if pauli.is_Z():
                    sign *= 1 - 2 * bits[pauli.qbit()]
            energy += sign * term.coef().real
        assert energy == pytest.approx(model.energy(bits), abs=1e-10)


@pytest.mark.parametrize("parameters", [[0.0, 0.0], [0.4, 0.7], [0.2, -0.5, 0.9, 1.1]])
def test_xy_preserves_each_domain_and_probability_normalization(parameters):
    model = small_model()
    probabilities = circuit_probabilities(model, parameters)
    assert probabilities.sum() == pytest.approx(1)
    for index, probability in enumerate(probabilities):
        if probability > 1e-12:
            assert all(
                sum((index >> i) & 1 for i in domain) == 1 for domain in model.domains
            )
    if parameters == [0.0, 0.0]:
        assert np.count_nonzero(probabilities > 1e-12) == 9
        assert probabilities[probabilities > 1e-12] == pytest.approx(np.full(9, 1 / 9))


def test_x_initial_state_is_uniform_and_not_one_hot():
    probabilities = circuit_probabilities(small_model(), [0.0, 0.0], "x")
    assert probabilities == pytest.approx(np.full(64, 1 / 64))


def test_phase_and_x_mixer_against_independent_dense_numpy_simulation():
    model = small_model()
    gamma, beta = 0.61, -0.32
    state = np.full(64, 1 / 8, dtype=complex)
    energies = np.array(
        [model.energy([(i >> j) & 1 for j in range(6)]) for i in range(64)]
    )
    state *= np.exp(-1j * gamma * energies / model.penalty)
    for qubit in range(6):
        previous = state.copy()
        for index in range(64):
            state[index] = (
                np.cos(beta) * previous[index]
                - 1j * np.sin(beta) * previous[index ^ (1 << qubit)]
            )
    assert circuit_probabilities(model, [gamma, beta], "x") == pytest.approx(
        np.abs(state) ** 2
    )


def test_fixed_seed_reproducibility_and_best_is_actually_sampled():
    model = small_model()
    first = solve_qaoa(model, seed=7, maxiter=25, restarts=1)
    second = solve_qaoa(model, seed=7, maxiter=25, restarts=1)
    assert first["counts"] == second["counts"]
    assert first["parameters"] == pytest.approx(second["parameters"])
    assert first["best"] == second["best"]
    assert sum(first["counts"].values()) == first["shots"]
    bitstring = "".join(map(str, first["best"]["bits"][::-1]))
    assert first["counts"][bitstring] > 0
    assert first["expected_energy"] <= first["initial_expected_energy"]
    assert first["best"]["objective"] == 5


def test_infeasible_not_mislabeled_as_quantum_certificate():
    model = compile_qubo(load_problem(DATA / "infeasible.json"))
    result = solve_qaoa(model, maxiter=6, restarts=1)
    assert result["status"] == "no_feasible_sample"
    assert result["best"]["objective"] is None
    assert result["best"]["violation_count"] == 1
    assert result["one_hot_probability"] == pytest.approx(1)


def test_singleton_empty_domain_and_zero_task_boundaries():
    data = {"name": "boundary", "horizon": 1, "resources": [{"id": "r"}], "tasks": []}
    assert solve_qaoa(compile_qubo(Problem.from_dict(data)))["status"] == "trivial"
    data["tasks"] = [{"id": "t", "group": "g", "duration": 1, "options": []}]
    assert (
        solve_qaoa(compile_qubo(Problem.from_dict(data)))["status"]
        == "infeasible_domain"
    )
    data["tasks"][0]["options"] = [{"resource": "r", "start": 0, "cost": 0}]
    model = compile_qubo(Problem.from_dict(data))
    assert circuit_probabilities(model, [0.4, 0.7]) == pytest.approx([0, 1])
    assert solve_qaoa(model, maxiter=6, restarts=1)["best"]["objective"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"shots": 0},
        {"layers": 0},
        {"layers": 9},
        {"restarts": 0},
        {"maxiter": 0},
        {"mixer": "bad"},
    ],
)
def test_invalid_solver_options(kwargs):
    with pytest.raises(ValueError):
        solve_qaoa(small_model(), **kwargs)


def test_invalid_circuit_and_scale_and_qubit_budget():
    model = small_model()
    for params in [[], [1], [float("nan"), 0]]:
        with pytest.raises(ValueError):
            build_circuit(model, params)
    with pytest.raises(ValueError):
        build_circuit(model, [0, 0], "bad")
    for scale in [0, -1, float("inf")]:
        with pytest.raises(ValueError):
            to_ising(model, scale)
    data = {
        "name": "large",
        "horizon": 20,
        "resources": [{"id": "r"}],
        "tasks": [
            {
                "id": "t",
                "group": "g",
                "duration": 1,
                "options": [
                    {"resource": "r", "start": i, "cost": 0} for i in range(17)
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="1..16"):
        solve_qaoa(compile_qubo(Problem.from_dict(data)))


def test_cobyla_budget_is_not_silently_increased():
    with pytest.raises(ValueError, match="COBYLA"):
        solve_qaoa(small_model(), layers=4, maxiter=6)
