"""QShadow end-to-end example: planning, CPUQVM sampling, and confidence budget."""

from __future__ import annotations

from pyqpanda3.core import CNOT, CPUQVM, QCircuit, QProg, RY

from pyqpanda_alg.QShadow import (
    PauliObservable,
    PyQPandaRunner,
    ShadowBudgetController,
    estimate_observables,
    optimize_basis_probabilities,
    planning_loss,
    run_measurement_plan,
    sample_measurement_plan,
    uniform_probabilities,
)


def prepare(qubits):
    """A reproducible three-qubit entangled state-preparation circuit."""

    return (
        QCircuit()
        << RY(qubits[0], 0.7)
        << RY(qubits[1], -0.4)
        << RY(qubits[2], 0.3)
        << CNOT(qubits[0], qubits[1])
        << CNOT(qubits[1], qubits[2])
    )


def exact_statevector():
    program = QProg(3)
    program << prepare(program.qubits())
    machine = CPUQVM()
    machine.run(program, shots=1)
    return machine.result().get_state_vector()


def main():
    observables = [
        PauliObservable.from_terms(
            {
                "ZZI": -1.0,
                "IZZ": -0.8,
                "XII": 0.20,
                "IXI": 0.15,
                "IIX": 0.10,
            },
            name="structured_energy",
        ),
        PauliObservable.from_terms(
            {"ZIZ": 1.0, "XXI": 0.25}, name="long_range_order"
        ),
    ]

    uniform = uniform_probabilities(3)
    optimisation = optimize_basis_probabilities(
        observables, probability_floor=0.04
    )
    print("uniform planning loss :", planning_loss(uniform, observables))
    print("optimised planning loss:", optimisation.final_loss)
    print("basis probabilities [X,Y,Z]:\n", optimisation.probabilities)

    plan = sample_measurement_plan(
        optimisation.probabilities, shots=4_000, seed=20260726
    )
    print("shots / unique backend jobs:", plan.shots, len(plan.grouped_shots))

    runner = PyQPandaRunner(prepare, n_qubits=3)
    dataset = run_measurement_plan(plan, runner, max_jobs=27)
    estimates = estimate_observables(
        dataset, observables, family_confidence=0.95
    )
    state = exact_statevector()
    for observable in observables:
        estimate = estimates[observable.name]
        exact = observable.exact_expectation(state)
        print(
            f"{observable.name}: estimate={estimate.value:.5f}, "
            f"exact={exact:.5f}, CI=[{estimate.lower:.5f}, {estimate.upper:.5f}], "
            f"min_matches={estimate.minimum_term_matches}"
        )

    controller = ShadowBudgetController(
        observables,
        target_half_width=0.25,
        family_confidence=0.95,
        min_shots=1_000,
        max_shots=8_000,
        batch_shots=500,
    )
    report = controller.evaluate(dataset)
    print(
        "budget:",
        report.reason,
        "next_batch_shots=",
        report.next_batch_shots,
        "hard_max=",
        report.max_shots,
    )


if __name__ == "__main__":
    main()
