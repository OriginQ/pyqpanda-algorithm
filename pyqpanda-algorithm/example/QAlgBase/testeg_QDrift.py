"""qDRIFT randomized Hamiltonian-simulation example."""

import numpy as np

from pyqpanda_alg.QDrift import (
    PauliTerm,
    apply_qdrift_plan,
    exact_time_evolution,
    sample_qdrift,
    state_fidelity,
)


def main():
    terms = [
        PauliTerm(0.70, "ZI"),
        PauliTerm(-0.45, "XZ"),
        PauliTerm(0.25, "YY"),
    ]
    initial = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    plan = sample_qdrift(
        terms,
        time=0.8,
        epsilon=0.05,
        seed=7,
    )
    approximate = apply_qdrift_plan(initial, plan)
    exact = exact_time_evolution(terms, initial, time=0.8)

    print("segments:", plan.segment_count)
    print("qDRIFT bound:", plan.diamond_error_bound)
    print("trajectory fidelity:", state_fidelity(approximate, exact))
    print("first rotations:", plan.rotations[:5])


if __name__ == "__main__":
    main()
