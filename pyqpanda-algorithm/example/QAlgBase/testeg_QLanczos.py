"""QLanczos exact-state reference example.

Replace the exact callbacks with PyQPanda or hardware matrix-element estimators
to run the same generalized eigenproblem on measured Krylov states.
"""

import numpy as np

from pyqpanda_alg.QLanczos import (
    QuantumLanczos,
    dense_hamiltonian_element,
    imaginary_time_evolver,
    statevector_overlap,
)


def main():
    hamiltonian = np.asarray(
        [
            [-1.20, 0.18, 0.00],
            [0.18, 0.15, -0.22],
            [0.00, -0.22, 1.10],
        ],
        dtype=np.complex128,
    )
    seed = np.asarray([1.0, 0.7, -0.4], dtype=np.complex128)
    seed = seed / np.linalg.norm(seed)

    solver = QuantumLanczos(
        seed_state=seed,
        evolve=imaginary_time_evolver(hamiltonian, delta_tau=0.45),
        overlap=statevector_overlap,
        hamiltonian=dense_hamiltonian_element(hamiltonian),
        order=3,
        num_roots=3,
        overlap_cutoff=1e-12,
    )
    result = solver.run()

    print("Ritz energies:", result.energies)
    print("Residual norms:", result.residual_norms)
    print(
        "Overlap rank: %d/%d; condition number: %.3e"
        % (
            result.overlap_rank,
            len(result.basis_states),
            result.condition_number,
        )
    )


if __name__ == "__main__":
    main()
