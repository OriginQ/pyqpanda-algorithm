"""Minimal Variational Quantum Linear Solver example."""

import numpy as np

from pyqpanda_alg.VQLS import VQLS


def main():
    matrix = np.array(
        [
            [1.0, -1.0 / 3.0],
            [-1.0 / 3.0, 1.0],
        ]
    )
    vector = np.array([1.0, 0.0])

    solver = VQLS(
        matrix,
        vector,
        layers=2,
        max_iterations=400,
        tolerance=1e-7,
        learning_rate=0.35,
        perturbation=0.15,
        seed=7,
    )
    result = solver.solve()

    print("cost:", result.cost)
    print("residual norm:", result.residual_norm)
    print("solution:", np.real_if_close(result.solution))


if __name__ == "__main__":
    main()
