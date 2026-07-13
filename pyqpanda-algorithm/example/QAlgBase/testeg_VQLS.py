import numpy as np
from pyqpanda_alg.VQLS import VQLS

"""
VQLS solves a linear system Ax = b for a Hermitian matrix A variationally.
A hardware-efficient RY/CZ ansatz is optimized with scipy to minimize a global
cost built from Pauli expectation values and overlap amplitudes between the
ansatz state and b. VQLS(A, b).run() returns the normalized solution vector.
final_cost near 0 indicates convergence. ansatz_layers, seed, and optimizer are
tunable. Unlike HHL, VQLS is NISQ-friendly and needs no ancilla or clock register.
"""


def main():
    A = np.array([[1.0, -1.0 / 3],
                  [-1.0 / 3, 1.0]])
    b = np.array([0.0, 1.0])

    solver = VQLS(A, b, seed=0)
    x_q = solver.run()

    x_c = np.linalg.solve(A, b)
    x_c = x_c / np.linalg.norm(x_c)

    print("quantum solution  :", np.round(x_q, 4))
    print("classical solution:", np.round(x_c, 4))
    print("final cost:", round(solver.final_cost, 4))


if __name__ == "__main__":
    main()
