import numpy as np
from pyqpanda_alg.HHL import HHL

"""
HHL solves a linear system Ax = b for a Hermitian matrix A. HHL(A, b).run()
returns the normalized solution vector. clock_qubits sets the eigenvalue
resolution and success_prob is the post-selection probability (ancilla measured
as |1>). When the eigenvalues of A cannot be represented exactly by the clock
register the fidelity improves as clock_qubits grows.
"""


def main():
    A = np.array([[1.0, -1.0 / 3],
                  [-1.0 / 3, 1.0]])
    b = np.array([0.0, 1.0])

    solver = HHL(A, b, clock_qubits=2)
    x_q = solver.run()

    x_c = np.linalg.solve(A, b)
    x_c = x_c / np.linalg.norm(x_c)

    print("quantum solution  :", np.round(x_q, 4))
    print("classical solution:", np.round(x_c, 4))
    print("success probability:", round(solver.success_prob, 4))


if __name__ == "__main__":
    main()
