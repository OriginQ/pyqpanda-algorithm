"""Bell-state example for the QST module.

Replace the ideal probability dictionaries below with raw counts returned by
a PyQPanda simulator or hardware backend after executing the same X/Y/Z basis
settings. The reconstruction accepts either counts or probabilities.
"""

import numpy as np

from pyqpanda_alg.QST import (
    linear_inversion_tomography,
    pauli_measurement_plan,
    probabilities_from_density_matrix,
    pure_state_fidelity,
)


def main():
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    target = np.outer(bell, bell.conj())

    measurements = {
        basis: probabilities_from_density_matrix(target, basis)
        for basis in pauli_measurement_plan(2)
    }
    reconstructed = linear_inversion_tomography(measurements)

    print("Measurement bases:", ", ".join(measurements))
    print("Reconstructed density matrix:")
    print(np.round(reconstructed, 6))
    print("Bell-state fidelity:", round(pure_state_fidelity(reconstructed, bell), 8))


if __name__ == "__main__":
    main()
