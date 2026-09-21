"""Readout mitigation and zero-noise extrapolation example."""

import numpy as np

from pyqpanda_alg.QMitigation import (
    assignment_matrix_from_calibration,
    mitigate_probabilities,
    polynomial_extrapolate,
    richardson_extrapolate,
)


def main():
    calibration = {
        "0": {"0": 950, "1": 50},
        "1": {"0": 80, "1": 920},
    }
    assignment = assignment_matrix_from_calibration(calibration)
    corrected = mitigate_probabilities(
        {"0": 540, "1": 460},
        assignment,
    )

    print("assignment matrix:")
    print(assignment)
    print("readout-mitigated probabilities:", corrected)

    scales = np.array([1.0, 3.0, 5.0])
    noisy_expectations = 0.82 + 0.04 * scales - 0.003 * scales**2

    richardson = richardson_extrapolate(scales, noisy_expectations)
    polynomial = polynomial_extrapolate(
        scales,
        noisy_expectations,
        degree=2,
    )

    print("Richardson zero-noise estimate:", richardson)
    print("polynomial zero-noise estimate:", polynomial.value)
    print("polynomial residual norm:", polynomial.residual_norm)


if __name__ == "__main__":
    main()
