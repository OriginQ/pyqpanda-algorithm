"""Parameter-shift gradient example.

This example uses analytic functions so the differentiation API is easy to
inspect. Replace expectation() with a CPUQVM or QCloud circuit evaluator in a
real variational workload.
"""

import numpy as np

from pyqpanda_alg.ParameterShift import (
    build_shift_schedule,
    parameter_shift_gradient,
    parameter_shift_jacobian,
)


def expectation(theta):
    return np.cos(theta[0]) + 0.25 * np.cos(theta[1])


def observables(theta):
    return np.array(
        [
            np.cos(theta[0]),
            np.sin(theta[0]) * np.cos(theta[1]),
        ]
    )


def main():
    theta = np.array([0.2, -0.4])

    schedule = build_shift_schedule(theta)
    print("remote batch points:")
    for entry in schedule.entries:
        print(
            "parameter",
            entry.parameter_index,
            "coefficient",
            entry.coefficient,
            "point",
            entry.parameters,
        )

    print("gradient:", parameter_shift_gradient(expectation, theta))
    print("jacobian:", parameter_shift_jacobian(observables, theta))


if __name__ == "__main__":
    main()
