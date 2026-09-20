"""Quantum Natural Gradient example for a one-qubit RY/RX-style state."""

import numpy as np

from pyqpanda_alg.QNG import fubini_study_metric, natural_gradient_step


def main():
    theta = np.array([0.4])

    state = np.array(
        [
            np.cos(theta[0] / 2.0),
            -1j * np.sin(theta[0] / 2.0),
        ]
    )
    state_jacobian = np.array(
        [
            [
                -0.5 * np.sin(theta[0] / 2.0),
                -0.5j * np.cos(theta[0] / 2.0),
            ]
        ]
    )

    # Example Euclidean objective gradient. In a variational algorithm this can
    # come from parameter-shift differentiation of the measured cost.
    cost_gradient = np.array([np.sin(theta[0])])

    metric = fubini_study_metric(state, state_jacobian)
    step = natural_gradient_step(
        theta,
        state,
        state_jacobian,
        cost_gradient,
        learning_rate=0.05,
    )

    print("Fubini-Study metric:")
    print(metric)
    print("natural-gradient direction:", step.result.direction)
    print("updated parameters:", step.parameters)
    print("metric diagnostics:", step.result.diagnostics)


if __name__ == "__main__":
    main()
