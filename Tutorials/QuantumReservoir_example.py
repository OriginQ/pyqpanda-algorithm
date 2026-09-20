"""Quantum reservoir computing example."""

import numpy as np

from pyqpanda_alg.QuantumReservoir import (
    QuantumReservoirRegressor,
    ReservoirConfig,
)


config = ReservoirConfig(
    n_qubits=3,
    layers=3,
    input_scale=0.8,
    reservoir_scale=1.0,
    seed=17,
)
model = QuantumReservoirRegressor(config, alpha=1e-4)

print("measurement order:", model.observables)

# Build the quantum feature-map circuit for one input sample.
# Replace [0, 1, 2] with allocated PyQPanda3 qubits in a backend program.
print(model.circuit([0, 1, 2], [0.2, -0.4, 0.7]))

# Example expectation rows. In an application these values come from executing
# the corresponding reservoir circuits and measuring model.observables.
z_expectations = np.array(
    [
        [0.72, -0.10, 0.31],
        [0.41, 0.22, 0.08],
        [-0.16, 0.63, 0.37],
        [-0.52, 0.28, 0.66],
    ]
)
zz_expectations = np.array(
    [
        [0.21, -0.08, 0.12],
        [0.04, 0.13, 0.19],
        [-0.11, 0.25, -0.06],
        [0.18, -0.03, 0.30],
    ]
)
targets = np.array([0.0, 0.3, 0.7, 1.0])

model.fit_measurements(
    z_expectations,
    targets,
    zz_expectations=zz_expectations,
)

prediction = model.predict_measurements(
    [[0.05, 0.51, 0.42]],
    zz_expectations=[[0.09, 0.20, 0.02]],
)
print("prediction:", prediction)
