# QuantumReservoir: fixed quantum reservoir computing

QuantumReservoir adds a compact hybrid quantum reservoir-computing workflow to
pyqpanda_alg. The quantum circuit acts as a fixed nonlinear feature map; model
training happens only in a classical ridge readout. That makes the same
reservoir reusable across simulator, cloud, and hardware backends without a
variational quantum optimization loop.

## Reservoir dynamics

For an input sample x, each layer:

1. streams input features cyclically across qubits with RY encoding;
2. applies fixed seeded RX and RZ rotations;
3. applies nearest-neighbour ring ZZ interactions using CNOT-RZ-CNOT.

The fixed angles are generated once from ReservoirConfig.seed. Reusing the same
ReservoirParameters preserves the reservoir map across training and inference.

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

    circuit = model.circuit([q0, q1, q2], [0.2, -0.4, 0.7])

The circuit builder imports PyQPanda3 lazily. Constructing the configuration,
fixed parameter set, measurement features, or ridge readout does not initialize
a quantum backend.

## Measurement contract

The default reservoir features are:

- every single-qubit Z expectation;
- every nearest-neighbour ring ZZ correlation.

For three qubits the exact order is:

    Z0, Z1, Z2, Z0Z1, Z1Z2, Z2Z0

Use model.observables or reservoir_observables() rather than hard-coding the
labels. Execute each sample circuit on the desired backend, measure those
expectations, then pass the rows to fit_measurements().

    z = [
        [0.72, -0.10, 0.31],
        [0.41,  0.22, 0.08],
        [-0.16, 0.63, 0.37],
        [-0.52, 0.28, 0.66],
    ]
    zz = [
        [0.21, -0.08, 0.12],
        [0.04,  0.13, 0.19],
        [-0.11, 0.25, -0.06],
        [0.18, -0.03, 0.30],
    ]
    y = [0.0, 0.3, 0.7, 1.0]

    model.fit_measurements(z, y, zz_expectations=zz)
    prediction = model.predict_measurements(
        [[0.05, 0.51, 0.42]],
        zz_expectations=[[0.09, 0.20, 0.02]],
    )

Measurement acquisition remains explicit and backend-owned. The module does not
pretend that a local statevector, cloud job, and real-device sampler expose the
same execution API.

## Readout

RidgeReadout uses a closed-form regularized least-squares solve with an
unregularized intercept. It accepts scalar or multi-output targets. The quantum
reservoir stays fixed while only these classical coefficients are learned.

This separation is useful for time-series regression, nonlinear system
identification, sensor streams, and other tasks where repeated quantum
parameter optimization would dominate the cost.

## Reproducibility and scope

ReservoirConfig.seed determines the fixed random angles. Persist the config and
ReservoirParameters when an experiment must be replayed exactly.

QuantumReservoir provides the algorithmic circuit, observable contract, feature
packing, and trainable classical readout. Shot allocation, error mitigation,
backend transpilation, and device calibration remain caller responsibilities.
