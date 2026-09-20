# Maximum-Likelihood Amplitude Estimation (MLAE)

This package implements amplitude estimation without a phase-estimation
register. It prepares the state, applies a schedule of Grover powers, records
target-state successes, and recovers the base amplitude with a global
maximum-likelihood fit.

For Grover power k, MLAE models the success probability as

    p_k(theta) = sin^2((2 k + 1) theta)

and returns a = sin^2(theta).

## CPUQVM example

    from pyqpanda3.core import QCircuit, RY
    from pyqpanda_alg.MLAE import MLAE

    def prepare(qubits):
        circuit = QCircuit()
        circuit << RY(qubits[0], 1.0471975512)
        return circuit

    estimator = MLAE(
        operator_in=prepare,
        qnumber=1,
        res_index=0,
        powers=(0, 1, 2, 4, 8),
        shots=256,
    )
    amplitude = estimator.run()

## Hardware/backend-independent decoding

If a backend already produced success counts, no CPUQVM dependency is needed
for the likelihood fit itself:

    amplitude = MLAE.estimate_from_counts(
        successes=(64, 252, 8, 32, 219),
        shots=256,
        powers=(0, 1, 2, 4, 8),
    )

Pass return_details=True to obtain theta, log likelihood, the schedule, counts,
and total oracle-query accounting.

Reference: Suzuki et al., "Amplitude Estimation without Phase Estimation",
Quantum Information Processing 19, 75 (2020).
