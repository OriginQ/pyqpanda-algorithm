# QMitigation

QMitigation adds two backend-independent error-mitigation building blocks that
can sit around CPUQVM, QCloud, or hardware execution.

## 1. Readout mitigation

Build a complete assignment matrix from basis-state calibration counts:

~~~python
from pyqpanda_alg.QMitigation import (
    assignment_matrix_from_calibration,
    mitigate_probabilities,
)

calibration = {
    "0": {"0": 950, "1": 50},
    "1": {"0": 80, "1": 920},
}
matrix = assignment_matrix_from_calibration(calibration)
corrected = mitigate_probabilities({"0": 540, "1": 460}, matrix)
~~~

For larger systems, factorized_assignment_matrix composes independently
calibrated 2x2 readout matrices in MSB-to-LSB bitstring order. The inverse is a
pseudoinverse with configurable rcond, followed by Euclidean probability-simplex
projection by default so the result remains a physical distribution.

## 2. Zero-noise extrapolation

Richardson extrapolation cancels low-order noise dependence without requiring
QMitigation to own a device-specific noise-scaling mechanism:

~~~python
from pyqpanda_alg.QMitigation import zero_noise_extrapolate

def run_at_scale(scale):
    # Build/fold/transpile a circuit for the requested noise scale, execute it,
    # and return the measured expectation.
    return 0.8 + 0.05 * scale

estimate = zero_noise_extrapolate(run_at_scale, [1.0, 3.0])
~~~

global_fold_sequence provides the standard U (U_dagger U)^n operation schedule
for positive odd integer scale factors when the caller can supply an operation
inverse callback. polynomial_extrapolate provides a weighted least-squares
alternative and reports fit coefficients plus residual norm.

The module deliberately separates mitigation mathematics from backend-specific
circuit and calibration acquisition. That lets the same correction and
extrapolation code consume local simulator, cloud, and hardware results.
