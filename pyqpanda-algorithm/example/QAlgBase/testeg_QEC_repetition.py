"""Three-qubit repetition-code reference example."""

import numpy as np

from pyqpanda_alg.QEC import reference_repetition_recovery


logical = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)

for code in ("bit_flip", "phase_flip"):
    print(f"{code}:")
    for error_qubit in (None, 0, 1, 2):
        result = reference_repetition_recovery(
            code,
            logical_state=logical,
            error_qubit=error_qubit,
        )
        syndrome = int(np.argmax(result["syndrome_probabilities"]))
        print(
            f"  error={error_qubit!r} "
            f"fidelity={result['logical_fidelity']:.12f} "
            f"syndrome={syndrome}"
        )
