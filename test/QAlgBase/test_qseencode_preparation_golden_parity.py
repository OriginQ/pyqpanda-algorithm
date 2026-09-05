"""Copied constants only; Frozen evidence is never imported at runtime.

Provenance:
- phase3a_preflight.py SHA-256
  B7252BF0D52E22250AB6BB973EED9EA366ED2D3ADB8B0D435DDB49CADDF9641B
- final_technical_hardening.py SHA-256
  B24336428BD62DFF3C479B5C1207329BC61AA68B92009E77A2EE6537981D11A8
- case semantics: small-N sparse_isometry and DS compatible state preparation,
  output-register-aware logical fidelity.
"""

import numpy as np

from pyqpanda_alg.QSEncode import PreparationMethod
from pyqpanda_alg.QSEncode._capability import assess_capability
from pyqpanda_alg.QSEncode._preparation import adapt_preparation_input


FROZEN_COMPATIBLE_STATE = np.array(
    [1 / np.sqrt(3), 0, 0, 1j / np.sqrt(3), 0, -1 / np.sqrt(3), 0, 0]
)


def test_frozen_sparse_isometry_compatible_case_semantics():
    assessment = assess_capability(
        PreparationMethod.SPARSE_ISOMETRY,
        adapt_preparation_input(FROZEN_COMPATIBLE_STATE),
    )
    assert assessment.report.compatible
    assert assessment.report.observed_output_qubits == (0, 1, 2)
    assert assessment.report.ancillas == ()
    assert assessment.report.logical_fidelity >= 1.0 - 1e-10


def test_frozen_ds_compatible_case_semantics():
    assessment = assess_capability(
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
        adapt_preparation_input(FROZEN_COMPATIBLE_STATE),
    )
    assert assessment.report.compatible
    assert assessment.report.observed_output_qubits == (3, 4, 5)
    assert assessment.report.ancillas == (0, 1, 2)
    assert assessment.report.required_qubits == 6
    assert assessment.report.logical_fidelity >= 1.0 - 1e-10
