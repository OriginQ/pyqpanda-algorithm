import numpy as np
import pytest

from pyqpanda_alg.QSEncode import PreparationMethod
from pyqpanda_alg.QSEncode._capability import assess_capability
from pyqpanda_alg.QSEncode._preparation import adapt_preparation_input


FIXTURES = (
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)]),
    np.array([1 / np.sqrt(2), 0.0, 0.0, 1j / np.sqrt(2)]),
    np.array([1 / np.sqrt(3), 0, 0, 1j / np.sqrt(3), 0, -1 / np.sqrt(3), 0, 0]),
)


@pytest.mark.parametrize("method", list(PreparationMethod))
@pytest.mark.parametrize("state", FIXTURES)
def test_small_n_logical_state_fidelity_is_global_phase_safe(method, state):
    assessment = assess_capability(method, adapt_preparation_input(state))

    assert assessment.report.compatible is True
    assert assessment.report.reason_code == "compatible"
    assert assessment.report.logical_fidelity >= 1.0 - 1e-10
    assert assessment.report.failure_stage is None


def test_ds_reports_output_and_ancilla_semantics_from_actual_build():
    assessment = assess_capability(
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
        adapt_preparation_input([1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2)]),
    )
    report = assessment.report

    assert report.observed_output_qubits == (2, 3)
    assert report.ancillas == (0, 1)
    assert report.required_qubits == 4
    assert set(report.observed_output_qubits).isdisjoint(report.ancillas)
