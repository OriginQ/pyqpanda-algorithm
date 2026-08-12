"""Runtime-contract tests for QmRMR estimator execution.

Plan 3 Task 2: Feature_Selection objective evaluations flow through
submit_estimate.  The recording backend replays canned expectations and
a uniform statevector so the full SPSA pipeline runs deterministically
without any QVM, and the call mix proves the objective is estimated
instead of sampled.
"""

import numpy as np
import pytest

from pyqpanda_alg.QmRMR.QmRMR_core import Feature_Selection
from pyqpanda_alg.execution import BackendCapabilities
from test.execution.fakes import RecordingBackend


def test_qmrmr_runtime_uses_estimator_without_sampling():
    linear = np.array([0.6, 0.4])
    quadratic = [[0.3, 0.1], [0.1, 0.2]]
    backend = RecordingBackend(
        expectations=[0.5, -0.25, -0.5],
        sample_counts=[{"00": 250, "01": 250, "10": 250, "11": 250}],
    )
    backend.capabilities = BackendCapabilities(statevector=False)
    model = Feature_Selection(quadratic, linear, 1)
    his, choice, dic = model.get_his_res([0.5, 0.5], backend=backend)
    assert backend.estimate_call_count > 0
    assert backend.sample_call_count == 1
    assert backend.statevector_call_count == 0
    assert len(choice) == 2
    assert sum(choice) == 1
    assert his


def test_qmrmr_objective_sign_pins_min_redundancy_minus_relevance():
    """The objective is min E[x^T Q x] - E[x . l] (redundancy - relevance).

    Standard QmRMR/mRMR form: the linear coefficients are relevance weights
    (higher = more preferred), so relevance SUBTRACTS from the loss.  Hand
    computation for linear=[0.6, 0.4], quadratic=[[0.3, 0.1], [0.1, 0.2]]:

        x       x^T Q x           x . l       x^T Q x - x . l
        [0, 0]  0.0               0.0         0.0
        [1, 0]  0.3               0.6         -0.3
        [0, 1]  0.2               0.4         -0.2
        [1, 1]  0.3+0.1+0.1+0.2  1.0          -0.3
                = 0.7

    Bit strings are most-significant first and feature p maps to qubit
    (m - 1 - p), so the observable's basis-state expectations are
    {00: 0, 01: -0.2, 10: -0.3, 11: -0.3}.
    """
    linear = [0.6, 0.4]
    quadratic = [[0.3, 0.1], [0.1, 0.2]]
    model = Feature_Selection(quadratic, linear, 1)

    # Independent classical evaluation of the objective per basis state:
    # objective(x) = x^T Q x - x . l.
    def objective(x):
        x_quad = sum(
            quadratic[row][col] * x[row] * x[col]
            for row in range(2)
            for col in range(2)
        )
        relevance = sum(linear[p] * x[p] for p in range(2))
        return x_quad - relevance

    # Each bit string is MSB-first (leftmost bit = feature 0), so "01"
    # denotes feature set [0, 1] and "10" denotes [1, 0].
    expected = {
        "00": objective([0, 0]),
        "01": objective([0, 1]),
        "10": objective([1, 0]),
        "11": objective([1, 1]),
    }
    hand_values = {"00": 0.0, "01": -0.2, "10": -0.3, "11": -0.3}
    for bits, value in expected.items():
        assert value == pytest.approx(hand_values[bits], abs=1e-12)

    # The observable's diagonal must match those hand-computed values.
    # The pipeline's bit string for basis state |i> is the MSB-first binary
    # of the matrix row index (format(i, '0%db')), so the index is the
    # bit string itself -- no flip.
    matrix = model._observable.matrix()
    for bits, value in expected.items():
        index = int(bits, 2)
        assert matrix[index, index].real == pytest.approx(value, abs=1e-12)

    # The estimator returns the hand-computed value for the basis state the
    # zero-parameter circuit prepares (qubit 0 in |1>, pipeline bit string
    # "01", i.e. feature 1 selected).
    backend = RecordingBackend(expectations=[expected["01"]])
    model._backend = backend
    assert model.cal_loss([0.0]) == pytest.approx(expected["01"])
