"""Runtime-contract tests for QmRMR estimator execution.

Plan 3 Task 2: Feature_Selection objective evaluations flow through
submit_estimate.  The recording backend replays canned expectations and
a uniform statevector so the full SPSA pipeline runs deterministically
without any QVM, and the call mix proves the objective is estimated
instead of sampled.
"""

import numpy as np

from pyqpanda_alg.QmRMR.QmRMR_core import Feature_Selection
from test.execution.fakes import RecordingBackend


def test_qmrmr_runtime_uses_estimator_without_sampling():
    linear = np.array([0.6, 0.4])
    quadratic = [[0.3, 0.1], [0.1, 0.2]]
    backend = RecordingBackend(
        expectations=[0.5, -0.25, -0.5],
        statevector=[[0.5, 0.5, 0.5, 0.5]],
    )
    model = Feature_Selection(quadratic, linear, 1)
    his, choice, dic = model.get_his_res([0.5, 0.5], backend=backend)
    assert backend.estimate_call_count > 0
    assert backend.sample_call_count == 0
    assert len(choice) == 2
    assert sum(choice) == 1
    assert his
