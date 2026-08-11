"""Runtime-contract tests for QUBO_QAOA estimator execution.

Plan 3 Task 2: QUBO_QAOA delegates to the migrated QAOA estimator API.
The recording backend replays canned expectations so the whole solver
pipeline runs deterministically without any QVM, and the call mix
proves the objective is estimated instead of sampled.
"""

import sympy as sp

from pyqpanda_alg.QUBO.QUBO import QUBO_QAOA
from test.execution.fakes import RecordingBackend


def test_qubo_qaoa_runtime_uses_estimator_without_sampling():
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
    backend = RecordingBackend(expectations=[0.5, -0.25, -0.5])
    model = QUBO_QAOA(function)
    result = model.run(
        layer=1, optimizer_option={"options": {"maxiter": 1}}, backend=backend
    )
    assert backend.estimate_call_count > 0
    assert backend.sample_call_count == 0
    assert isinstance(result, dict)
