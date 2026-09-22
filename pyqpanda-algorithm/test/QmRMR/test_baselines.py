import numpy as np

from pyqpanda_alg.QmRMR.constrained import QmRMRProblem, greedy_forward_selection


def test_greedy_forward_selection_returns_feasible_state():
    problem = QmRMRProblem(np.zeros((4, 4)), [-4.0, -3.0, -2.0, -1.0], 2)
    result = greedy_forward_selection(problem)
    assert result.state == (1, 1, 0, 0)
    np.testing.assert_allclose(result.value, -7.0, atol=1e-12)
    assert result.evaluations == 7

