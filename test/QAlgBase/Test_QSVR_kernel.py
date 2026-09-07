"""Check kernel values independently and guard the simulator-call budget."""

import numpy as np
import pytest
from sklearn.svm import SVR

from pyqpanda_alg.QSVR import Quantum_SVR


def encoded_state(point):
    """Evaluate the two-qubit feature map with NumPy gate matrices only."""
    def rx(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    def ry(angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        return np.array([[c, -s], [s, c]])

    # q0 is the least significant bit. H on both qubits prepares |++>.
    state = np.ones(4, dtype=complex) / 2
    state = np.kron(ry(point[1]), ry(point[0])) @ state
    state = np.diag([1, 1, 1, -1]) @ state
    return np.kron(rx(point[0]), rx(point[1])) @ state


def reference_kernel(x, y):
    result = np.empty((len(x), len(y)))
    for i, first in enumerate(x):
        for j, second in enumerate(y):
            result[i, j] = abs(np.vdot(encoded_state(second),
                                       encoded_state(first))) ** 2
    return result


@pytest.fixture
def model():
    x = np.array([[-0.9, 0.1], [0.3, -0.4], [1.2, 0.8], [-0.2, 1.7]])
    return Quantum_SVR(x, np.array([0.1, -0.3, 1.4, 0.7]))


@pytest.mark.parametrize("case, expected_calls", [
    ("same_object", 10),
    ("equal_copy", 10),
    ("equal_lists", 10),
    ("duplicates", 10),
    ("permuted", 16),
    ("rectangular", 8),
    ("singleton", 1),
    ("empty_left", 0),
    ("empty_right", 0),
    ("empty_both", 0),
])
def test_qsvr_kernel_values_and_call_budget(model, monkeypatch, case, expected_calls):
    x = model.x.copy()
    if case == "same_object":
        y = x
    elif case == "equal_copy":
        y = x.copy()
    elif case == "equal_lists":
        x, y = x.tolist(), x.tolist()
    elif case == "duplicates":
        x[1] = x[0]
        y = x.copy()
    elif case == "permuted":
        y = x[::-1].copy()
    elif case == "rectangular":
        y = x[:2].copy()
    elif case == "singleton":
        x, y = x[:1], x[:1].copy()
    elif case == "empty_left":
        x, y = x[:0], x.copy()
    elif case == "empty_right":
        y = x[:0]
    else:
        x, y = x[:0], x[:0].copy()

    calls = []
    original_dist = model.dist

    def measured_dist(first, second):
        calls.append((first, second))
        return original_dist(first, second)

    monkeypatch.setattr(model, "dist", measured_dist)
    actual = model.k_kernel(x, y)
    np.testing.assert_allclose(actual, reference_kernel(x, y), atol=1e-12, rtol=1e-12)
    assert actual.shape == (len(x), len(y))
    assert len(calls) == expected_calls


def test_qsvr_predictions_match_independent_precomputed_kernel(model):
    expected_model = SVR(kernel="precomputed", gamma=0.1)
    kernel = reference_kernel(model.x, model.x)
    expected_model.fit(kernel, model.y)

    predicted, targets = model.get_res()

    # Gate-level roundoff can change libsvm's last step within its stopping
    # tolerance; matrix entries themselves are checked to 1e-12 above.
    np.testing.assert_allclose(predicted, expected_model.predict(kernel),
                               atol=2 * expected_model.tol, rtol=0)
    np.testing.assert_array_equal(targets, model.y)
