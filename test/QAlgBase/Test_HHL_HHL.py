import pytest
import numpy as np
from pyqpanda_alg.HHL import HHL
import warnings


def _classical_unit_solution(A, b):
    x = np.linalg.solve(A, b)
    x = x / np.linalg.norm(x)
    pivot = np.argmax(np.abs(x))
    return x * np.sign(x[pivot])


def _fidelity(x, y):
    return float(np.abs(np.vdot(x, y)) ** 2)


class Test_HHL_HHL:

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_2x2_textbook(self):
        A = np.array([[1.0, -1.0 / 3], [-1.0 / 3, 1.0]])
        b = np.array([0.0, 1.0])
        x_q = np.array(HHL(A, b, clock_qubits=2).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99, "HHL solution should match classical solver"

    def test_2x2_diagonal(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 1.0])
        x_q = np.array(HHL(A, b, clock_qubits=3).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99

    def test_4x4_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(float)
        b = np.array([1.0, 1.0, 1.0, 1.0])
        x_q = np.array(HHL(A, b, clock_qubits=4).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99

    def test_success_probability_is_valid(self):
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 1.0])
        solver = HHL(A, b, clock_qubits=3)
        solver.run()
        assert 0.0 < solver.success_prob <= 1.0, "success probability must lie in (0, 1]"

    def test_non_hermitian_raises(self):
        A = np.array([[1.0, 2.0], [0.0, 1.0]])  # not Hermitian
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            HHL(A, b, clock_qubits=2)

    def test_non_power_of_two_raises(self):
        A = np.eye(3)
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            HHL(A, b, clock_qubits=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
