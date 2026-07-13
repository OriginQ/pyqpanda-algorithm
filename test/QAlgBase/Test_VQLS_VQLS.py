import pytest
import numpy as np
from pyqpanda_alg.VQLS import VQLS
import warnings


def _classical_unit_solution(A, b):
    x = np.linalg.solve(A, b)
    x = x / np.linalg.norm(x)
    pivot = np.argmax(np.abs(x))
    return x * np.sign(x[pivot])


def _fidelity(x, y):
    return float(np.abs(np.vdot(x, y)) ** 2)


class Test_VQLS_VQLS:

    def setup_method(self):
        warnings.filterwarnings("ignore")

    def test_2x2_textbook(self):
        A = np.array([[1.0, -1.0 / 3], [-1.0 / 3, 1.0]])
        b = np.array([0.0, 1.0])
        x_q = np.array(VQLS(A, b, seed=0).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99, "VQLS solution should match classical solver"

    def test_2x2_diagonal(self):
        A = np.diag([2.0, 1.0])
        b = np.array([1.0, 1.0])
        x_q = np.array(VQLS(A, b, seed=0).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99

    def test_4x4_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0]).astype(float)
        b = np.array([1.0, 1.0, 1.0, 1.0])
        x_q = np.array(VQLS(A, b, seed=0).run(), dtype=complex)
        x_c = _classical_unit_solution(A, b)
        assert _fidelity(x_q, x_c) > 0.99

    def test_final_cost_converged(self):
        A = np.diag([2.0, 1.0])
        b = np.array([1.0, 1.0])
        solver = VQLS(A, b, seed=0)
        solver.run()
        assert 0.0 <= solver.final_cost < 1e-2
        assert solver.converged

    def test_non_hermitian_raises(self):
        A = np.array([[1.0, 2.0], [0.0, 1.0]])  # not Hermitian
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            VQLS(A, b, seed=0)

    def test_non_power_of_two_raises(self):
        A = np.eye(3)
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            VQLS(A, b, seed=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
