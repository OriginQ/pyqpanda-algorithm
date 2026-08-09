import time

import pytest
import numpy as np
from pyqpanda_alg.QSVD import SVD
from pyqpanda_alg.QSVD import QSVD as qsvd_module
import warnings




class Test_QSVD_SVD:

    def setup_method(self):
        warnings.filterwarnings("ignore")
        np.random.seed(42)

    def test_orthogonality_properties(self):
        matrix = np.random.random(16).reshape([4, 4])  # 使用方阵便于测试

        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()
        qeig = qsvd_instance.return_diag(para)
        q_singular_values = np.diag(qeig)

        u_np, s_np, v_np = np.linalg.svd(matrix)

        assert np.all(q_singular_values >= 0), "QSVD奇异值应该非负"
        assert np.all(s_np >= 0), "NumPy奇异值应该非负"

        q_sorted = np.sort(q_singular_values)[::-1]
        np_sorted = np.sort(s_np)[::-1]
        assert np.all(np.diff(q_sorted) <= 0), "QSVD奇异值应该降序排列"
        assert np.all(np.diff(np_sorted) <= 0), "NumPy奇异值应该降序排列"

    def test_singular_values_against_numpy(self):
        matrix = np.random.random(16).reshape([4, 4])

        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()
        q_singular = np.sort(np.diag(qsvd_instance.return_diag(para)))[::-1]

        np_singular = np.linalg.svd(matrix, compute_uv=False)

        rel_err = np.max(np.abs(q_singular - np_singular) / np_singular)
        assert rel_err < 1e-2, f"singular value relative error too large: {rel_err}"

    def test_singular_vectors_against_numpy(self):
        matrix = np.random.random(32).reshape([4, 8])

        qsvd_instance = SVD(matrix_in=matrix)
        para = qsvd_instance.QSVD_min()
        # loss() reports the index the dominant singular value ended up on
        max_index = qsvd_instance.loss(para, return_type=False)[1]
        left = qsvd_instance.max_eig('0', para, max_index)
        right = qsvd_instance.max_eig('1', para, max_index)

        u_np, s_np, v_np = np.linalg.svd(matrix)

        cos_left = abs(np.dot(left, u_np[:, 0])) / (np.linalg.norm(left) * np.linalg.norm(u_np[:, 0]))
        cos_right = abs(np.dot(right, v_np[0])) / (np.linalg.norm(right) * np.linalg.norm(v_np[0]))
        assert cos_left > 0.99, f"left singular vector overlap too low: {cos_left}"
        assert cos_right > 0.99, f"right singular vector overlap too low: {cos_right}"

    def test_loss_skips_state_vector_on_scalar_path(self, monkeypatch):
        # the scalar loss is the optimizer inner loop, it must not simulate the state vector
        built = []
        real_state_vector = qsvd_module.StateVector

        def counting_state_vector(*args, **kwargs):
            built.append(1)
            return real_state_vector(*args, **kwargs)

        monkeypatch.setattr(qsvd_module, 'StateVector', counting_state_vector)

        qsvd_instance = SVD(matrix_in=np.random.random(16).reshape([4, 4]))
        qsvd_instance.loss(qsvd_instance.parameter)
        assert len(built) == 0, "scalar loss built a state vector"

        qsvd_instance.loss(qsvd_instance.parameter, return_type=False)
        assert len(built) == 1, "matrix loss did not build a state vector"

    def test_qsvd_min_runtime(self):
        matrix = np.random.random(64).reshape([8, 8])

        start = time.perf_counter()
        SVD(matrix_in=matrix).QSVD_min()
        elapsed = time.perf_counter() - start

        assert elapsed < 20, f"QSVD_min on an 8x8 matrix took {elapsed:.1f}s"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])