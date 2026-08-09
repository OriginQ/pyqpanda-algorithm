# -*-coding:utf-8-*-
import sys
from pathlib import Path
import pytest
import numpy as np

# 添加项目路径到系统路径
sys.path.append((Path.cwd().parent.parent).__str__())


def _classical_pca(x):
    """Reference PCA: eigendecomposition of the covariance matrix, descending."""
    norm_x = x - x.mean(axis=0)
    cov = np.dot(norm_x.T, norm_x)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return norm_x, w[order], v[:, order]


class TestQPCA:
    """QPCA测试类"""

    @pytest.fixture
    def sample_data(self):
        """提供标准测试数据"""
        return np.array([[-1, 2], [-2, -1], [-1, -2], [1, 3], [2, 1], [3, 2]])

    @pytest.fixture
    def sample_data_4d(self):
        rng = np.random.RandomState(0)
        return rng.randn(20, 4)

    def test_qpca_normal(self, sample_data):
        """测试QPCA正常功能"""
        from pyqpanda_alg.QPCA import qpca

        # 执行QPCA
        data_q = qpca(sample_data, 1)

        assert data_q.shape == (6,1)
        assert isinstance(data_q, np.ndarray)

    def test_qpca_k1_matches_classical_pca(self, sample_data):
        from pyqpanda_alg.QPCA import qpca

        data_q = qpca(sample_data, 1)[:, 0]
        norm_x, _, v = _classical_pca(sample_data)
        expected = np.dot(norm_x, v[:, 0])
        if np.dot(data_q, expected) < 0:
            expected = -expected

        # sampling noise on 8192 shots keeps this loose
        assert np.corrcoef(data_q, expected)[0, 1] > 0.999
        assert np.max(np.abs(data_q - expected)) < 0.15

    def test_qpca_k1_direction_is_unit_norm(self, sample_data):
        """Holds for this fixture, not in general: a degenerate spectrum leaves
        the post-selection with a single outcome and the norm comes out at
        sqrt(2). Pinned to the documented example on purpose."""
        from pyqpanda_alg.QPCA import qpca

        norm_x, _, _ = _classical_pca(sample_data)
        data_q = qpca(sample_data, 1)
        # recover the projection direction the circuit produced
        vec = np.linalg.lstsq(norm_x, data_q, rcond=None)[0][:, 0]
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    def test_qpca_k2_returns_pca_scores(self, sample_data):
        from pyqpanda_alg.QPCA import qpca

        data_q = qpca(sample_data, 2)
        norm_x, w, v = _classical_pca(sample_data)

        assert isinstance(data_q, np.ndarray)
        assert data_q.shape == (6, 2)

        # PCA scores are decorrelated and carry the eigenvalues as their
        # column variances, in descending order. neither holds for the
        # centred input, so this pins the basis and not just the subspace.
        cov_q = np.dot(data_q.T, data_q)
        assert abs(cov_q[0, 1]) < 1e-9
        assert np.allclose(np.diag(cov_q), w)

        expected = np.dot(norm_x, v)
        for j in range(2):
            sign = np.sign(np.dot(data_q[:, j], expected[:, j]))
            assert np.allclose(data_q[:, j], sign * expected[:, j])

    @pytest.mark.parametrize('k', [0, 3, -1])
    def test_qpca_invalid_k_raises(self, sample_data, k):
        from pyqpanda_alg.QPCA import qpca

        with pytest.raises(ValueError):
            qpca(sample_data, k)

    def test_qpca_four_features_raises(self, sample_data_4d):
        from pyqpanda_alg.QPCA import qpca

        with pytest.raises(NotImplementedError):
            qpca(sample_data_4d, 1)

        with pytest.raises(NotImplementedError):
            qpca(sample_data_4d, 2)
