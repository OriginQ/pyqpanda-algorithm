# -*-coding:utf-8-*-
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.append((Path.cwd().parent.parent).__str__())


class TestQPCA:

    @pytest.fixture
    def sample_data(self):
        return np.array([[-1, 2], [-2, -1], [-1, -2], [1, 3], [2, 1], [3, 2]])

    def test_qpca_normal(self, sample_data):
        from pyqpanda_alg.QPCA import qpca

        data_q = qpca(sample_data, 1)

        assert data_q.shape == (6,1)
        assert isinstance(data_q, np.ndarray)



