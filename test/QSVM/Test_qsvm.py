# -*-coding:utf-8-*-
import sys
from pathlib import Path
import pytest

sys.path.append((Path.cwd().parent.parent).__str__())

import os
import numpy as np
from pyqpanda_alg.QSVM import QuantumKernel_vqnet
from pyqpanda_alg import QSVM

data_path = QSVM.__path__[0]


def _read_vqc_qsvm_data(path):
    """读取量子SVM数据"""
    train_features = np.loadtxt(os.path.join(path, "dataset/qsvm_train_features.txt"))
    test_features = np.loadtxt(os.path.join(path, "dataset/qsvm_test_features.txt"))
    train_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_train_labels.txt"))
    test_labels = np.loadtxt(os.path.join(path, "dataset/qsvm_test_labels.txt"))
    samples = np.loadtxt(os.path.join(path, "dataset/qsvm_samples.txt"))
    return train_features, test_features, train_labels, test_labels, samples


def test_data_loading():
    """测试数据加载功能"""
    train_features, test_features, train_labels, test_labels, samples = _read_vqc_qsvm_data(data_path)

    assert train_features.shape[1] == 2
    assert test_features.shape[1] == 2
    assert len(train_labels.shape) == 1
    assert len(test_labels.shape) == 1

    assert np.all(
        train_labels == [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.all(test_labels == [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])


class TestQuantumKernelVqnet:
    """量子核矩阵的数值性质与实现约束。

    这些用例针对三处原有缺陷建立回归保护：

    1. 核矩阵带 1024 次采样的随机噪声（两次调用结果不同）；
    2. 对称矩阵经 ``np.linalg.eig`` 投影后出现复数 dtype、对角线不再为 1，
       甚至出现大于 1 的"概率"；
    3. 非对称调用（``y_vec`` 指定）时，若某测试点与训练点完全相同，该处的
       保真度会被跳过而留下 0，而不是 1。
    """

    @staticmethod
    def _samples(n=8, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, 2)) * 2 * np.pi

    def test_kernel_is_symmetric_with_unit_diagonal(self):
        x = self._samples()
        kernel = QuantumKernel_vqnet(n_qbits=2).evaluate(x_vec=x)

        assert kernel.shape == (len(x), len(x))
        assert np.allclose(kernel, kernel.T, atol=1e-12)
        assert np.allclose(np.diag(kernel), 1.0, atol=1e-12)

    def test_kernel_is_deterministic(self):
        """回归：去掉 measure 采样后，同一输入必须给出完全相同的核矩阵。"""
        x = self._samples()
        kernel_oracle = QuantumKernel_vqnet(n_qbits=2)

        first = kernel_oracle.evaluate(x_vec=x)
        second = kernel_oracle.evaluate(x_vec=x)

        assert np.array_equal(first, second)

    def test_kernel_values_are_real_and_bounded(self):
        """保真度必须是 [0, 1] 内的实数。"""
        x = self._samples()
        kernel = QuantumKernel_vqnet(n_qbits=2).evaluate(x_vec=x)

        assert not np.iscomplexobj(kernel)
        assert kernel.min() >= 0.0
        assert kernel.max() <= 1.0

    def test_asymmetric_kernel_matches_identical_rows(self):
        """回归：非对称分支下相同的样本对必须得到保真度 1。"""
        x = self._samples()
        kernel = QuantumKernel_vqnet(n_qbits=2).evaluate(x_vec=x[:4], y_vec=x[:6])

        assert kernel.shape == (4, 6)
        for i in range(4):
            assert kernel[i, i] == pytest.approx(1.0, abs=1e-12)

    def test_unsupported_n_qbits_raises(self):
        """回归：n_qbits != 2 曾静默给出错误的核矩阵，现在必须显式报错。"""
        x = self._samples(3)
        with pytest.raises(ValueError):
            QuantumKernel_vqnet(n_qbits=3).evaluate(x_vec=x)
        with pytest.raises(ValueError):
            QuantumKernel_vqnet().evaluate(x_vec=x)

    def test_single_simulator_per_matrix(self, monkeypatch):
        """回归：整个核矩阵只应构造 1 个 CPUQVM，而不是每个矩阵元一个。"""
        import pyqpanda3.core as core

        module = sys.modules["pyqpanda_alg.QSVM.quantum_kernel_svm"]
        created = []

        class _CountingCPUQVM(core.CPUQVM):
            def __init__(self, *args, **kwargs):
                created.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(module, "CPUQVM", _CountingCPUQVM)
        QuantumKernel_vqnet(n_qbits=2).evaluate(x_vec=self._samples())
        assert len(created) == 1

    def test_usable_as_sklearn_kernel(self):
        """核矩阵仍可直接作为 ``sklearn.svm.SVC`` 的 kernel 回调使用。"""
        from sklearn.svm import SVC

        train_features, test_features, train_labels, test_labels, _ = _read_vqc_qsvm_data(data_path)
        svc = SVC(kernel=QuantumKernel_vqnet(n_qbits=2).evaluate)
        svc.fit(train_features, train_labels)

        assert svc.score(test_features, test_labels) >= 0.5

