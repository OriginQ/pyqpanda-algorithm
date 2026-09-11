# -*-coding:utf-8-*-
"""QSVR 单元测试。

除原有的接口冒烟测试外，这里补充核矩阵的数值性质与实现约束的回归用例：

* 核矩阵对称、对角线为 1、取值落在 [0, 1]；
* 同一输入可完全复现；
* 整个核矩阵只构造 1 个 ``CPUQVM``（原实现每个矩阵元新建一个）；
* ``show_res`` 的绘图网格取自各自维度的取值范围。
"""
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能运行

import numpy as np
import pytest

sys.path.append((Path.cwd().parent.parent).__str__())

from pyqpanda_alg.QSVR import Quantum_SVR


class Test_class_qsvr_Quantum_SVR:

    def setup_method(self):
        """测试方法前置设置"""
        warnings.filterwarnings("ignore")
        np.random.seed(42)  # 设置随机种子保证结果可重现
        os.makedirs('test_outputs', exist_ok=True)

    def test_interface11_show_res_basic(self):
        print("\n" + "=" * 60)
        print("测试接口11: Quantum_SVR.show_res() - 基本功能")
        print("=" * 60)

        # 创建测试数据（与示例相同）
        n_samples = 100
        n_features = 2
        X = np.random.rand(n_samples, n_features) * 10
        y = 2 * np.sin(X[:, 0]) + 1.5 * np.cos(X[:, 1])

        # 创建Quantum_SVR实例
        qsvr = Quantum_SVR(X, y)
        assert qsvr is not None, "Quantum_SVR实例创建失败"
        print("✓ Quantum_SVR实例创建成功")

        # 调用show_res()方法
        try:
            result = qsvr.show_res()
            print(result)

        except Exception as e:
            pytest.fail(f"show_res()方法执行失败: {e}")


class TestQuantumSVRKernel:
    """量子核矩阵的数值性质。"""

    @staticmethod
    def _data(n_samples=8, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.random((n_samples, 2)) * 10
        y = 2 * np.sin(x[:, 0]) + 1.5 * np.cos(x[:, 1])
        return Quantum_SVR(x, y)

    def test_kernel_is_symmetric_with_unit_diagonal(self):
        qsvr = self._data()
        kernel = qsvr.k_kernel(qsvr.x, qsvr.x)

        assert kernel.shape == (len(qsvr.x), len(qsvr.x))
        assert np.array_equal(kernel, kernel.T)
        assert np.array_equal(np.diag(kernel), np.ones(len(qsvr.x)))

    def test_kernel_is_deterministic(self):
        """同一输入两次构建必须给出完全相同的核矩阵。"""
        qsvr = self._data()
        first = qsvr.k_kernel(qsvr.x, qsvr.x)
        second = qsvr.k_kernel(qsvr.x, qsvr.x)

        assert np.array_equal(first, second)

    def test_kernel_values_are_bounded(self):
        qsvr = self._data()
        kernel = qsvr.k_kernel(qsvr.x, qsvr.x)

        assert kernel.min() >= 0.0
        assert kernel.max() <= 1.0

    def test_dist_is_symmetric_and_self_fidelity_is_one(self):
        qsvr = self._data()
        a, b = qsvr.x[0], qsvr.x[1]

        assert qsvr.dist(a, b) == pytest.approx(qsvr.dist(b, a), abs=1e-12)
        assert qsvr.dist(a, a) == pytest.approx(1.0, abs=1e-12)

    def test_single_simulator_per_kernel_matrix(self, monkeypatch):
        """回归：整个核矩阵只应构造 1 个 CPUQVM，而不是每个矩阵元一个。"""
        import sys as _sys

        import pyqpanda3.core as core

        module = _sys.modules["pyqpanda_alg.QSVR.QSVR"]
        created = []

        class _CountingCPUQVM(core.CPUQVM):
            def __init__(self, *args, **kwargs):
                created.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(module, "CPUQVM", _CountingCPUQVM)
        qsvr = self._data()
        qsvr.k_kernel(qsvr.x, qsvr.x)
        assert len(created) == 1

    def test_get_res_returns_prediction_for_every_sample(self):
        qsvr = self._data()
        predicted, actual = qsvr.get_res()

        assert predicted.shape == actual.shape
        assert np.all(np.isfinite(predicted))

    def test_show_res_grid_covers_each_axis_range(self, monkeypatch):
        """回归：绘图网格两个维度应各自取自本维度的取值范围。

        原实现两个 ``np.linspace`` 都用 ``min(x[:, 0])`` 与 ``max(x[:, 1])``，
        导致第 0 维的上界取自第 1 维。
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        captured = {}

        def _fake_plot_surface(self, x_grid, y_grid, z, *args, **kwargs):
            captured["x"] = np.asarray(x_grid)
            captured["y"] = np.asarray(y_grid)
            return None

        monkeypatch.setattr(Axes3D, "plot_surface", _fake_plot_surface)
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)

        qsvr = self._data(n_samples=6)
        qsvr.show_res()

        assert captured["x"].min() == pytest.approx(min(qsvr.x[:, 0]))
        assert captured["x"].max() == pytest.approx(max(qsvr.x[:, 0]))
        assert captured["y"].min() == pytest.approx(min(qsvr.x[:, 1]))
        assert captured["y"].max() == pytest.approx(max(qsvr.x[:, 1]))

