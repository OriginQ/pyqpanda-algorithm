import pytest
import numpy as np
import os
from pyqpanda_alg.QSVR import Quantum_SVR
from pyqpanda_alg.QSVR import QSVR as qsvr_module
import warnings
import os


class _RecordingSVR:
    def __init__(self):
        self.seen = []

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X)
        self.seen.append(X)
        return np.zeros(len(X))


class _FakeAxes:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _FakeFigure:
    def add_subplot(self, *args, **kwargs):
        return _FakeAxes()


def _grid_points(monkeypatch, qsvr):
    svr = _RecordingSVR()
    monkeypatch.setattr(qsvr_module, "SVR", lambda **kwargs: svr)
    monkeypatch.setattr(qsvr_module.plt, "figure", lambda *args, **kwargs: _FakeFigure())
    monkeypatch.setattr(qsvr_module.plt, "show", lambda *args, **kwargs: None)
    qsvr.show_res()
    return svr.seen[0]


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

    def test_show_res_grid_matches_column_ranges(self, monkeypatch):
        t = np.random.randn(60)
        X = np.column_stack([t + 0.05 * np.random.randn(60), 20 * t + np.random.randn(60)])
        qsvr = Quantum_SVR(X, np.sin(t))

        points = _grid_points(monkeypatch, qsvr)

        assert points.shape == (900, 2)
        for col in (0, 1):
            axis = np.unique(points[:, col])
            assert len(axis) == 30
            expected = np.linspace(qsvr.x[:, col].min(), qsvr.x[:, col].max(), 30)
            assert np.allclose(axis, expected)

    def test_show_res_grid_axes_are_independent(self, monkeypatch):
        x = np.column_stack([np.linspace(0, 2, 40), np.linspace(10, 30, 40)])
        qsvr = Quantum_SVR(x, np.zeros(40))
        # bypass the scaler/PCA so the two columns keep exact, unmistakable ranges
        qsvr.x = x

        points = _grid_points(monkeypatch, qsvr)

        assert np.allclose([points[:, 0].min(), points[:, 0].max()], [0.0, 2.0])
        assert np.allclose([points[:, 1].min(), points[:, 1].max()], [10.0, 30.0])


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])