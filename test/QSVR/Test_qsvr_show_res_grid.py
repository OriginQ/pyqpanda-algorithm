import importlib

import numpy as np


qsvr_module = importlib.import_module("pyqpanda_alg.QSVR.QSVR")
Quantum_SVR = qsvr_module.Quantum_SVR


class _FakeAxes:
    def scatter(self, *args, **kwargs):
        pass

    def plot_surface(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_zlabel(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass


class _FakeFigure:
    def add_subplot(self, *args, **kwargs):
        return _FakeAxes()


def test_show_res_uses_each_feature_domain_for_prediction_grid(monkeypatch):
    prediction_grid = {}

    class _FakeSVR:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            prediction_grid["x"] = np.asarray(x)
            return np.zeros(len(x))

    monkeypatch.setattr(qsvr_module, "SVR", _FakeSVR)
    monkeypatch.setattr(qsvr_module.plt, "figure", lambda **kwargs: _FakeFigure())
    monkeypatch.setattr(qsvr_module.plt, "show", lambda: None)

    qsvr = Quantum_SVR.__new__(Quantum_SVR)
    qsvr.x = np.array(
        [
            [-3.0, 10.0],
            [2.0, 40.0],
            [7.0, 25.0],
        ]
    )
    qsvr.y = np.array([0.0, 1.0, 2.0])

    qsvr.show_res()

    grid = prediction_grid["x"]
    assert grid.shape == (30 * 30, 2)
    assert np.min(grid[:, 0]) == np.min(qsvr.x[:, 0])
    assert np.max(grid[:, 0]) == np.max(qsvr.x[:, 0])
    assert np.min(grid[:, 1]) == np.min(qsvr.x[:, 1])
    assert np.max(grid[:, 1]) == np.max(qsvr.x[:, 1])
