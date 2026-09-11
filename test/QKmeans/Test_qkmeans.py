# -*-coding:utf-8-*-
"""QKmeans 单元测试。

本模块此前没有任何单元测试。覆盖点：

* 距离电路返回归一化的 ancilla 概率；
* 去掉 ``measure`` 之后结果**精确且可复现**（原实现对 1024 次采样取值）；
* 显式传入的模拟器被复用，``fit`` 全程只构造 1 个 ``CPUQVM``
  （原实现每个"样本 × 质心"组合都新建一个）；
* ``fit`` 的聚类输出形状、簇数正确，固定随机种子下可复现。

.. note::

    ``pyqpanda_alg.QKmeans.QuantumKmeans`` 这个属性被 ``__init__.py`` 导出的
    同名**类**遮蔽了，因此这里通过 ``sys.modules`` 取真正的子模块。
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append((Path.cwd().parent.parent).__str__())

import pyqpanda_alg.QKmeans  # noqa: F401  确保子模块已被导入
from pyqpanda_alg.QKmeans import QuantumKmeans

_qk_module = sys.modules["pyqpanda_alg.QKmeans.QuantumKmeans"]


def _make_two_blobs(seed=0, per_cluster=4):
    """构造两团线性可分的数据，避免聚类过程中出现空簇。"""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal(-1.0, 0.1, size=(per_cluster, 2)),
            rng.normal(1.0, 0.1, size=(per_cluster, 2)),
        ]
    )


class TestQuantumKmeansCircuit:
    """底层 swap-test 距离电路。"""

    def test_ancilla_probabilities_are_normalised(self):
        """ancilla 的 0/1 概率之和应为 1。"""
        result = _qk_module._QuantumKmeansCircuit(1.0, 1.0, 2.0, 2.0)
        assert set(result.keys()) == {"0", "1"}
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_are_exact_not_sampled(self):
        """同一输入多次求值必须完全一致。

        原实现带 ``measure`` + 1024 shots，返回的是 ``k/1024`` 形式的采样值，
        边界样本上的簇归属会因此抖动。
        """
        first = _qk_module._QuantumKmeansCircuit(1.0, 1.0, 2.0, 2.0)
        for _ in range(3):
            assert _qk_module._QuantumKmeansCircuit(1.0, 1.0, 2.0, 2.0) == first

    def test_identical_encoded_states_have_zero_ancilla_probability(self):
        """两个参数完全相同的态，测量到 ancilla=1 的概率应为 0。"""
        result = _qk_module._QuantumKmeansCircuit(1.3, 0.4, 1.3, 0.4)
        assert result["1"] == pytest.approx(0.0, abs=1e-9)

    def test_supplied_machine_is_reused(self, monkeypatch):
        """传入模拟器后不应再新建 ``CPUQVM``。"""
        import pyqpanda3.core as core

        created = []

        class _CountingCPUQVM(core.CPUQVM):
            def __init__(self, *args, **kwargs):
                created.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(_qk_module, "CPUQVM", _CountingCPUQVM)
        machine = core.CPUQVM()
        _qk_module._QuantumKmeansCircuit(1.0, 1.0, 2.0, 2.0, machine)
        assert created == []


class TestPointCentroidDistances:

    def test_one_distance_per_centroid(self):
        centroids = np.array([[0.5, 0.5], [-0.5, -0.5]])
        values = _qk_module._point_centroid_distances(np.array([0.2, 0.3]), centroids, 2)
        assert len(values) == 2
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_reproducible(self):
        centroids = np.array([[0.5, 0.5], [-0.5, -0.5]])
        first = _qk_module._point_centroid_distances(np.array([0.2, 0.3]), centroids, 2)
        second = _qk_module._point_centroid_distances(np.array([0.2, 0.3]), centroids, 2)
        assert first == second


class TestQuantumKmeansFit:

    def test_fit_two_blobs(self):
        data = _make_two_blobs()
        np.random.seed(7)
        centers, clusters = QuantumKmeans(k=2).fit(data)

        assert np.asarray(centers).shape == (2, 2)
        assert len(clusters) == len(data)
        assert set(np.unique(clusters).tolist()).issubset({0, 1})
        assert np.all(np.isfinite(centers))

    def test_fit_is_reproducible_for_fixed_seed(self):
        data = _make_two_blobs()

        np.random.seed(7)
        centers_a, clusters_a = QuantumKmeans(k=2).fit(data)
        np.random.seed(7)
        centers_b, clusters_b = QuantumKmeans(k=2).fit(data)

        assert np.array_equal(clusters_a, clusters_b)
        assert np.allclose(centers_a, centers_b, atol=1e-12)

    def test_fit_uses_a_single_simulator(self, monkeypatch):
        """整个 ``fit`` 只应构造 1 个模拟器。"""
        import pyqpanda3.core as core

        created = []

        class _CountingCPUQVM(core.CPUQVM):
            def __init__(self, *args, **kwargs):
                created.append(1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(_qk_module, "CPUQVM", _CountingCPUQVM)
        np.random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            QuantumKmeans(k=2).fit(_make_two_blobs())
        assert len(created) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
