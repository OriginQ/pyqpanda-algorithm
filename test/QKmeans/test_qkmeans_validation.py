import importlib

import numpy as np
import pytest

qkmeans_module = importlib.import_module("pyqpanda_alg.QKmeans.QuantumKmeans")
QuantumKmeans = qkmeans_module.QuantumKmeans


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_fit_rejects_non_positive_or_non_integer_cluster_counts(k):
    data = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        QuantumKmeans(k=k).fit(data)


def test_fit_rejects_more_clusters_than_samples_before_quantum_distance(monkeypatch):
    data = np.array([[0.0, 0.0], [1.0, 1.0]])

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("quantum distance must not run for an impossible cluster count")

    monkeypatch.setattr(qkmeans_module, "_point_centroid_distances", fail_if_called)

    with pytest.raises(
        ValueError,
        match=r"k \(3\) cannot exceed the number of samples \(2\)",
    ):
        QuantumKmeans(k=3).fit(data)
