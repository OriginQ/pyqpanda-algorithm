import importlib.metadata
import pyqpanda_alg


def test_runtime_version_matches_distribution():
    assert pyqpanda_alg.__version__ == "2.1.0"
    assert pyqpanda_alg.__version__ == importlib.metadata.version("pyqpanda_alg")
