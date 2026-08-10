"""Regression guards for the public top-level import surface."""

import importlib.util

import pytest

import pyqpanda_alg


def test_top_level_package_imports_cleanly():
    """The top-level package and all of its submodules must import without error."""
    assert hasattr(pyqpanda_alg, "QAOA")
    assert hasattr(pyqpanda_alg, "QmRMR")


def test_extensions_not_present_in_package():
    """The binary-only extensions package must no longer be findable anywhere."""
    assert importlib.util.find_spec("pyqpanda_alg.extensions") is None


def test_extensions_not_importable():
    """Importing the removed extensions package must fail."""
    with pytest.raises(ImportError):
        import pyqpanda_alg.extensions  # noqa: F401
