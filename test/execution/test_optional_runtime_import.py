"""Missing-extra tests: ``pyqpanda_alg`` stays importable without the
qpanda3-runtime dependency.

The runtime adapter is optional and imported lazily, so an environment
without the ``runtime`` extra must still import ``pyqpanda_alg``; only
constructing :class:`QPandaRuntimeBackend` raises
:class:`MissingRuntimeDependencyError` with the documented install
command.
"""

import importlib
import sys

import pytest

from pyqpanda_alg.execution import (
    MissingRuntimeDependencyError,
    QPandaRuntimeBackend,
)

_INSTALL_HINT = r"pip install pyqpanda-algorithm\[runtime\]"


@pytest.mark.runtime_contract
def test_importing_pyqpanda_alg_succeeds_without_runtime_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "qpanda3_runtime", None)
    assert importlib.import_module("pyqpanda_alg")
    assert importlib.import_module("pyqpanda_alg.execution")


@pytest.mark.runtime_contract
def test_runtime_backend_construction_requires_the_runtime_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "qpanda3_runtime", None)
    with pytest.raises(MissingRuntimeDependencyError, match=_INSTALL_HINT):
        QPandaRuntimeBackend(object(), object())
