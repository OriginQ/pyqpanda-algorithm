"""Repository-wide fixtures shared by the execution-layer test suites.

These fixtures are consumed by the execution plan's tasks 5-8 and by
later algorithm plans: deterministic programs and observables, plus the
recording backend and the qpanda3-runtime stubs that keep runtime
contract tests credential-free.
"""

import pytest

from pyqpanda3.core import CNOT, H, QProg, RX, RY, measure
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda3.vqcircuit import VQCircuit

from pyqpanda_alg.execution import ExecutionOptions, QPandaRuntimeBackend

from test.execution.fakes import FakeDevice, FakeRuntimeService, RecordingBackend


@pytest.fixture
def recording_backend():
    """Backend recording every submission with deterministic results."""
    return RecordingBackend()


@pytest.fixture
def fake_runtime_service():
    """qpanda3-runtime stand-in that records calls and needs no credentials."""
    return FakeRuntimeService()


@pytest.fixture
def fake_device():
    """Deterministic QDevice stand-in with mock-configurable capability data."""
    return FakeDevice()


@pytest.fixture
def runtime_backend(fake_runtime_service, fake_device):
    """QPandaRuntimeBackend wired to the fake service and device.

    Constructing the backend requires the optional qpanda3-runtime
    package, which CI does not install; tests consuming this fixture
    SKIP cleanly (instead of failing) without it.
    """
    pytest.importorskip("qpanda3_runtime")
    return QPandaRuntimeBackend(fake_runtime_service, fake_device)


@pytest.fixture
def bell_program():
    """Two-qubit Bell-state circuit with measurement."""
    prog = QProg()
    prog << H(0) << CNOT(0, 1) << measure([0, 1], [0, 1])
    return prog


@pytest.fixture
def five_qubit_prog():
    """Five-qubit circuit used by capability and preflight tests."""
    prog = QProg()
    for qubit in range(5):
        prog << H(qubit)
    return prog


@pytest.fixture
def ansatz():
    """Two-parameter, two-qubit variational ansatz."""
    ansatz = VQCircuit(2)
    ansatz.set_Param([2])
    ansatz << RX(0, ansatz.Param([0]))
    ansatz << RY(1, ansatz.Param([1]))
    return ansatz


@pytest.fixture
def observable():
    """ZZ observable on the first two qubits."""
    return Hamiltonian({"Z0 Z1": 1.0})
