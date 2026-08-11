"""VQE-specific fixtures for the runtime and checkpoint contract tests.

The one-qubit ``hamiltonian`` fixture (two ansatz parameters, one
qubit) feeds the verbatim VQSession-preference tests, and the
``estimator_only_backend`` fixture advertises a backend without
variational sessions so the fallback path can be exercised.  The
``runtime_backend`` fixture overrides the repository-wide one so the
VQE runtime tests can count session creations.
"""

import pytest
from pyqpanda3.hamiltonian import Hamiltonian

from pyqpanda_alg.execution import BackendCapabilities, QPandaRuntimeBackend

from test.execution.fakes import RecordingBackend


@pytest.fixture
def hamiltonian():
    """One-qubit Z observable: a two-parameter, single-qubit ansatz."""
    return Hamiltonian({"Z0": 1.0})


@pytest.fixture
def estimator_only_backend():
    """RecordingBackend without variational sessions: estimates only.

    The flat canned expectation keeps the verbatim fallback test
    deterministic; checkpoint tests pin their own scripted expectations
    inline.
    """
    backend = RecordingBackend(expectations=[0.5])
    backend.capabilities = BackendCapabilities(variational_session=False)
    return backend


class _CountingRuntimeBackend(QPandaRuntimeBackend):
    """QPandaRuntimeBackend recording every variational session creation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.variational_session_calls = 0

    def create_variational_session(self, ansatz, observable, *, options):
        self.variational_session_calls += 1
        return super().create_variational_session(
            ansatz, observable, options=options
        )


@pytest.fixture
def runtime_backend(fake_runtime_service, fake_device):
    """Runtime backend wired to the fake service, counting sessions.

    Overrides the repository-wide fixture so VQE tests can assert how
    many variational sessions one run creates; without the optional
    qpanda3-runtime package the tests skip cleanly.
    """
    pytest.importorskip("qpanda3_runtime")
    return _CountingRuntimeBackend(fake_runtime_service, fake_device)
