"""Optional qpanda3-runtime execution backend.

:class:`QPandaRuntimeBackend` submits work to a qpanda3-runtime
``RuntimeService`` and an explicit device.  The runtime dependency is
optional and imported lazily: importing ``pyqpanda_alg`` never requires
``qpanda3-runtime``, and constructing the backend without it raises
:class:`~pyqpanda_alg.execution.errors.MissingRuntimeDependencyError`
with the documented install command.  Runtime failures surface as
public execution-layer errors — there is never a silent fallback to
:class:`~pyqpanda_alg.execution.local.LocalBackend`, and submission is
never retried.
"""

from typing import Any, Callable

from .capabilities import BackendCapabilities
from .errors import (
    DeviceCapabilityError,
    MissingRuntimeDependencyError,
    TaskSubmissionError,
)
from .options import ExecutionOptions
from .preflight import run_preflight
from .runtime_task import RuntimeBackendTask

#: The documented install command for the optional runtime extra.
_INSTALL_HINT = "pip install pyqpanda-algorithm[runtime]"


def _require_runtime_dependency() -> None:
    """Import qpanda3-runtime lazily or raise with the install command."""
    try:
        import qpanda3_runtime  # noqa: F401
    except ImportError as exc:
        raise MissingRuntimeDependencyError(
            "QPandaRuntimeBackend requires the optional qpanda3-runtime "
            f"package; install it with `{_INSTALL_HINT}`"
        ) from exc


class QPandaRuntimeBackend:
    """Execution backend submitting work to a qpanda3-runtime service.

    ``service`` must already be logged in and ``device`` is an explicit
    QDevice (or FakeBackend) reference; the backend never authenticates
    and never replaces a failed runtime path with a local one.
    """

    capabilities = BackendCapabilities(statevector=False, variational_session=False)

    def __init__(self, service: Any, device: Any) -> None:
        _require_runtime_dependency()
        self.service = service
        self.device = device

    def submit_sample(self, circuit: Any, *, options: ExecutionOptions) -> RuntimeBackendTask:
        """Submit sampling of ``circuit`` and return the adapted task.

        The preflight steps selected by ``options.preflight`` run before
        the service call; ``FAKE_EXECUTE`` records the fake task's
        metadata on the returned task as ``fake_execution``.  The
        runtime-relevant options map exactly onto the qpanda3-runtime
        ``sample()`` keyword arguments; the device and the circuit are
        forwarded verbatim.
        """
        fake_execution = run_preflight(self.device, circuit, options=options)
        qtask = self._submit(
            lambda: self.service.sample(
                circuits=circuit,
                device=self.device,
                specified_block=_specified_block(options),
                shots=options.shots,
                is_amend=options.is_amend,
                is_mapping=options.is_mapping,
                is_optimization=options.is_optimization,
            ),
            "sample",
        )
        task = RuntimeBackendTask(
            qtask, kind="sample", shots=options.shots, timeout=options.timeout
        )
        if fake_execution is not None:
            task.fake_execution = fake_execution
        return task

    def submit_estimate(
        self, circuit_and_observable: Any, *, options: ExecutionOptions
    ) -> RuntimeBackendTask:
        """Submit expectation estimation and return the adapted task.

        ``circuit_and_observable`` is the ``(circuit, observable)`` pair
        forwarded verbatim to qpanda3-runtime ``estimate()``; the pair's
        observable also feeds the preflight capability checks and the
        ``FAKE_EXECUTE`` record.
        """
        circuit, observable = circuit_and_observable
        fake_execution = run_preflight(
            self.device, circuit, observable=observable, options=options
        )
        qtask = self._submit(
            lambda: self.service.estimate(
                circuit_with_observable=circuit_and_observable,
                device=self.device,
                specified_block=_specified_block(options),
                shots=options.shots,
                is_amend=options.is_amend,
                is_mapping=options.is_mapping,
                is_optimization=options.is_optimization,
            ),
            "estimate",
        )
        task = RuntimeBackendTask(
            qtask, kind="estimate", shots=options.shots, timeout=options.timeout
        )
        if fake_execution is not None:
            task.fake_execution = fake_execution
        return task

    def submit_statevector(self, circuit: Any, *, options: ExecutionOptions) -> RuntimeBackendTask:
        """Reject state-vector execution before any service submission."""
        raise DeviceCapabilityError(
            "QPandaRuntimeBackend does not support state-vector execution"
        )

    def _submit(self, call: Callable[[], Any], operation: str) -> Any:
        """Run the service submission, converting transport failures."""
        try:
            return call()
        except Exception as exc:
            raise TaskSubmissionError(
                f"submitting the {operation} task failed: {exc}"
            ) from exc


def _specified_block(options: ExecutionOptions):
    """Return the tuple option in the list shape qpanda3-runtime expects."""
    if options.specified_block is None:
        return None
    return list(options.specified_block)
