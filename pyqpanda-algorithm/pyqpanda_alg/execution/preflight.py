"""Preflight capability validation and fake execution for runtime work.

:class:`~pyqpanda_alg.execution.runtime.QPandaRuntimeBackend` runs
:func:`run_preflight` before every submission.  The steps selected by
``ExecutionOptions.preflight`` are:

- ``NONE`` — no preflight work at all; the submission is forwarded
  verbatim.
- ``TRANSPILE_ONLY`` — static capability validation, then the fake
  backend's transpile path (the same ``FakeBackend`` the real device
  exposes through ``device.fake_backend()``).
- ``FAKE_EXECUTE`` — validation and transpile, then the fake task
  itself (``FakeBackend.sample``/``estimate``); its task/result
  metadata is returned before the real submission happens.

Every preflight step fails fast with a public execution-layer error
before the service is touched, so a rejected or failed preflight never
leaves a half-submitted task behind.  Capability data comes from the
``QDevice`` surface (``available_qubits``, ``basic_gates``,
``chip_topo_edges``, ``fake_backend``), which keeps this module free of
any qpanda3-runtime import.
"""

from typing import Any, Optional

from .errors import (
    BackendUnavailableError,
    DeviceCapabilityError,
    TranspilationError,
)
from .options import ExecutionOptions, PreflightMode


def run_preflight(
    device: Any,
    circuit: Any,
    *,
    observable: Any = None,
    options: ExecutionOptions,
) -> Optional[dict]:
    """Run the preflight steps selected by ``options.preflight``.

    Returns the fake-execution metadata record for ``FAKE_EXECUTE`` and
    ``None`` otherwise.  Raises a public execution-layer error when any
    step fails; the caller then never submits.
    """
    if options.preflight is PreflightMode.NONE:
        return None
    _validate(device, circuit, observable=observable, options=options)
    if options.preflight is PreflightMode.TRANSPILE_ONLY:
        _transpile(device.fake_backend(), circuit, options)
        return None
    return _fake_execute(device, circuit, observable=observable, options=options)


def _validate(
    device: Any,
    circuit: Any,
    *,
    observable: Any,
    options: ExecutionOptions,
) -> None:
    """Reject submissions the device cannot run, before anything else."""
    available = device.available_qubits()
    _check_device_availability(available)
    _check_qubit_capacity(available, circuit)
    if observable is not None:
        _check_observable_qubits(available, observable)
    _check_gate_set(device, circuit)
    _check_topology(device, circuit)
    _check_specified_block(available, options.specified_block)


def _check_device_availability(available: list) -> None:
    """Reject devices that expose no usable qubits."""
    if not available:
        raise DeviceCapabilityError("device has no available qubits")


def _check_qubit_capacity(available: list, circuit: Any) -> None:
    """Reject circuits whose qubit span exceeds the device capacity."""
    qubits = _circuit_qubits(circuit)
    if not qubits:
        return
    required = max(qubits) + 1
    if required > len(available):
        raise DeviceCapabilityError(
            f"the circuit requires {required} qubits but the device "
            f"provides {len(available)}"
        )


def _check_observable_qubits(available: list, observable: Any) -> None:
    """Reject observables acting on qubits the device does not provide."""
    for qubit in _observable_qubits(observable):
        if qubit not in available:
            raise DeviceCapabilityError(
                f"observable acts on qubit {qubit} which is not available "
                f"on the device (available: {available})"
            )


def _check_gate_set(device: Any, circuit: Any) -> None:
    """Reject circuits using gates the device does not support."""
    used = _circuit_gates(circuit)
    if not used:
        return
    unsupported = sorted(used - set(device.basic_gates()))
    if unsupported:
        raise DeviceCapabilityError(
            f"device does not support gate(s): {', '.join(unsupported)}"
        )


def _check_topology(device: Any, circuit: Any) -> None:
    """Reject two-qubit gates acting across unconnected device qubits."""
    edges = {frozenset(edge) for edge in device.chip_topo_edges()}
    for pair in _two_qubit_pairs(circuit):
        if pair not in edges:
            raise DeviceCapabilityError(
                f"device topology does not connect qubits {sorted(pair)}"
            )


def _check_specified_block(available: list, specified_block: Optional[tuple]) -> None:
    """Reject placement blocks that include unavailable qubits."""
    if specified_block is None:
        return
    outside = [qubit for qubit in specified_block if qubit not in available]
    if outside:
        raise DeviceCapabilityError(
            f"specified block qubits {outside} are not available on the device"
        )


def _transpile(fake: Any, circuit: Any, options: ExecutionOptions) -> list:
    """Run the fake-backend transpile path, returning transpiled circuits.

    The fake backend mirrors the real device's topology and gate set, so
    a circuit it cannot transpile would fail on the real device too.
    """
    try:
        transpiled, failed = fake.transpile(
            [circuit],
            specified_block=_specified_block(options),
            is_optimization=options.is_optimization,
        )
    except Exception as exc:
        raise TranspilationError(
            f"transpile preflight could not transpile the circuit on the "
            f"fake backend: {exc}"
        ) from exc
    if failed:
        raise TranspilationError(
            f"transpile preflight: the fake backend could not transpile "
            f"{len(failed)} circuit(s)"
        )
    return transpiled


def _fake_execute(
    device: Any, circuit: Any, *, observable: Any, options: ExecutionOptions
) -> dict:
    """Run the fake task and record its metadata before real submission.

    The fake backend cannot always run in the current process (on some
    platforms its multiprocessing entry point requires a ``__main__``
    guard); such environment failures surface as
    :class:`BackendUnavailableError` naming ``fake_execute`` so callers
    can switch the preflight mode instead of guessing.
    """
    fake = device.fake_backend()
    transpiled = _transpile(fake, circuit, options)
    try:
        if observable is None:
            fake_result = fake.sample(circuit, shots=options.shots)
            kind = "sample"
        else:
            fake_result = fake.estimate(
                circuit, _as_pauli_operator(observable), shots=options.shots
            )
            kind = "estimate"
    except Exception as exc:
        raise BackendUnavailableError(
            f"fake_execute could not run the fake backend in this process: {exc}"
        ) from exc
    return {
        "mode": "fake_execute",
        "kind": kind,
        "shots": options.shots,
        "transpiled": transpiled,
        "result": fake_result,
    }


def _circuit_qubits(circuit: Any) -> list:
    """Return the qubit indices used by ``circuit`` (attribute or method)."""
    qubits = getattr(circuit, "qubits", None)
    if qubits is None:
        return []
    return list(qubits() if callable(qubits) else qubits)


def _circuit_gates(circuit: Any) -> set:
    """Return the gate names used by ``circuit``, excluding measurement.

    Circuits without gate introspection (for example parameterized
    ansatze) contribute no gates, so the gate-set check is skipped for
    them while qubit capacity still applies.
    """
    count_ops = getattr(circuit, "count_ops", None)
    if count_ops is None:
        return set()
    return set(count_ops())


def _two_qubit_pairs(circuit: Any) -> list:
    """Return every two-qubit gate's qubit pair as an unordered set."""
    operations = getattr(circuit, "operations", None)
    if operations is None:
        return []
    pairs = []
    for operation in operations():
        qubits = getattr(operation, "qubits", None)
        if callable(qubits):
            qubits = qubits()
        if isinstance(qubits, (list, tuple)) and len(qubits) == 2:
            pairs.append(frozenset(qubits))
    return pairs


def _observable_qubits(observable: Any) -> list:
    """Return the qubit indices an observable acts on.

    Hamiltonians are converted to their Pauli operator, which exposes
    ``qubits()`` as ``(non-z qubits, z qubits)``.
    """
    pauli = getattr(observable, "pauli_operator", None)
    if pauli is not None:
        observable = pauli()
    first, second = observable.qubits()
    return list(first) + list(second)


def _as_pauli_operator(observable: Any) -> Any:
    """Return the observable in the Pauli form the fake backend expects."""
    pauli = getattr(observable, "pauli_operator", None)
    if pauli is None:
        return observable
    return pauli()


def _specified_block(options: ExecutionOptions):
    """Return the placement block in the list shape the fake expects."""
    if options.specified_block is None:
        return None
    return list(options.specified_block)
