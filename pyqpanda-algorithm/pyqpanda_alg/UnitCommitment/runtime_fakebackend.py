"""Optional OriginQ Runtime FakeBackend adapter for the case5 application.

The regular :mod:`.backend` module deliberately keeps the cloud-QVM path
separate from the QPanda3 Runtime path.  This adapter is therefore optional:
it is imported only by the Runtime rehearsal script and requires the
``qpanda3_runtime`` package at run time.

The adapter uses the official Runtime objects in the following order::

    RuntimeService -> QDevice -> FakeBackend -> transpile -> sample

``FakeBackend.sample`` is a local finite-shot simulation.  It does not create
or submit a real-QPU task.  The API key is read from the caller-selected
environment variable and is never returned, logged, or written to an output
file.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from threading import RLock
from typing import Any, Iterator, Sequence


_DEFAULT_BLOCKS: dict[str, tuple[int, ...]] = {
    # Connected calibration-ranked blocks recorded by the read-only preflight.
    "WK_C180": (85, 94, 95, 96, 103, 104, 105, 113, 114, 123),
    "WK_C180_2": (39, 40, 48, 49, 50, 57, 58, 59, 67, 68),
}


@dataclass(frozen=True)
class RuntimeFakeBackendResult:
    """Finite-shot result and non-secret execution metadata."""

    probabilities: dict[str, float]
    metadata: dict[str, Any]


def _require_runtime() -> Any:
    if importlib.util.find_spec("qpanda3_runtime") is None:
        raise RuntimeError(
            "qpanda3_runtime is not installed. Install the optional OriginQ "
            "Runtime package before selecting Runtime FakeBackend mode."
        )
    return importlib.import_module("qpanda3_runtime")


def _normalise_block(block: Sequence[int] | None) -> list[int] | None:
    if block is None:
        return None
    values = [int(value) for value in block]
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("specified_block must contain unique non-negative qubit indices")
    return values


def _first_connected_block(edges: Sequence[Sequence[int]], width: int) -> list[int]:
    """Return a deterministic connected block when no device-specific block is supplied."""

    width = int(width)
    if width < 1:
        raise ValueError("width must be positive")
    adjacency: dict[int, set[int]] = {}
    for raw_edge in edges:
        if len(raw_edge) != 2:
            continue
        left, right = (int(raw_edge[0]), int(raw_edge[1]))
        if left == right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    for root in sorted(adjacency):
        selected = [root]
        selected_set = {root}
        frontier = [root]
        while frontier and len(selected) < width:
            current = frontier.pop(0)
            for neighbour in sorted(adjacency.get(current, ())):
                if neighbour in selected_set:
                    continue
                selected_set.add(neighbour)
                selected.append(neighbour)
                frontier.append(neighbour)
                if len(selected) == width:
                    break
        if len(selected) == width:
            return selected
    raise RuntimeError(
        f"unable to find a connected Runtime device block with {width} qubits"
    )


def _get_attr_or_call(value: Any, name: str, default: Any = None) -> Any:
    member = getattr(value, name, default)
    try:
        return member() if callable(member) else member
    except Exception:
        return default


def _normalise_probabilities(raw: Any, shots: int) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Runtime FakeBackend returned {type(raw).__name__}; expected a probability dictionary"
        )
    result = {str(key): float(value) for key, value in raw.items()}
    if any(value < -1.0e-12 or not math.isfinite(value) for value in result.values()):
        raise RuntimeError("Runtime FakeBackend returned an invalid probability value")
    total = float(sum(result.values()))
    if total <= 0.0:
        raise RuntimeError("Runtime FakeBackend returned zero probability mass")
    result = {key: max(0.0, value / total) for key, value in result.items()}
    lattice_error = max(
        (
            abs(value * int(shots) - round(value * int(shots)))
            for value in result.values()
        ),
        default=0.0,
    )
    if lattice_error > 1.0e-6:
        raise RuntimeError(
            "Runtime FakeBackend result is not consistent with the requested "
            f"{int(shots)}-shot empirical-frequency lattice"
        )
    return result


def _program_sha256(program: Any) -> str:
    return hashlib.sha256(str(program).encode("utf-8")).hexdigest()


def _scale_qubit_calibration_to_gate_units(
    calibration: dict[str, Any], scale_to_microseconds: float
) -> dict[str, Any]:
    """Copy T1/T2 calibration records into QGateClock units.

    Runtime calibration records are expressed in microseconds.  The SDK passes
    those numbers and ``QGateClock.IntValue`` directly to ``decoherence_error``;
    because the physical unit of ``IntValue`` is not documented in the Runtime
    metadata, conversion is only performed when the caller explicitly supplies
    the assumed number of microseconds per clock unit.
    """

    scale = float(scale_to_microseconds)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_to_microseconds must be finite and positive")
    converted: dict[str, Any] = {}
    for qubit, raw_parameters in calibration.items():
        if not isinstance(raw_parameters, dict):
            converted[qubit] = raw_parameters
            continue
        parameters = dict(raw_parameters)
        for name in ("T1", "T2"):
            if name not in parameters:
                continue
            value = float(parameters[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"qubit {qubit} has an invalid {name} calibration value"
                )
            parameters[name] = value / scale
        converted[qubit] = parameters
    return converted


def _gate_counts(program: Any) -> dict[str, int]:
    text = str(program)
    counts: dict[str, int] = {}
    # Runtime transpilation returns OriginIR text.  Counting the leading gate
    # token keeps the record independent of SDK-private program classes.
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s", line)
        if match:
            token = match.group(1).upper()
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items()))


class RuntimeFakeBackendRunner:
    """Run measured QProg objects through a local Runtime FakeBackend.

    ``execution_path='direct'`` is intentional.  It invokes the Runtime
    FakeBackend transpiler and sampler in-process, which makes the physical
    block and compiled OriginIR auditable while guaranteeing that no remote
    quantum task is created.
    """

    def __init__(
        self,
        *,
        real_device_id: str = "WK_C180",
        api_key_env: str = "QPANDA3_API_KEY",
        specified_block: Sequence[int] | None = None,
        is_optimization: bool = True,
        compensation_mode: str = "neutral",
        gate_clock_scale_to_microseconds: float | None = None,
    ) -> None:
        self.runtime = _require_runtime()
        key = os.environ.get(str(api_key_env))
        if not key:
            raise RuntimeError(
                f"{api_key_env} is not visible to this process; the key value is not accepted as a function argument"
            )
        self.api_key_env = str(api_key_env)
        self.real_device_id = str(real_device_id)
        self.service = self.runtime.RuntimeService(auto_track=False)
        # Do not include the key in exceptions, records, or reprs.
        self.service.login(key)
        self.qdevice = self.service.device(self.real_device_id)
        self.device = self.qdevice.fake_backend()
        if gate_clock_scale_to_microseconds is None:
            self.gate_clock_scale_to_microseconds = None
        else:
            scale = float(gate_clock_scale_to_microseconds)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "gate_clock_scale_to_microseconds must be finite and positive"
                )
            self.gate_clock_scale_to_microseconds = scale
        self._timebase_lock = RLock()
        self.fakebackend_timebase_metadata = {
            "normalization_applied": self.gate_clock_scale_to_microseconds is not None,
            "t1_t2_calibration_unit_assumption": "microseconds",
            "gate_clock_intvalue_physical_unit": (
                "unconfirmed; caller-provided scale is a research assumption"
                if self.gate_clock_scale_to_microseconds is not None
                else "unconfirmed; no conversion applied"
            ),
            "gate_clock_scale_to_microseconds": self.gate_clock_scale_to_microseconds,
            "conversion": (
                "temporarily divide T1/T2 by the supplied microseconds-per-clock-unit scale"
                if self.gate_clock_scale_to_microseconds is not None
                else "none; SDK raw numeric values are preserved"
            ),
            "status": (
                "research-assumption-pending-official-confirmation"
                if self.gate_clock_scale_to_microseconds is not None
                else "unscaled-sdk-default"
            ),
            "scope": "local FakeBackend noise generation only; does not change QPU execution",
        }
        self.is_optimization = bool(is_optimization)
        normalized_compensation = str(compensation_mode).strip().lower().replace("_", "-")
        if normalized_compensation not in {"neutral", "device"}:
            raise ValueError("compensation_mode must be neutral or device")
        self.compensation_mode = normalized_compensation
        before = getattr(self.device, "_compensate_angle_map", {})
        self.compensation_entries_before = len(before) if isinstance(before, dict) else 0
        if normalized_compensation == "neutral" and isinstance(before, dict):
            # Runtime device compensation describes coherent hardware effects;
            # the FakeBackend noise model is calibration-derived but does not
            # include that coherent error.  Neutralizing the map preserves the
            # logical circuit while retaining topology and stochastic noise.
            self.device._compensate_angle_map = {}
        requested = _normalise_block(specified_block)
        if requested is None:
            requested = list(_DEFAULT_BLOCKS.get(self.real_device_id, ()))
        if requested:
            available = {
                int(value)
                for value in (_get_attr_or_call(self.qdevice, "available_qubits", []) or [])
            }
            if available and any(value not in available for value in requested):
                requested = []
        if not requested:
            edges = _get_attr_or_call(self.qdevice, "chip_topo_edges", []) or []
            requested = _first_connected_block(edges, 10)
        self.specified_block = requested

    @contextmanager
    def fakebackend_timebase(self) -> Iterator[None]:
        """Temporarily normalize FakeBackend T1/T2 values for local simulation.

        ``gate_clock_scale_to_microseconds`` is deliberately optional.  When
        supplied, Runtime's per-qubit T1/T2 values (microseconds) are converted
        to the assumed QGateClock unit only while the local noise model is
        generated.  The original SDK calibration dictionary is restored even
        if sampling raises.  This does not alter the QDevice snapshot or any
        real-hardware task.
        """

        scale = self.gate_clock_scale_to_microseconds
        if scale is None:
            yield
            return
        with self._timebase_lock:
            if getattr(self.device, "_noise_info", None) is not None:
                raise RuntimeError(
                    "timebase normalization requires calibration-backed FakeBackend noise, "
                    "not a custom noise_info override"
                )
            original = getattr(self.device, "_qubit_params_origin", None)
            if not isinstance(original, dict):
                raise RuntimeError(
                    "Runtime FakeBackend does not expose its calibration dictionary"
                )
            normalized = _scale_qubit_calibration_to_gate_units(original, scale)
            self.device._qubit_params_origin = normalized
            try:
                yield
            finally:
                self.device._qubit_params_origin = original

    def _compile(self, program: Any) -> tuple[Any, dict[str, Any]]:
        started = perf_counter()
        transpiled, failed = self.device.transpile(
            [program],
            specified_block=list(self.specified_block),
            is_optimization=self.is_optimization,
        )
        elapsed = perf_counter() - started
        if failed or len(transpiled) != 1:
            raise RuntimeError(
                f"Runtime FakeBackend transpilation failed: {failed!r}"
            )
        compiled = transpiled[0]
        return compiled, {
            "transpilation_seconds": float(elapsed),
            "transpiled_program_sha256": _program_sha256(compiled),
            "compiled_gate_counts": _gate_counts(compiled),
            "transpiled_program_type": type(compiled).__name__,
        }

    def sample(self, program: Any, *, shots: int = 256) -> RuntimeFakeBackendResult:
        shot_count = int(shots)
        if shot_count < 1:
            raise ValueError("shots must be positive for Runtime FakeBackend sampling")
        compiled, compile_record = self._compile(program)
        started = perf_counter()
        with self.fakebackend_timebase():
            raw = self.device.sample(compiled, shots=shot_count)
        elapsed = perf_counter() - started
        probabilities = _normalise_probabilities(raw, shot_count)
        metadata: dict[str, Any] = {
            "runtime_package": "qpanda3_runtime",
            "runtime_version": str(getattr(self.runtime, "__version__", "unknown")),
            "pyqpanda3_version": str(
                getattr(importlib.import_module("pyqpanda3"), "__version__", "unknown")
            ),
            "real_device_id": self.real_device_id,
            "execution_device": type(self.device).__name__,
            "execution_path": "direct",
            "specified_block": list(self.specified_block),
            "is_optimization": self.is_optimization,
            "compensation_mode": self.compensation_mode,
            "compensation_entries_before": self.compensation_entries_before,
            "compensation_entries_used": len(getattr(self.device, "_compensate_angle_map", {}) or {}),
            "shots": shot_count,
            "sample_seconds": float(elapsed),
            "finite_shot_empirical_frequencies": True,
            "raw_integer_counts_exposed": False,
            "used_fake_backend": True,
            "used_remote_backend": False,
            "real_qpu_submitted": False,
            "qcloud_service_used_by_project": False,
            "api_key_environment_variable": self.api_key_env,
            "api_key_present": True,
            "api_key_value_saved": False,
            "fakebackend_timebase": dict(self.fakebackend_timebase_metadata),
        }
        metadata.update(compile_record)
        return RuntimeFakeBackendResult(probabilities=probabilities, metadata=metadata)

    def bell_smoke(self, *, shots: int = 128) -> dict[str, Any]:
        """Run a small Bell-state smoke circuit through the same direct path."""

        from pyqpanda3.core import CNOT, H, QProg, measure

        program = QProg(2)
        qubits = program.qubits()
        program << H(qubits[0]) << CNOT(qubits[0], qubits[1])
        program << measure(qubits[0], 0) << measure(qubits[1], 1)
        result = self.sample(program, shots=shots)
        even_parity = float(result.probabilities.get("00", 0.0)) + float(
            result.probabilities.get("11", 0.0)
        )
        return {
            "shots": int(shots),
            "probabilities": result.probabilities,
            "bell_even_parity_probability": even_parity,
            "passed": bool(even_parity >= 0.5),
            "metadata": result.metadata,
        }


__all__ = ["RuntimeFakeBackendResult", "RuntimeFakeBackendRunner"]

