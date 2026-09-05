"""Frozen PyQPanda3 compilation profile and per-attempt records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping

from pyqpanda3.core import H, QCircuit, QProg
from pyqpanda3.transpilation import Transpiler

from pyqpanda_alg.plugin import QFT

from .config import CompilerConfig, PreparationMethod
from .exceptions import ConfigurationError, InternalInvariantError
from ._preparation import PreparationBuild


FROZEN_COMPILER_PROFILE = CompilerConfig()
TECHNICAL_REPETITIONS = 5
_PROFILE_DOMAIN = b"qseencode-compiler-profile-v1\0"
_IGNORED_IR_TOKENS = {
    "QINIT", "CREG", "DAGGER", "ENDDAGGER", "CONTROL", "ENDCONTROL",
    "MEASURE", "RESET", "BARRIER",
}


@dataclass(frozen=True, slots=True)
class EndToEndProgram:
    program: QProg
    basis: str
    method: PreparationMethod
    output_qubits: tuple[int, ...]
    ancillas: tuple[int, ...]
    required_qubits: int
    allocated_qubits: int


@dataclass(frozen=True, slots=True)
class CompilationAttempt:
    attempt_index: int
    success: bool
    status: str
    compiled_program: QProg | None
    compiled_originir: str | None
    originir_sha256: str | None
    compiled_depth: int | None
    compiled_total_gates: int | None
    compiled_one_qubit_gates: int | None
    compiled_two_qubit_gates: int | None
    compiled_cnot_gates: int | None
    required_qubits: int
    allocated_qubits: int
    ancilla_count: int
    compiler_profile_fingerprint: str
    compiler_profile: Mapping[str, Any]
    exception_type: str | None = None
    exception_message: str | None = None
    diagnostics: Mapping[str, Any] | None = None


def compiler_profile_payload(profile: CompilerConfig) -> dict[str, Any]:
    return {
        "pyqpanda_version": profile.pyqpanda_version,
        "topology": profile.topology,
        "physical_capacity_multiplier": profile.physical_capacity_multiplier,
        "initial_mapping": profile.initial_mapping,
        "optimization_level": profile.optimization_level,
        "basis_gates": list(profile.basis_gates),
        "technical_repetitions": profile.technical_repetitions,
        "resource_aggregation": profile.resource_aggregation,
    }


def compiler_profile_fingerprint(profile: CompilerConfig) -> str:
    encoded = json.dumps(
        compiler_profile_payload(profile), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(_PROFILE_DOMAIN + encoded).hexdigest()


def linear_topology(width: int) -> list[list[int]]:
    if type(width) is not int or width < 2:
        raise InternalInvariantError(code="invalid_allocated_width")
    return [[index, index + 1] for index in range(width - 1)]


def compose_end_to_end_program(
    build: PreparationBuild,
    *,
    basis: str,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
) -> EndToEndProgram:
    """Place every method on the same 2n hardware and append basis decoding."""

    if basis not in {"walsh", "fourier"}:
        raise ConfigurationError(code="invalid_basis")
    logical = build.logical_output_qubits
    allocated = profile.physical_capacity_multiplier * logical
    if allocated != 2 * logical:
        raise ConfigurationError(code="unsupported_physical_capacity_profile")
    if build.required_qubits > allocated:
        raise InternalInvariantError(code="preparation_exceeds_allocated_width")
    outputs = tuple(range(allocated - logical, allocated))

    circuit = QCircuit()
    circuit << build.circuit
    if build.method is not PreparationMethod.DS_QUANTUM_STATE_PREPARATION:
        # The list overload is old-index -> new-index and leaves the Phase 3
        # build untouched because the circuit above is a fresh container.
        circuit.remap(list(outputs))
    elif build.output_qubits != outputs:
        raise InternalInvariantError(code="ds_output_register_contract_changed")

    program = QProg(allocated)
    program << circuit
    if basis == "walsh":
        program << [H(qubit) for qubit in outputs]
    else:
        program << QFT(list(outputs))
    return EndToEndProgram(
        program=program,
        basis=basis,
        method=build.method,
        output_qubits=outputs,
        ancillas=build.ancillas,
        required_qubits=build.required_qubits,
        allocated_qubits=allocated,
    )


def _parse_operations(originir: str) -> list[tuple[str, tuple[int, ...]]]:
    operations: list[tuple[str, tuple[int, ...]]] = []
    for raw in originir.splitlines():
        line = raw.strip()
        if not line:
            continue
        gate = line.split(maxsplit=1)[0].upper()
        if gate in _IGNORED_IR_TOKENS:
            continue
        qubits = tuple(int(value) for value in re.findall(r"q\[(\d+)\]", line))
        if qubits:
            operations.append((gate, qubits))
    return operations


def _successful_attempt(
    compilation: EndToEndProgram,
    attempt_index: int,
    profile: CompilerConfig,
    compiled: QProg,
) -> CompilationAttempt:
    originir = compiled.originir()
    if not isinstance(originir, str) or not originir.strip():
        raise InternalInvariantError(code="empty_compiled_representation")
    operations = _parse_operations(originir)
    counts = Counter(gate for gate, _ in operations)
    invalid_basis = sorted(set(counts) - set(profile.basis_gates))
    topology = {tuple(edge) for edge in linear_topology(compilation.allocated_qubits)}
    topology |= {(right, left) for left, right in topology}
    topology_violations = [
        qubits for _, qubits in operations
        if len(qubits) == 2 and tuple(qubits) not in topology
    ]
    one_qubit = sum(len(qubits) == 1 for _, qubits in operations)
    two_qubit = sum(len(qubits) == 2 for _, qubits in operations)
    cnot = counts.get("CNOT", 0)
    total = len(operations)
    depth = int(compiled.depth())
    if invalid_basis or topology_violations or min(one_qubit, two_qubit, total, depth) < 0:
        raise InternalInvariantError(code="compiled_resource_validation_failed")
    if two_qubit != cnot:
        raise InternalInvariantError(code="compiled_two_qubit_cnot_mismatch")
    payload = compiler_profile_payload(profile)
    return CompilationAttempt(
        attempt_index=attempt_index,
        success=True,
        status="success",
        compiled_program=compiled,
        compiled_originir=originir,
        originir_sha256=hashlib.sha256(originir.encode("utf-8")).hexdigest(),
        compiled_depth=depth,
        compiled_total_gates=total,
        compiled_one_qubit_gates=one_qubit,
        compiled_two_qubit_gates=two_qubit,
        compiled_cnot_gates=cnot,
        required_qubits=compilation.required_qubits,
        allocated_qubits=compilation.allocated_qubits,
        ancilla_count=len(compilation.ancillas),
        compiler_profile_fingerprint=compiler_profile_fingerprint(profile),
        compiler_profile=payload,
        diagnostics={"basis_valid": True, "topology_valid": True},
    )


def compile_attempt(
    compilation: EndToEndProgram,
    attempt_index: int,
    *,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
    transpiler_factory: Callable[[], Any] = Transpiler,
) -> CompilationAttempt:
    if type(attempt_index) is not int or not 0 <= attempt_index < TECHNICAL_REPETITIONS:
        raise InternalInvariantError(code="invalid_compilation_attempt_index")
    payload = compiler_profile_payload(profile)
    fingerprint = compiler_profile_fingerprint(profile)
    try:
        compiled = transpiler_factory().transpile(
            compilation.program,
            linear_topology(compilation.allocated_qubits),
            {qubit: qubit for qubit in range(compilation.allocated_qubits)},
            profile.optimization_level,
            list(profile.basis_gates),
        )
        if compiled is None:
            raise InternalInvariantError(code="empty_compiled_program")
        return _successful_attempt(compilation, attempt_index, profile, compiled)
    except InternalInvariantError:
        raise
    except Exception as error:
        return CompilationAttempt(
            attempt_index=attempt_index,
            success=False,
            status="compile_failure",
            compiled_program=None,
            compiled_originir=None,
            originir_sha256=None,
            compiled_depth=None,
            compiled_total_gates=None,
            compiled_one_qubit_gates=None,
            compiled_two_qubit_gates=None,
            compiled_cnot_gates=None,
            required_qubits=compilation.required_qubits,
            allocated_qubits=compilation.allocated_qubits,
            ancilla_count=len(compilation.ancillas),
            compiler_profile_fingerprint=fingerprint,
            compiler_profile=payload,
            exception_type=type(error).__name__,
            exception_message=str(error),
            diagnostics={"basis_valid": False, "topology_valid": False},
        )


def compile_five_repetitions(
    compilation: EndToEndProgram,
    *,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
    transpiler_factory: Callable[[], Any] = Transpiler,
) -> tuple[CompilationAttempt, ...]:
    if profile.technical_repetitions != TECHNICAL_REPETITIONS:
        raise ConfigurationError(code="technical_repetitions_must_equal_five")
    return tuple(
        compile_attempt(
            compilation,
            attempt_index,
            profile=profile,
            transpiler_factory=transpiler_factory,
        )
        for attempt_index in range(TECHNICAL_REPETITIONS)
    )


__all__ = [
    "FROZEN_COMPILER_PROFILE", "TECHNICAL_REPETITIONS", "EndToEndProgram",
    "CompilationAttempt", "compiler_profile_payload", "compiler_profile_fingerprint",
    "linear_topology", "compose_end_to_end_program", "compile_attempt",
    "compile_five_repetitions",
]
