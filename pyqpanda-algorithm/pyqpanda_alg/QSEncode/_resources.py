"""Five-repeat compiled-resource auditing without selection logic."""

from __future__ import annotations

import statistics
from typing import Any, Callable

from pyqpanda3.transpilation import Transpiler

from .config import CompilerConfig
from .exceptions import InternalInvariantError
from .models import ResourceAudit
from ._compiler import (
    FROZEN_COMPILER_PROFILE,
    TECHNICAL_REPETITIONS,
    CompilationAttempt,
    compile_five_repetitions,
    compose_end_to_end_program,
)
from ._preparation import PreparationBuild


def _median(values: list[int]) -> float:
    return float(statistics.median(values))


def aggregate_resource_audit(
    build: PreparationBuild,
    attempts: tuple[CompilationAttempt, ...],
) -> ResourceAudit:
    if len(attempts) != TECHNICAL_REPETITIONS:
        raise InternalInvariantError(code="resource_audit_requires_five_attempts")
    if tuple(attempt.attempt_index for attempt in attempts) != tuple(range(5)):
        raise InternalInvariantError(code="resource_attempt_index_mismatch")
    if len({attempt.compiler_profile_fingerprint for attempt in attempts}) != 1:
        raise InternalInvariantError(code="mixed_compiler_profiles")
    successes = [attempt for attempt in attempts if attempt.success]
    common = dict(
        required_qubits=build.required_qubits,
        allocated_qubits=attempts[0].allocated_qubits,
        repetitions=TECHNICAL_REPETITIONS,
        ancillas=build.ancillas,
        successful_attempts=len(successes),
        failed_attempts=TECHNICAL_REPETITIONS - len(successes),
        compiler_profile=attempts[0].compiler_profile,
        compiler_profile_fingerprint=attempts[0].compiler_profile_fingerprint,
        compilation_attempts=attempts,
    )
    if len(successes) != TECHNICAL_REPETITIONS:
        return ResourceAudit(
            compiled_two_qubit_gates=None,
            compiled_depth=None,
            compiled_total_gates=None,
            valid=False,
            status="compile_failure",
            failure_reason="one_or_more_technical_repetitions_failed",
            **common,
        )

    def values(field: str) -> list[int]:
        result = [getattr(attempt, field) for attempt in successes]
        if any(value is None for value in result):
            raise InternalInvariantError(code="missing_success_resource_metric")
        return [int(value) for value in result]

    twoq = values("compiled_two_qubit_gates")
    depth = values("compiled_depth")
    total = values("compiled_total_gates")
    oneq = values("compiled_one_qubit_gates")
    cnot = values("compiled_cnot_gates")
    twoq_median = _median(twoq)
    depth_median = _median(depth)
    if twoq != cnot:
        raise InternalInvariantError(code="compiled_two_qubit_cnot_series_mismatch")
    return ResourceAudit(
        compiled_two_qubit_gates=twoq_median,
        compiled_depth=depth_median,
        compiled_total_gates=_median(total),
        compiled_one_qubit_gates=_median(oneq),
        compiled_cnot_gates=_median(cnot),
        two_qubit_range=(float(min(twoq)), float(max(twoq))),
        depth_range=(float(min(depth)), float(max(depth))),
        total_gate_range=(float(min(total)), float(max(total))),
        one_qubit_range=(float(min(oneq)), float(max(oneq))),
        q_required_times_depth=build.required_qubits * depth_median,
        q_allocated_times_depth=attempts[0].allocated_qubits * depth_median,
        valid=True,
        status="valid",
        failure_reason=None,
        **common,
    )


def audit_build_resources(
    build: PreparationBuild,
    *,
    basis: str,
    profile: CompilerConfig = FROZEN_COMPILER_PROFILE,
    transpiler_factory: Callable[[], Any] = Transpiler,
    attempts: tuple[CompilationAttempt, ...] | None = None,
) -> ResourceAudit:
    compilation = compose_end_to_end_program(build, basis=basis, profile=profile)
    actual_attempts = attempts
    if actual_attempts is None:
        actual_attempts = compile_five_repetitions(
            compilation, profile=profile, transpiler_factory=transpiler_factory
        )
    return aggregate_resource_audit(build, actual_attempts)


__all__ = ["aggregate_resource_audit", "audit_build_resources"]
