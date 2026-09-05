"""Immutable schema skeletons for QSEncode-Insight v1.

The types reserve the public result contract without fabricating Phase 2+
scientific values.  Nested records use tuples for collection fields; the
top-level analysis-config mapping is a serialized provenance snapshot and must
be treated as read-only by producers and consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .config import (
    SCHEMA_VERSION,
    EvidenceScopeStatus,
    PreparationMethod,
    SelectionDecision,
    VerificationLevel,
)
from .exceptions import SerializationError


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_deep_freeze(item) for item in value), key=repr))
    return value


def _freeze_fields(instance: Any, *names: str) -> None:
    for name in names:
        value = getattr(instance, name)
        if value is not None:
            object.__setattr__(instance, name, _deep_freeze(value))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError(code="nonfinite_result_float")
        return value
    raise SerializationError(
        code="unsupported_result_value",
        message=f"Unsupported structured-result value: {type(value).__name__}",
    )


@dataclass(frozen=True, slots=True)
class InputSummary:
    original_length: int
    padded_length: int
    original_sum: float
    normalized: bool
    padding_count: int
    original_input_sha256: str
    effective_probability_sha256: str


@dataclass(frozen=True, slots=True)
class TransformDiagnostics:
    basis: str
    coefficient_count: int
    parseval_error: float
    implementation: str
    normalization: str
    oracle_checked: bool
    oracle_max_abs_error: float | None = None


@dataclass(frozen=True, slots=True)
class ErrorBudgetResult:
    fidelity_target: float
    k_star: int
    retained_energy: float
    previous_retained_energy: float
    candidate_k: tuple[int, ...] = ()
    minimality_pass: bool = False
    ranking_policy: str = "frozen_stable_v1"

    def __post_init__(self) -> None:
        _freeze_fields(self, "candidate_k")


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    method: PreparationMethod
    compatible: bool
    reason_code: str | None = None
    reason: str | None = None
    required_qubits: int | None = None
    ancillas: tuple[int, ...] = ()
    input_constraints: tuple[str, ...] = ()
    observed_output_qubits: tuple[int, ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None
    diagnostics: Mapping[str, Any] | None = None
    failure_stage: str | None = None
    logical_fidelity: float | None = None

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            "ancillas",
            "input_constraints",
            "observed_output_qubits",
            "diagnostics",
        )


@dataclass(frozen=True, slots=True)
class ResourceAudit:
    compiled_two_qubit_gates: float | None
    compiled_depth: float | None
    compiled_total_gates: float | None
    required_qubits: int
    allocated_qubits: int
    repetitions: int
    compiled_one_qubit_gates: float | None = None
    compiled_cnot_gates: float | None = None
    ancillas: tuple[int, ...] = ()
    two_qubit_range: tuple[float, float] | None = None
    depth_range: tuple[float, float] | None = None
    total_gate_range: tuple[float, float] | None = None
    one_qubit_range: tuple[float, float] | None = None
    q_required_times_depth: float | None = None
    q_allocated_times_depth: float | None = None
    successful_attempts: int = 0
    failed_attempts: int = 0
    compiler_profile: Mapping[str, Any] | None = None
    compiler_profile_fingerprint: str | None = None
    valid: bool = False
    status: str = "not_run"
    failure_reason: str | None = None
    compilation_attempts: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "compiler_profile")
        _freeze_fields(
            self,
            "ancillas",
            "two_qubit_range",
            "depth_range",
            "total_gate_range",
            "one_qubit_range",
            "compilation_attempts",
        )


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate_id: str
    method: PreparationMethod
    k: int
    status: str
    verified_fidelity: float | None = None
    resource_audit: ResourceAudit | None = None
    failure_reason: str | None = None
    capability: CapabilityReport | None = None
    retained_fidelity: float | None = None
    eligible: bool = False
    eligibility_reason: str | None = None
    role: str = "compressed"


@dataclass(frozen=True, slots=True)
class SelectionResult:
    decision: SelectionDecision
    reason_code: str
    selected_candidate_id: str | None = None
    method: PreparationMethod | None = None
    k: int | None = None
    reason: str | None = None
    baseline_resource: ResourceAudit | None = None
    best_compressed_candidate_id: str | None = None
    best_compressed_resource: ResourceAudit | None = None
    comparison_metrics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _freeze_fields(self, "comparison_metrics")


@dataclass(frozen=True, slots=True)
class SemanticVerificationAttempt:
    attempt_index: int
    status: str
    fidelity: float | None
    mapping_method: str | None
    output_qubits: tuple[int, ...]
    mapping: tuple[int, ...] = ()
    ancilla_treatment: str = "partial_trace"
    exception_type: str | None = None
    exception_message: str | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _freeze_fields(self, "output_qubits", "mapping", "diagnostics")


@dataclass(frozen=True, slots=True)
class SemanticVerification:
    level: VerificationLevel
    status: str
    recommendation_valid: bool
    minimum_fidelity: float | None = None
    technical_repetitions: int = 0
    selected_candidate_id: str | None = None
    attempts: tuple[SemanticVerificationAttempt, ...] = ()

    def __post_init__(self) -> None:
        _freeze_fields(self, "attempts")


@dataclass(frozen=True, slots=True)
class AttributionReport:
    truncation_two_qubit_gain: float | None = None
    preparation_two_qubit_gain: float | None = None
    truncation_depth_gain: float | None = None
    preparation_depth_gain: float | None = None
    total_two_qubit_difference: float | None = None
    truncation_two_qubit_difference: float | None = None
    preparation_two_qubit_difference: float | None = None
    total_depth_difference: float | None = None
    truncation_depth_difference: float | None = None
    preparation_depth_difference: float | None = None
    two_qubit_identity_error: float | None = None
    depth_identity_error: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    status: EvidenceScopeStatus
    reasons: tuple[str, ...]
    reference: str | None

    def __post_init__(self) -> None:
        _freeze_fields(self, "reasons")


@dataclass(frozen=True, slots=True)
class InsightResult:
    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    input: InputSummary
    analysis_config: Mapping[str, Any]
    analysis_config_fingerprint: str
    transform: TransformDiagnostics | None = None
    error_budget: ErrorBudgetResult | None = None
    capabilities: tuple[CapabilityReport, ...] = ()
    candidates: tuple[CandidateResult, ...] = ()
    selection: SelectionResult | None = None
    semantic_verification: SemanticVerification | None = None
    attribution: AttributionReport | None = None
    evidence_scope: EvidenceScope | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("unsupported InsightResult schema_version")
        _freeze_fields(self, "analysis_config", "capabilities", "candidates")

    def to_dict(self) -> dict[str, Any]:
        serialized = _serialize(self)
        if not isinstance(serialized, dict):
            raise SerializationError(code="invalid_result_root")
        return serialized

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )


@dataclass(frozen=True, slots=True)
class PreparationArtifact:
    program: Any
    output_qubits: tuple[int, ...]
    ancillas: tuple[int, ...]
    selected_candidate_id: str
    decision: SelectionDecision
    decision_reason: str
    basis: str
    k: int | None
    compiler_metadata: Mapping[str, Any]
    verification_status: str
    evidence_scope: EvidenceScope
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            "output_qubits",
            "ancillas",
            "compiler_metadata",
            "provenance",
        )


__all__ = [
    "InputSummary",
    "TransformDiagnostics",
    "ErrorBudgetResult",
    "CapabilityReport",
    "CandidateResult",
    "SelectionResult",
    "ResourceAudit",
    "SemanticVerificationAttempt",
    "SemanticVerification",
    "AttributionReport",
    "EvidenceScope",
    "InsightResult",
    "PreparationArtifact",
]
