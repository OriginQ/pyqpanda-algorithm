"""Frozen Phase 1 configuration and fingerprint contracts.

This module contains no probability processing, transform, candidate,
transpilation, selection, or verification implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping

from .exceptions import ConfigurationError


SCHEMA_VERSION = "qseencode-insight-v1"
SELECTION_POLICY = "frozen_lexicographic_v1"
_FINGERPRINT_DOMAIN = b"qseencode-analysis-config-v1\0"


class PreparationMethod(str, Enum):
    AMPLITUDE_ENCODE = "amplitude_encode"
    SPARSE_ISOMETRY = "sparse_isometry"
    DS_QUANTUM_STATE_PREPARATION = "ds_quantum_state_preparation"


DEFAULT_METHODS = (
    PreparationMethod.AMPLITUDE_ENCODE,
    PreparationMethod.SPARSE_ISOMETRY,
    PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
)


class EvidenceScopeStatus(str, Enum):
    VALIDATED_DEFAULT = "validated_default"
    OUTSIDE_VALIDATED_SCOPE = "outside_validated_scope"


class VerificationLevel(str, Enum):
    STANDARD = "standard"
    AUDIT = "audit"


class SelectionDecision(str, Enum):
    COMPRESS = "compress"
    DO_NOT_COMPRESS = "do_not_compress"


@dataclass(frozen=True, slots=True)
class InputPolicy:
    normalization: str = "normalize"
    padding: str = "next_power_of_two"
    normalization_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.normalization not in {"normalize", "strict"}:
            raise ConfigurationError(code="invalid_normalization_policy")
        if self.padding not in {"next_power_of_two", "reject"}:
            raise ConfigurationError(code="invalid_padding_policy")
        if (
            not isinstance(self.normalization_tolerance, float)
            or not math.isfinite(self.normalization_tolerance)
            or self.normalization_tolerance < 0.0
        ):
            raise ConfigurationError(code="invalid_normalization_tolerance")


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    pyqpanda_version: str = "0.3.5"
    topology: str = "linear"
    physical_capacity_multiplier: int = 2
    initial_mapping: str = "identity"
    optimization_level: int = 2
    basis_gates: tuple[str, ...] = ("U3", "CNOT")
    technical_repetitions: int = 5
    resource_aggregation: str = "median_with_range"

    def __post_init__(self) -> None:
        if self.technical_repetitions <= 0:
            raise ConfigurationError(code="invalid_technical_repetitions")
        if self.physical_capacity_multiplier <= 0:
            raise ConfigurationError(code="invalid_physical_capacity")
        if not self.basis_gates:
            raise ConfigurationError(code="empty_basis_gates")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    basis: str
    fidelity_target: float = 0.99
    verification: VerificationLevel = VerificationLevel.STANDARD
    methods: tuple[PreparationMethod, ...] = DEFAULT_METHODS
    selection_policy: str = SELECTION_POLICY
    compiler: CompilerConfig = CompilerConfig()
    input_policy: InputPolicy = InputPolicy()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.basis not in {"walsh", "fourier"}:
            raise ConfigurationError(code="invalid_basis")
        if (
            not isinstance(self.fidelity_target, float)
            or not math.isfinite(self.fidelity_target)
            or not 0.0 < self.fidelity_target <= 1.0
        ):
            raise ConfigurationError(code="invalid_fidelity_target")
        if not isinstance(self.verification, VerificationLevel):
            raise ConfigurationError(code="invalid_verification")
        if not self.methods or not all(
            isinstance(method, PreparationMethod) for method in self.methods
        ):
            raise ConfigurationError(code="invalid_methods")
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigurationError(code="unsupported_schema_version")

    def canonical_payload(self) -> dict[str, Any]:
        """Return the canonical, JSON-ready v1 configuration snapshot."""

        payload = _canonicalize(self)
        if not isinstance(payload, dict):  # defensive type narrowing
            raise ConfigurationError(code="invalid_analysis_config")
        return payload


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ConfigurationError(code="non_string_config_key")
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(code="nonfinite_config_float")
        return value.hex()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ConfigurationError(code="unsupported_config_value")


def canonical_analysis_config_json(config: AnalysisConfig | Mapping[str, Any]) -> str:
    """Serialize a scientific configuration under the ADR v1 canonical rules."""

    canonical = _canonicalize(config)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def analysis_config_fingerprint(
    config: AnalysisConfig | Mapping[str, Any],
) -> str:
    """Return the stable ADR v1 analysis-configuration SHA-256."""

    canonical = canonical_analysis_config_json(config).encode("utf-8")
    return hashlib.sha256(_FINGERPRINT_DOMAIN + canonical).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "SELECTION_POLICY",
    "PreparationMethod",
    "DEFAULT_METHODS",
    "EvidenceScopeStatus",
    "VerificationLevel",
    "SelectionDecision",
    "InputPolicy",
    "CompilerConfig",
    "AnalysisConfig",
    "canonical_analysis_config_json",
    "analysis_config_fingerprint",
]
