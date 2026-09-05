"""Public orchestration facade for QSEncode-Insight v1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import numpy as np

from .config import (
    DEFAULT_METHODS,
    AnalysisConfig,
    CompilerConfig,
    EvidenceScopeStatus,
    InputPolicy,
    PreparationMethod,
    VerificationLevel,
    analysis_config_fingerprint,
)
from .exceptions import (
    ConfigurationError,
    InternalInvariantError,
    ResultBindingError,
    UncertifiedSelectionError,
)
from .models import (
    CandidateResult,
    EvidenceScope,
    InsightResult,
    PreparationArtifact,
    ResourceAudit,
    SelectionResult,
)
from ._compiler import (
    FROZEN_COMPILER_PROFILE,
    compiler_profile_payload,
    compose_end_to_end_program,
)
from ._error_budget import RANKING_POLICY, find_k_star, top_k_coefficients
from ._preparation import adapt_preparation_input, build_preparation
from ._selection import ResourceSelectionRun, run_resource_selection
from ._transforms import analyze_transform
from ._validation import ValidatedInput, canonicalize_probabilities
from ._verification import verify_resource_selection


class QSEncodeInsight:
    """Analyze and prepare one explicitly selected Walsh or Fourier mode."""

    def __init__(
        self,
        *,
        basis: Literal["walsh", "fourier"],
        fidelity_target: float = 0.99,
        verification: Literal["standard", "audit"] = "standard",
        methods: Sequence[PreparationMethod] = DEFAULT_METHODS,
        compiler: CompilerConfig | None = None,
        input_policy: InputPolicy | None = None,
    ) -> None:
        try:
            verification_level = VerificationLevel(verification)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(code="invalid_verification") from error

        try:
            method_tuple = tuple(
                method
                if isinstance(method, PreparationMethod)
                else PreparationMethod(method)
                for method in methods
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(code="invalid_methods") from error

        if compiler is not None and not isinstance(compiler, CompilerConfig):
            raise ConfigurationError(code="invalid_compiler_config")
        if input_policy is not None and not isinstance(input_policy, InputPolicy):
            raise ConfigurationError(code="invalid_input_policy")

        self._config = AnalysisConfig(
            basis=basis,
            fidelity_target=fidelity_target,
            verification=verification_level,
            methods=method_tuple,
            compiler=compiler if compiler is not None else CompilerConfig(),
            input_policy=input_policy if input_policy is not None else InputPolicy(),
        )
        self._analysis_config_fingerprint = analysis_config_fingerprint(self._config)

    @property
    def basis(self) -> str:
        return self._config.basis

    @property
    def fidelity_target(self) -> float:
        return self._config.fidelity_target

    @property
    def verification(self) -> VerificationLevel:
        return self._config.verification

    @property
    def methods(self) -> tuple[PreparationMethod, ...]:
        return self._config.methods

    @property
    def compiler(self) -> CompilerConfig:
        return self._config.compiler

    @property
    def input_policy(self) -> InputPolicy:
        return self._config.input_policy

    @property
    def analysis_config_fingerprint(self) -> str:
        return self._analysis_config_fingerprint

    def analyze(self, probabilities: Any) -> InsightResult:
        validated = canonicalize_probabilities(
            probabilities, policy=self.input_policy
        )
        amplitudes = np.sqrt(validated.probabilities)
        coefficients, transform = analyze_transform(
            amplitudes,
            basis=self.basis,
            oracle_check=self.basis == "walsh" and validated.probabilities.size <= 64,
        )
        error_budget = find_k_star(coefficients, self.fidelity_target)
        resource_run = run_resource_selection(
            coefficients,
            basis=self.basis,
            error_budget=error_budget,
            profile=self.compiler,
            methods=self.methods,
        )
        verification = verify_resource_selection(
            coefficients,
            basis=self.basis,
            run=resource_run,
            level=self.verification,
        )
        evidence = self._evidence_scope(
            padded_length=validated.summary.padded_length,
            transform_implementation=transform.implementation,
            ranking_policy=error_budget.ranking_policy,
        )
        candidates = tuple(
            _public_candidate(candidate) for candidate in resource_run.grid.candidates
        )
        capabilities = _unique_capabilities(candidates)
        return InsightResult(
            input=validated.summary,
            analysis_config=self._config.canonical_payload(),
            analysis_config_fingerprint=self.analysis_config_fingerprint,
            transform=transform,
            error_budget=error_budget,
            capabilities=capabilities,
            candidates=candidates,
            selection=_public_selection(resource_run.selection),
            semantic_verification=verification,
            attribution=resource_run.attribution,
            evidence_scope=evidence,
        )

    def prepare(
        self,
        probabilities: Any,
        *,
        result: InsightResult | None = None,
        fallback_to_baseline: bool = False,
    ) -> PreparationArtifact:
        if result is None:
            return self.prepare(
                probabilities,
                result=self.analyze(probabilities),
                fallback_to_baseline=fallback_to_baseline,
            )

        validated = canonicalize_probabilities(
            probabilities, policy=self.input_policy
        )
        self._validate_result_binding(validated, result)
        if result.selection is None or result.semantic_verification is None:
            raise ResultBindingError(code="configuration_mismatch")

        audit_invalid = (
            self.verification is VerificationLevel.AUDIT
            and not result.semantic_verification.recommendation_valid
        )
        if audit_invalid and not fallback_to_baseline:
            raise UncertifiedSelectionError(code="audit_recommendation_invalid")

        coefficients, _ = analyze_transform(
            np.sqrt(validated.probabilities), basis=self.basis, oracle_check=False
        )
        fallback = bool(audit_invalid and fallback_to_baseline)
        if fallback or result.selection.decision.value == "do_not_compress":
            candidate_id = "dense_full__amplitude_encode"
            method = PreparationMethod.AMPLITUDE_ENCODE
            selected_coefficients = coefficients
            artifact_k = None
            audit = result.selection.baseline_resource
        else:
            candidate_id = result.selection.selected_candidate_id
            if candidate_id is None or result.selection.method is None or result.selection.k is None:
                raise InternalInvariantError(code="selected_artifact_metadata_missing")
            method = result.selection.method
            artifact_k = result.selection.k
            selected_coefficients = top_k_coefficients(
                coefficients, artifact_k, normalize=True
            )
            selected = next(
                (item for item in result.candidates if item.candidate_id == candidate_id),
                None,
            )
            if selected is None:
                raise InternalInvariantError(code="selected_artifact_candidate_missing")
            audit = selected.resource_audit

        prepared = adapt_preparation_input(selected_coefficients)
        build = build_preparation(method, prepared)
        end_to_end = compose_end_to_end_program(
            build, basis=self.basis, profile=self.compiler
        )
        provenance: dict[str, Any] = {
            "analysis_config_fingerprint": result.analysis_config_fingerprint,
            "effective_probability_sha256": result.input.effective_probability_sha256,
            "artifact_kind": (
                "fallback_from_uncertified_selection"
                if fallback
                else "dense_full_baseline"
                if result.selection.decision.value == "do_not_compress"
                else "selected_compressed_candidate"
            ),
            "selection_reason_code": result.selection.reason_code,
        }
        if fallback:
            provenance.update(
                {
                    "original_selected_candidate_id": result.selection.selected_candidate_id,
                    "audit_status": result.semantic_verification.status,
                    "fallback_explicitly_requested": True,
                }
            )
        compiler_metadata = {
            "profile": compiler_profile_payload(self.compiler),
            "profile_fingerprint": (
                audit.compiler_profile_fingerprint if audit is not None else None
            ),
        }
        verification_status = (
            "fallback_from_uncertified_selection"
            if fallback
            else "audit_certified_5_of_5"
            if self.verification is VerificationLevel.AUDIT
            else "standard_validated"
        )
        return PreparationArtifact(
            program=end_to_end.program,
            output_qubits=end_to_end.output_qubits,
            ancillas=end_to_end.ancillas,
            selected_candidate_id=candidate_id,
            decision=result.selection.decision,
            decision_reason=result.selection.reason_code,
            basis=self.basis,
            k=artifact_k,
            compiler_metadata=compiler_metadata,
            verification_status=verification_status,
            evidence_scope=result.evidence_scope,
            provenance=provenance,
        )

    def _validate_result_binding(
        self, validated: ValidatedInput, result: InsightResult
    ) -> None:
        if not isinstance(result, InsightResult):
            raise ResultBindingError(code="configuration_mismatch")
        if result.schema_version != InsightResult.SCHEMA_VERSION:
            raise ResultBindingError(code="configuration_mismatch")
        if (
            validated.summary.effective_probability_sha256
            != result.input.effective_probability_sha256
        ):
            raise ResultBindingError(code="input_mismatch")
        if self.analysis_config_fingerprint != result.analysis_config_fingerprint:
            raise ResultBindingError(code="configuration_mismatch")

    def _evidence_scope(
        self,
        *,
        padded_length: int,
        transform_implementation: str,
        ranking_policy: str,
    ) -> EvidenceScope:
        reasons: list[str] = []
        if self.fidelity_target != 0.99:
            reasons.append("fidelity_target_not_0.99")
        if self.methods != DEFAULT_METHODS:
            reasons.append("methods_or_order_not_frozen_default")
        if self._config.selection_policy != "frozen_lexicographic_v1":
            reasons.append("selection_policy_not_frozen_default")
        if self.compiler != FROZEN_COMPILER_PROFILE:
            reasons.append("compiler_profile_not_frozen_default")
        if padded_length not in {8, 16, 32, 64}:
            reasons.append("dimension_outside_validated_scope")
        expected_transform = (
            "iterative_fwht_v1" if self.basis == "walsh" else "scipy_fft_ortho_v1"
        )
        if transform_implementation != expected_transform:
            reasons.append("transform_convention_mismatch")
        if ranking_policy != RANKING_POLICY:
            reasons.append("ranking_policy_mismatch")
        try:
            runtime_version = version("pyqpanda3")
        except PackageNotFoundError:
            runtime_version = "unavailable"
        if runtime_version != "0.3.5":
            reasons.append("pyqpanda_runtime_not_0.3.5")
        status = (
            EvidenceScopeStatus.VALIDATED_DEFAULT
            if not reasons
            else EvidenceScopeStatus.OUTSIDE_VALIDATED_SCOPE
        )
        return EvidenceScope(
            status=status,
            reasons=tuple(reasons),
            reference="Generalization Benchmark Protocol v1.1 / Locked Test PASS",
        )


def _public_attempt(attempt: Any) -> dict[str, Any]:
    return {
        "attempt_index": attempt.attempt_index,
        "success": attempt.success,
        "status": attempt.status,
        "originir_sha256": attempt.originir_sha256,
        "compiled_depth": attempt.compiled_depth,
        "compiled_total_gates": attempt.compiled_total_gates,
        "compiled_one_qubit_gates": attempt.compiled_one_qubit_gates,
        "compiled_two_qubit_gates": attempt.compiled_two_qubit_gates,
        "compiled_cnot_gates": attempt.compiled_cnot_gates,
        "exception_type": attempt.exception_type,
        "exception_message": attempt.exception_message,
        "diagnostics": attempt.diagnostics,
    }


def _public_resource(audit: ResourceAudit | None) -> ResourceAudit | None:
    if audit is None:
        return None
    return replace(
        audit,
        compilation_attempts=tuple(
            _public_attempt(attempt) for attempt in audit.compilation_attempts
        ),
    )


def _public_candidate(candidate: CandidateResult) -> CandidateResult:
    return replace(candidate, resource_audit=_public_resource(candidate.resource_audit))


def _public_selection(selection: SelectionResult) -> SelectionResult:
    return replace(
        selection,
        baseline_resource=_public_resource(selection.baseline_resource),
        best_compressed_resource=_public_resource(selection.best_compressed_resource),
    )


def _unique_capabilities(
    candidates: tuple[CandidateResult, ...],
) -> tuple[Any, ...]:
    seen: set[PreparationMethod] = set()
    reports = []
    for candidate in candidates:
        if candidate.capability is not None and candidate.method not in seen:
            reports.append(candidate.capability)
            seen.add(candidate.method)
    return tuple(reports)


__all__ = ["QSEncodeInsight"]
