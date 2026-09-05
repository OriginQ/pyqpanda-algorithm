"""Static and construction capability filters for state preparation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from pyqpanda3.core import CPUQVM

from .config import DEFAULT_METHODS, PreparationMethod
from .exceptions import InternalInvariantError
from .models import CapabilityReport
from ._preparation import (
    PreparationBuild,
    PreparationInput,
    build_preparation,
    required_qubit_contract,
)


LOGICAL_FIDELITY_TOLERANCE = 1e-10
CAPABILITY_REASON_CODES = frozenset(
    {
        "compatible",
        "unsupported_complex",
        "invalid_sparse_keys",
        "support_too_dense",
        "insufficient_qubits",
        "unexpected_output_register",
        "constructor_rejected_input",
        "logical_fidelity_mismatch",
        "unknown_backend_error",
    }
)

_REASONS = {
    "compatible": "The method constructed the requested logical state.",
    "unsupported_complex": "The method does not support this complex state.",
    "invalid_sparse_keys": "Sparse binary keys do not match the logical width.",
    "support_too_dense": "The support violates a method-specific density constraint.",
    "insufficient_qubits": "The available qubits are below the method requirement.",
    "unexpected_output_register": "The constructed output/ancilla metadata is inconsistent.",
    "constructor_rejected_input": "The backend constructor rejected the adapted input.",
    "logical_fidelity_mismatch": "The constructed logical state failed exact fidelity validation.",
    "unknown_backend_error": "The backend or simulator failed unexpectedly.",
}


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    report: CapabilityReport
    build: PreparationBuild | None = None


def _report(
    method: PreparationMethod,
    *,
    compatible: bool,
    reason_code: str,
    failure_stage: str | None,
    required_qubits: int | None,
    ancillas: tuple[int, ...] = (),
    output_qubits: tuple[int, ...] = (),
    logical_fidelity: float | None = None,
    exception: Exception | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> CapabilityReport:
    constraints = [
        "normalized_selected_coefficients",
        "power_of_two_dimension",
        "real_or_complex_supported_pyqpanda3_0_3_5",
    ]
    if method is not PreparationMethod.AMPLITUDE_ENCODE:
        constraints.append("binary_sparse_keys")
    return CapabilityReport(
        method=method,
        compatible=compatible,
        reason_code=reason_code,
        reason=_REASONS[reason_code],
        required_qubits=required_qubits,
        ancillas=ancillas,
        input_constraints=tuple(constraints),
        observed_output_qubits=output_qubits,
        exception_type=type(exception).__name__ if exception is not None else None,
        exception_message=str(exception) if exception is not None else None,
        diagnostics={} if diagnostics is None else diagnostics,
        failure_stage=failure_stage,
        logical_fidelity=logical_fidelity,
    )


def _valid_sparse_keys(prepared_input: PreparationInput) -> bool:
    width = prepared_input.logical_output_qubits
    keys = [key for key, _ in prepared_input.sparse_items]
    return bool(keys) and len(keys) == len(set(keys)) and all(
        isinstance(key, str)
        and len(key) == width
        and set(key) <= {"0", "1"}
        for key in keys
    )


def static_capability_check(
    method: PreparationMethod,
    prepared_input: PreparationInput,
    *,
    available_qubits: int | None = None,
) -> CapabilityReport:
    """Check deterministic method/input constraints before touching PyQPanda."""

    required = required_qubit_contract(
        method, prepared_input.logical_output_qubits
    )
    if method is not PreparationMethod.AMPLITUDE_ENCODE and not _valid_sparse_keys(
        prepared_input
    ):
        return _report(
            method,
            compatible=False,
            reason_code="invalid_sparse_keys",
            failure_stage="static",
            required_qubits=required,
        )
    if available_qubits is not None and available_qubits < required:
        return _report(
            method,
            compatible=False,
            reason_code="insufficient_qubits",
            failure_stage="static",
            required_qubits=required,
            diagnostics={"available_qubits": available_qubits},
        )
    return _report(
        method,
        compatible=True,
        reason_code="compatible",
        failure_stage=None,
        required_qubits=required,
        diagnostics={"stage": "static_only"},
    )


def _metadata_valid(build: PreparationBuild) -> bool:
    outputs = build.output_qubits
    ancillas = build.ancillas
    return (
        build.program is not None
        and build.circuit is not None
        and len(outputs) == build.logical_output_qubits
        and len(set(outputs)) == len(outputs)
        and len(set(ancillas)) == len(ancillas)
        and set(outputs).isdisjoint(ancillas)
        and len(outputs) + len(ancillas) == build.required_qubits
        and set(outputs).union(ancillas) == set(range(build.required_qubits))
        and (
            build.method is not PreparationMethod.DS_QUANTUM_STATE_PREPARATION
            or len(ancillas) == build.logical_output_qubits
        )
        and (
            build.method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION
            or not ancillas
        )
    )


def _logical_state_fidelity(
    build: PreparationBuild,
    expected: np.ndarray,
) -> float:
    qvm = CPUQVM()
    qvm.run(build.program, 1)
    state = np.asarray(qvm.result().get_state_vector(), dtype=np.complex128)
    total_qubits = int(round(math.log2(int(state.size))))
    if 2**total_qubits != state.size or total_qubits != build.required_qubits:
        raise RuntimeError("unexpected simulated state width")

    outputs = list(build.output_qubits)
    ancillas = [qubit for qubit in range(total_qubits) if qubit not in outputs]
    tensor = state.reshape([2] * total_qubits, order="F")
    ordered = np.transpose(tensor, outputs + ancillas).reshape(
        (2 ** len(outputs), -1), order="F"
    )
    reduced = ordered @ ordered.conj().T
    fidelity = float(np.real(np.vdot(expected, reduced @ expected)))
    if not math.isfinite(fidelity):
        raise RuntimeError("non-finite logical fidelity")
    return fidelity


def assess_capability(
    method: PreparationMethod,
    prepared_input: PreparationInput,
    *,
    available_qubits: int | None = None,
    verify_logical_state: bool = True,
) -> CapabilityAssessment:
    """Run static checks, construction, metadata checks, and optional smoke fidelity."""

    static = static_capability_check(
        method, prepared_input, available_qubits=available_qubits
    )
    if not static.compatible:
        return CapabilityAssessment(report=static)

    try:
        build = build_preparation(method, prepared_input)
    except InternalInvariantError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        return CapabilityAssessment(
            report=_report(
                method,
                compatible=False,
                reason_code="constructor_rejected_input",
                failure_stage="construction",
                required_qubits=static.required_qubits,
                exception=error,
            )
        )
    except Exception as error:  # backend bindings expose implementation-defined types
        return CapabilityAssessment(
            report=_report(
                method,
                compatible=False,
                reason_code="unknown_backend_error",
                failure_stage="construction",
                required_qubits=static.required_qubits,
                exception=error,
            )
        )

    if not _metadata_valid(build):
        return CapabilityAssessment(
            report=_report(
                method,
                compatible=False,
                reason_code="unexpected_output_register",
                failure_stage="construction",
                required_qubits=build.required_qubits,
                ancillas=build.ancillas,
                output_qubits=build.output_qubits,
                diagnostics=dict(build.diagnostics),
            ),
            build=build,
        )

    fidelity: float | None = None
    if verify_logical_state:
        try:
            fidelity = _logical_state_fidelity(
                build, np.asarray(prepared_input.coefficients, dtype=np.complex128)
            )
        except InternalInvariantError:
            raise
        except Exception as error:
            return CapabilityAssessment(
                report=_report(
                    method,
                    compatible=False,
                    reason_code="unknown_backend_error",
                    failure_stage="correctness",
                    required_qubits=build.required_qubits,
                    ancillas=build.ancillas,
                    output_qubits=build.output_qubits,
                    exception=error,
                    diagnostics=dict(build.diagnostics),
                ),
                build=build,
            )
        if fidelity < 1.0 - LOGICAL_FIDELITY_TOLERANCE:
            return CapabilityAssessment(
                report=_report(
                    method,
                    compatible=False,
                    reason_code="logical_fidelity_mismatch",
                    failure_stage="correctness",
                    required_qubits=build.required_qubits,
                    ancillas=build.ancillas,
                    output_qubits=build.output_qubits,
                    logical_fidelity=fidelity,
                    diagnostics=dict(build.diagnostics),
                ),
                build=build,
            )

    return CapabilityAssessment(
        report=_report(
            method,
            compatible=True,
            reason_code="compatible",
            failure_stage=None,
            required_qubits=build.required_qubits,
            ancillas=build.ancillas,
            output_qubits=build.output_qubits,
            logical_fidelity=fidelity,
            diagnostics=dict(build.diagnostics),
        ),
        build=build,
    )


def assess_all_capabilities(
    prepared_input: PreparationInput,
    *,
    methods: tuple[PreparationMethod, ...] = DEFAULT_METHODS,
    available_qubits: int | None = None,
    verify_logical_state: bool = True,
) -> tuple[CapabilityAssessment, ...]:
    """Preserve every candidate report, including incompatible methods."""

    return tuple(
        assess_capability(
            method,
            prepared_input,
            available_qubits=available_qubits,
            verify_logical_state=verify_logical_state,
        )
        for method in methods
    )


__all__ = [
    "LOGICAL_FIDELITY_TOLERANCE",
    "CAPABILITY_REASON_CODES",
    "CapabilityAssessment",
    "static_capability_check",
    "assess_capability",
    "assess_all_capabilities",
]
