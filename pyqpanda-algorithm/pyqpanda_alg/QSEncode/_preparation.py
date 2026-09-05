"""Private state-preparation adapters for QSEncode-Insight Phase 3.

The adapters consume an already selected and normalized coefficient vector.
They never rank coefficients, choose ``k``, transpile, or inspect resources.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyqpanda3.core import Encode, QCircuit, QProg

from .config import PreparationMethod
from .exceptions import InputValidationError, InternalInvariantError


# Frozen provenance: final_technical_hardening.py, SHA-256
# B24336428BD62DFF3C479B5C1207329BC61AA68B92009E77A2EE6537981D11A8.
# This is representation cleanup after Phase 2's exact top-k mask; it does not
# participate in retained-energy or k-star selection.
SPARSE_SUPPORT_THRESHOLD = 1e-14
STATE_NORM_ATOL = 1e-12


Scalar = float | complex


@dataclass(frozen=True, slots=True)
class PreparationInput:
    """Normalized selected coefficients plus deterministic API representations."""

    coefficients: NDArray[np.float64] | NDArray[np.complex128]
    logical_dimension: int
    logical_output_qubits: int
    dense_data: tuple[Scalar, ...]
    sparse_items: tuple[tuple[str, Scalar], ...]
    support_indices: tuple[int, ...]
    support_threshold: float
    is_complex: bool

    def sparse_data(self) -> dict[str, Scalar]:
        """Return a fresh mutable mapping for the PyQPanda binding."""

        return dict(self.sparse_items)


@dataclass(frozen=True, slots=True)
class PreparationBuild:
    """Internal construction result, not the public PreparationArtifact."""

    method: PreparationMethod
    program: QProg
    circuit: QCircuit
    output_qubits: tuple[int, ...]
    ancillas: tuple[int, ...]
    required_qubits: int
    logical_output_qubits: int
    input_representation: str
    status: str
    diagnostics: Mapping[str, Any]


def _is_power_of_two(value: int) -> bool:
    return value >= 2 and value & (value - 1) == 0


def _python_scalar(value: np.generic) -> Scalar:
    converted = value.item()
    return complex(converted) if isinstance(converted, complex) else float(converted)


def adapt_preparation_input(coefficients: ArrayLike) -> PreparationInput:
    """Validate and adapt an already normalized selected state.

    Complex values are preserved.  The caller-owned object is never modified.
    """

    try:
        source = np.asarray(coefficients)
        dtype = np.complex128 if np.iscomplexobj(source) else np.float64
        values = np.array(source, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(code="state_not_numeric") from error
    if values.ndim != 1 or not _is_power_of_two(int(values.size)):
        raise InputValidationError(code="invalid_state_dimension")
    if not np.all(np.isfinite(values)):
        raise InputValidationError(code="nonfinite_state")

    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or abs(norm - 1.0) > STATE_NORM_ATOL:
        raise InputValidationError(code="state_not_normalized")

    logical_dimension = int(values.size)
    logical_qubits = logical_dimension.bit_length() - 1
    support = tuple(
        int(index)
        for index in np.flatnonzero(np.abs(values) > SPARSE_SUPPORT_THRESHOLD)
    )
    if not support:
        raise InternalInvariantError(code="empty_preparation_support")

    dense = tuple(_python_scalar(value) for value in values)
    sparse = tuple(
        (format(index, f"0{logical_qubits}b"), _python_scalar(values[index]))
        for index in support
    )
    values.setflags(write=False)
    return PreparationInput(
        coefficients=values,
        logical_dimension=logical_dimension,
        logical_output_qubits=logical_qubits,
        dense_data=dense,
        sparse_items=sparse,
        support_indices=support,
        support_threshold=SPARSE_SUPPORT_THRESHOLD,
        is_complex=bool(np.iscomplexobj(values)),
    )


def required_qubit_contract(method: PreparationMethod, logical_qubits: int) -> int:
    """Return the PyQPanda 0.3.5/Frozen v1 allocation contract."""

    if method in {
        PreparationMethod.AMPLITUDE_ENCODE,
        PreparationMethod.SPARSE_ISOMETRY,
    }:
        return logical_qubits
    if method is PreparationMethod.DS_QUANTUM_STATE_PREPARATION:
        return 2 * logical_qubits
    raise InternalInvariantError(code="unsupported_preparation_method")


def _normalize_backend_output_register(
    method: PreparationMethod,
    allocated_qubits: tuple[int, ...],
    reported: tuple[int, ...],
    logical_qubits: int,
    *,
    is_complex: bool,
) -> tuple[tuple[int, ...], str | None]:
    if len(reported) == logical_qubits and len(set(reported)) == logical_qubits:
        return reported, None

    # PyQPanda3 0.3.5's complex amplitude overload reports the same output
    # register twice.  The circuit itself acts on exactly the supplied n-qubit
    # register.  This normalization is deliberately narrow and is retained in
    # diagnostics; any other malformed register remains a capability failure.
    if (
        method is PreparationMethod.AMPLITUDE_ENCODE
        and is_complex
        and reported == allocated_qubits + allocated_qubits
    ):
        return allocated_qubits, "deduplicated_exact_repetition"
    return reported, None


def build_preparation(
    method: PreparationMethod,
    prepared_input: PreparationInput,
) -> PreparationBuild:
    """Invoke exactly one supported PyQPanda state-preparation constructor."""

    if not isinstance(method, PreparationMethod):
        raise InternalInvariantError(code="unsupported_preparation_method")
    if not isinstance(prepared_input, PreparationInput):
        raise InternalInvariantError(code="invalid_preparation_input")

    required = required_qubit_contract(
        method, prepared_input.logical_output_qubits
    )
    allocated = tuple(range(required))
    encoder = Encode()

    if method is PreparationMethod.AMPLITUDE_ENCODE:
        representation = "dense_list"
        return_value = encoder.amplitude_encode(list(allocated), list(prepared_input.dense_data))
    elif method is PreparationMethod.SPARSE_ISOMETRY:
        representation = "sparse_binary_map"
        return_value = encoder.sparse_isometry(
            list(allocated), prepared_input.sparse_data()
        )
    else:
        representation = "sparse_binary_map"
        return_value = encoder.ds_quantum_state_preparation(
            list(allocated), prepared_input.sparse_data()
        )

    circuit = encoder.get_circuit()
    raw_output = tuple(int(qubit) for qubit in encoder.get_out_qubits())
    output, normalization = _normalize_backend_output_register(
        method,
        allocated,
        raw_output,
        prepared_input.logical_output_qubits,
        is_complex=prepared_input.is_complex,
    )
    ancillas = tuple(qubit for qubit in allocated if qubit not in output)
    program = QProg(required)
    program << circuit
    diagnostics: dict[str, Any] = {
        "backend": "pyqpanda3.Encode",
        "backend_version": "0.3.5",
        "backend_return_value": repr(return_value),
        "backend_reported_output_qubits": raw_output,
        "allocated_qubits": allocated,
        "support_size": len(prepared_input.support_indices),
    }
    if normalization is not None:
        diagnostics["output_register_normalization"] = normalization

    return PreparationBuild(
        method=method,
        program=program,
        circuit=circuit,
        output_qubits=output,
        ancillas=ancillas,
        required_qubits=required,
        logical_output_qubits=prepared_input.logical_output_qubits,
        input_representation=representation,
        status="success",
        diagnostics=diagnostics,
    )


__all__ = [
    "SPARSE_SUPPORT_THRESHOLD",
    "STATE_NORM_ATOL",
    "PreparationInput",
    "PreparationBuild",
    "adapt_preparation_input",
    "required_qubit_contract",
    "build_preparation",
]
