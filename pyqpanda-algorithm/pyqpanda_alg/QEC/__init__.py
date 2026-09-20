"""Three-qubit repetition quantum error-correction codes."""

from .repetition_code import (
    ThreeQubitRepetitionCode,
    bit_flip_encode,
    bit_flip_recover,
    expected_syndrome,
    inject_single_pauli_error,
    logical_density_matrix,
    logical_fidelity,
    phase_flip_encode,
    phase_flip_recover,
    reference_repetition_recovery,
    syndrome_probabilities,
)

__all__ = [
    "ThreeQubitRepetitionCode",
    "bit_flip_encode",
    "bit_flip_recover",
    "expected_syndrome",
    "inject_single_pauli_error",
    "logical_density_matrix",
    "logical_fidelity",
    "phase_flip_encode",
    "phase_flip_recover",
    "reference_repetition_recovery",
    "syndrome_probabilities",
]
