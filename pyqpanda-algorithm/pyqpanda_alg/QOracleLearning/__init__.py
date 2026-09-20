"""Oracle-learning algorithms: Deutsch-Jozsa and Bernstein-Vazirani."""

from .oracle_learning import (
    affine_phase_oracle,
    affine_truth_table,
    bernstein_vazirani_circuit,
    classify_deutsch_jozsa_promise,
    deutsch_jozsa_circuit,
    dominant_q0_bitstring,
    index_to_q0_bits,
    input_qubit_count,
    phase_oracle_diagonal,
    q0_bits_to_index,
    recover_affine_secret,
    run_bernstein_vazirani,
    run_deutsch_jozsa,
    truth_table_phase_oracle,
    validate_truth_table,
    walsh_probabilities,
)

__all__ = [
    "affine_phase_oracle",
    "affine_truth_table",
    "bernstein_vazirani_circuit",
    "classify_deutsch_jozsa_promise",
    "deutsch_jozsa_circuit",
    "dominant_q0_bitstring",
    "index_to_q0_bits",
    "input_qubit_count",
    "phase_oracle_diagonal",
    "q0_bits_to_index",
    "recover_affine_secret",
    "run_bernstein_vazirani",
    "run_deutsch_jozsa",
    "truth_table_phase_oracle",
    "validate_truth_table",
    "walsh_probabilities",
]
