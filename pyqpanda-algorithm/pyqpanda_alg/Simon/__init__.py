"""Simon's hidden-XOR algorithm and GF(2) recovery utilities."""

from .simon import (
    dot_mod2,
    equations_from_probability_list,
    gf2_rank,
    independent_equations,
    index_to_q0_bits,
    linear_simon_function,
    linear_simon_oracle,
    q0_bits_to_index,
    recover_secret_from_counts,
    recover_secret_from_equations,
    recover_secret_from_probabilities,
    run_simon,
    simon_circuit,
    simon_reference_probabilities,
)

__all__ = [
    "dot_mod2",
    "equations_from_probability_list",
    "gf2_rank",
    "independent_equations",
    "index_to_q0_bits",
    "linear_simon_function",
    "linear_simon_oracle",
    "q0_bits_to_index",
    "recover_secret_from_counts",
    "recover_secret_from_equations",
    "recover_secret_from_probabilities",
    "run_simon",
    "simon_circuit",
    "simon_reference_probabilities",
]
