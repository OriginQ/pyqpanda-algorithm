"""Canonical normalization of measurement outputs.

Bit-string keys use the pyqpanda3 convention throughout: the string is
the binary representation of the computational basis index, with qubit 0
as the least significant (rightmost) character.  Algorithms must use
these helpers instead of re-implementing bit-string ordering or
probability conversion themselves.
"""

from typing import Dict, Mapping, TypeVar

T = TypeVar("T")


def sort_by_basis_index(results: Mapping[str, T]) -> Dict[str, T]:
    """Return ``results`` reordered by ascending computational basis index.

    Sorting by the integer value of the bit string gives a deterministic
    order that does not depend on the insertion order of the source
    mapping.
    """
    return dict(sorted(results.items(), key=lambda item: int(item[0], 2)))


def probability_to_counts(
    probabilities: Mapping[str, float], shots: int
) -> Dict[str, int]:
    """Convert a probability dict into integer counts summing to ``shots``.

    Rounding uses largest-remainder apportionment so the converted
    counts never drift from the requested number of shots.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    exact = {key: probability * shots for key, probability in probabilities.items()}
    counts = {key: int(value) for key, value in exact.items()}
    remainder = shots - sum(counts.values())
    for key, value in sorted(
        exact.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        counts[key] += 1
    return counts
