"""Backend-neutral measurement records for local-Pauli classical shadows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .planner import AXES, _validate_probabilities

_AXIS_TO_INDEX = {axis: index for index, axis in enumerate(AXES)}


@dataclass(frozen=True, slots=True)
class ShadowDataset:
    """Local-Pauli bases, eigenvalue outcomes, and sampling probabilities.

    Arrays use the shape ``(shots, n_qubits)`` except ``probabilities``, whose
    shape is ``(shots, n_qubits, 3)`` in X/Y/Z order.  Keeping the complete
    distribution for every shot allows datasets from different planning rounds
    to be concatenated without bias.
    """

    bases: np.ndarray
    outcomes: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        bases = np.asarray(self.bases, dtype=np.int8).copy()
        outcomes = np.asarray(self.outcomes, dtype=np.int8).copy()
        probabilities = np.asarray(self.probabilities, dtype=float).copy()
        if bases.ndim != 2 or bases.shape[0] == 0 or bases.shape[1] == 0:
            raise ValueError("bases must have shape (positive shots, positive qubits)")
        if outcomes.shape != bases.shape:
            raise ValueError("outcomes must have the same shape as bases")
        if probabilities.shape != (*bases.shape, 3):
            raise ValueError(
                "probabilities must have shape (shots, n_qubits, 3)"
            )
        if np.any((bases < 0) | (bases > 2)):
            raise ValueError("basis indices must be 0 (X), 1 (Y), or 2 (Z)")
        if np.any((outcomes != 1) & (outcomes != -1)):
            raise ValueError("outcomes must be Pauli eigenvalues +1 or -1")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
            raise ValueError("all basis probabilities must be finite and positive")
        if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-10):
            raise ValueError("each per-shot X/Y/Z distribution must sum to one")
        bases.setflags(write=False)
        outcomes.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_strings(
        cls,
        bases: Sequence[str],
        bitstrings: Sequence[str],
        probabilities: Sequence[Sequence[float]] | np.ndarray,
    ) -> "ShadowDataset":
        """Construct records from q0-first basis and outcome strings.

        A bit ``0`` maps to eigenvalue ``+1`` and bit ``1`` to ``-1``.
        ``probabilities`` may be one static ``(n_qubits, 3)`` distribution or a
        per-shot ``(shots, n_qubits, 3)`` array.
        """

        basis_strings = tuple(bases)
        outcome_strings = tuple(bitstrings)
        if not basis_strings or len(basis_strings) != len(outcome_strings):
            raise ValueError("bases and bitstrings must have equal positive length")
        n_qubits = len(basis_strings[0])
        basis_array: np.ndarray = np.empty((len(basis_strings), n_qubits), dtype=np.int8)
        outcome_array: np.ndarray = np.empty_like(basis_array)
        for shot, (basis, bits) in enumerate(zip(basis_strings, outcome_strings)):
            if len(basis) != n_qubits or set(basis) - set(AXES):
                raise ValueError("bases must be equal-length q0-first X/Y/Z strings")
            if len(bits) != n_qubits or set(bits) - {"0", "1"}:
                raise ValueError("bitstrings must be equal-length q0-first binary strings")
            basis_array[shot] = [_AXIS_TO_INDEX[axis] for axis in basis]
            outcome_array[shot] = [1 if bit == "0" else -1 for bit in bits]

        raw_probabilities: np.ndarray = np.asarray(probabilities, dtype=float)
        if raw_probabilities.ndim == 2:
            static = _validate_probabilities(
                raw_probabilities, n_qubits=n_qubits
            )
            probability_array = np.broadcast_to(
                static, (len(basis_strings), n_qubits, 3)
            ).copy()
        elif raw_probabilities.shape == (len(basis_strings), n_qubits, 3):
            probability_array = raw_probabilities.copy()
        else:
            raise ValueError(
                "probabilities must have shape (n_qubits, 3) or "
                "(shots, n_qubits, 3)"
            )
        return cls(basis_array, outcome_array, probability_array)

    @classmethod
    def concatenate(cls, *datasets: "ShadowDataset") -> "ShadowDataset":
        """Combine independently collected rounds with compatible qubit counts."""

        if not datasets:
            raise ValueError("at least one dataset is required")
        if not all(isinstance(dataset, cls) for dataset in datasets):
            raise TypeError("all inputs must be ShadowDataset instances")
        n_qubits = datasets[0].n_qubits
        if any(dataset.n_qubits != n_qubits for dataset in datasets):
            raise ValueError("all datasets must use the same number of qubits")
        return cls(
            np.concatenate([dataset.bases for dataset in datasets], axis=0),
            np.concatenate([dataset.outcomes for dataset in datasets], axis=0),
            np.concatenate([dataset.probabilities for dataset in datasets], axis=0),
        )

    @property
    def shots(self) -> int:
        return int(self.bases.shape[0])

    @property
    def n_qubits(self) -> int:
        return int(self.bases.shape[1])

    @property
    def basis_strings(self) -> tuple[str, ...]:
        return tuple("".join(AXES[index] for index in row) for row in self.bases)

    @property
    def bitstrings(self) -> tuple[str, ...]:
        return tuple(
            "".join("0" if value == 1 else "1" for value in row)
            for row in self.outcomes
        )

    def take(self, indices: Sequence[int] | slice) -> "ShadowDataset":
        """Return a validated shot subset."""

        bases = self.bases[indices]
        outcomes = self.outcomes[indices]
        probabilities = self.probabilities[indices]
        if bases.ndim == 1:
            bases = bases[None, :]
            outcomes = outcomes[None, :]
            probabilities = probabilities[None, :, :]
        return ShadowDataset(bases, outcomes, probabilities)
