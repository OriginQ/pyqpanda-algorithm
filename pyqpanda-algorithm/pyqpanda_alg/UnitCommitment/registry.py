"""Stable binary-variable ordering and QAOA bit-string conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .data_model import UnitCommitmentInstance


@dataclass(frozen=True, order=True)
class VariableKey:
    kind: str
    unit: str
    period: int

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.unit}_t{self.period}"


class VariableRegistry:
    """Map semantic variables to the lexicographic order used by SymPy/QAOA."""

    def __init__(self, keys: Sequence[VariableKey]):
        ordered = tuple(sorted(keys, key=lambda key: key.name))
        if len({key.name for key in ordered}) != len(ordered):
            raise ValueError("variable names must be unique")
        self._keys = ordered
        self._indices = {key: index for index, key in enumerate(ordered)}
        self._names = {key.name: index for index, key in enumerate(ordered)}

    @classmethod
    def for_instance(cls, instance: UnitCommitmentInstance) -> "VariableRegistry":
        keys = [
            VariableKey(kind, unit.name, period)
            for kind in ("u", "z")
            for unit in instance.units
            for period in instance.periods
        ]
        return cls(keys)

    @classmethod
    def for_commitment(
        cls,
        unit_names: Sequence[str],
        periods: int | Iterable[int],
    ) -> "VariableRegistry":
        """Create a registry containing only commitment variables.

        The compact v0.1 formulation has both ``u`` and dispatch ``z`` bits.
        A network-scale case is deliberately reduced to commitment bits so
        that the dispatch can be solved by a continuous economic-dispatch
        subproblem after the QAOA step.  Keeping the same registry and bit
        ordering makes the new formulation compatible with the repository's
        existing QAOA implementation.
        """

        if isinstance(periods, int):
            if periods < 1:
                raise ValueError("periods must be positive")
            period_values = range(periods)
        else:
            period_values = tuple(int(period) for period in periods)
            if not period_values:
                raise ValueError("periods must not be empty")
        names = tuple(str(name) for name in unit_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("unit_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise ValueError("unit_names must be unique")
        return cls(
            [VariableKey("u", name, period) for name in names for period in period_values]
        )

    @property
    def keys(self) -> tuple[VariableKey, ...]:
        return self._keys

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(key.name for key in self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def index(self, key: VariableKey | str) -> int:
        if isinstance(key, str):
            try:
                return self._names[key]
            except KeyError as exc:
                raise KeyError(f"unknown variable {key!r}") from exc
        try:
            return self._indices[key]
        except KeyError as exc:
            raise KeyError(f"unknown variable {key!r}") from exc

    def key(self, index: int) -> VariableKey:
        try:
            return self._keys[int(index)]
        except (IndexError, ValueError) as exc:
            raise IndexError(f"unknown variable index {index}") from exc

    def bits_from_bitstring(self, bitstring: str) -> tuple[int, ...]:
        """Decode QAOA's convention: variable 0 is the right-most bit."""

        if len(bitstring) != len(self):
            raise ValueError(f"bitstring must have length {len(self)}")
        if any(character not in "01" for character in bitstring):
            raise ValueError("bitstring must contain only 0 and 1")
        return tuple(int(character) for character in bitstring[::-1])

    def bitstring_from_bits(self, bits: Iterable[int]) -> str:
        values = tuple(int(value) for value in bits)
        if len(values) != len(self) or any(value not in (0, 1) for value in values):
            raise ValueError(f"bits must be a binary sequence of length {len(self)}")
        return "".join(str(value) for value in values[::-1])


__all__ = ["VariableKey", "VariableRegistry"]

