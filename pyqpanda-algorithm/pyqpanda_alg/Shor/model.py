"""Shor data types: configuration, preprocessing outcome, and run record.

:class:`ShorConfig` carries the classical knobs of a factorization
run: the bounded number of quantum order-finding attempts and the
phase-register size.  :class:`PreprocessOutcome` is the discriminated
result of :func:`pyqpanda_alg.Shor.classical_preprocess`: either an
explicit classical factorization (even, prime, or perfect-power
modulus) or a marker that quantum order finding is required.
:class:`ShorResult` is the immutable snapshot of a completed run; it
always states whether a quantum task was actually used and records the
task IDs when one was.
"""

import copy
from dataclasses import dataclass
from typing import Any

RESOLVED = "resolved"
NEEDS_QUANTUM = "needs_quantum"


@dataclass(frozen=True)
class ShorConfig:
    """Immutable Shor run configuration.

    ``max_attempts`` bounds the number of quantum order-finding
    attempts before the run gives up.  ``phase_qubits`` is the size of
    the phase estimation register; None selects the derived default of
    twice the modulus bit length.
    """

    max_attempts: int = 6
    phase_qubits: int | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be at least 1, got {self.max_attempts}"
            )
        if self.phase_qubits is not None and self.phase_qubits < 1:
            raise ValueError(
                f"phase_qubits must be at least 1, got {self.phase_qubits}"
            )


@dataclass(frozen=True)
class PreprocessOutcome:
    """Discriminated result of :func:`classical_preprocess`.

    ``status`` is either ``RESOLVED`` — the modulus was factored
    classically and ``factors`` (or ``is_prime``) carries the answer —
    or ``NEEDS_QUANTUM``, meaning quantum order finding is required.
    """

    status: str
    factors: tuple[int, int] | None = None
    is_prime: bool = False

    @property
    def resolved(self) -> bool:
        """True when classical preprocessing produced the answer."""
        return self.status == RESOLVED

    def __post_init__(self) -> None:
        if self.status not in (RESOLVED, NEEDS_QUANTUM):
            raise ValueError(f"unknown preprocessing status {self.status!r}")
        if self.status == RESOLVED and self.factors is None and not self.is_prime:
            raise ValueError("a resolved outcome must carry factors or is_prime")
        if self.factors is not None:
            factors = tuple(int(factor) for factor in self.factors)
            if len(factors) != 2:
                raise ValueError(f"factors must be a pair, got {self.factors!r}")
            object.__setattr__(self, "factors", factors)


@dataclass(frozen=True)
class ShorResult:
    """Immutable outcome of a Shor run.

    ``factors`` is the nontrivial factor pair when one was found and
    None otherwise; ``is_prime`` reports a classically proven prime
    modulus.  ``used_quantum`` is True when the result emerged from
    the quantum attempt pipeline — order-finding success or exhaustion
    — regardless of later classical resolution, and ``task_ids``
    records the submitted task IDs when one was (empty otherwise).
    ``order`` is the recovered multiplicative order when order finding
    ran, and ``metadata`` keeps free-form provenance without exposing
    any live object.
    """

    factors: tuple[int, int] | None = None
    is_prime: bool = False
    used_quantum: bool = False
    task_ids: tuple[str, ...] = ()
    order: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.is_prime and self.factors is not None:
            raise ValueError("factors must be None for a prime modulus")
        if self.factors is not None:
            factors = tuple(int(factor) for factor in self.factors)
            if len(factors) != 2:
                raise ValueError(f"factors must be a pair, got {self.factors!r}")
            object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "is_prime", bool(self.is_prime))
        object.__setattr__(self, "used_quantum", bool(self.used_quantum))
        object.__setattr__(self, "task_ids", tuple(str(task) for task in self.task_ids))
        if self.order is not None:
            object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
