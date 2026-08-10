"""Execution options shared by every backend.

:class:`ExecutionOptions` carries the runtime-independent knobs of a
submission.  Field names follow the qpanda3-runtime API so that a
runtime backend can forward them verbatim.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PreflightMode(Enum):
    """How much of a run happens before real submission."""

    NONE = "none"
    TRANSPILE_ONLY = "transpile_only"
    FAKE_EXECUTE = "fake_execute"


@dataclass(frozen=True)
class ExecutionOptions:
    """Immutable submission options, safe to reuse across backends."""

    shots: int = 1000
    timeout: float = 1800.0
    preflight: PreflightMode = PreflightMode.TRANSPILE_ONLY
    specified_block: Optional[tuple[int, ...]] = None
    is_mapping: bool = True
    is_amend: bool = True
    is_optimization: bool = True

    def __post_init__(self) -> None:
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
