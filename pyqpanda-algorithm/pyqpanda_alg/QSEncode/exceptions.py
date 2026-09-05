"""Public exception contracts for QSEncode-Insight.

Phase 1 defines stable names and structured error codes only.  Scientific
pipeline code is intentionally absent.
"""

from __future__ import annotations


class QSEncodeInsightError(Exception):
    """Base class for structured QSEncode-Insight product errors."""

    def __init__(self, *, code: str, message: str | None = None) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("error code must be a non-empty string")
        self.code = code
        self.message = message
        super().__init__(message if message is not None else code)

    def __str__(self) -> str:
        return self.message if self.message is not None else self.code


class InputValidationError(QSEncodeInsightError):
    """The new facade received an invalid probability input."""


class ConfigurationError(QSEncodeInsightError):
    """The new facade received an invalid or unsupported configuration."""


class BaselineConstructionError(QSEncodeInsightError):
    """The mandatory dense baseline could not be constructed."""


class ResourceAuditError(QSEncodeInsightError):
    """Compiled-resource auditing could not establish a valid result."""


class SerializationError(QSEncodeInsightError):
    """A structured result could not be serialized under the v1 schema."""


class ResultBindingError(QSEncodeInsightError):
    """A supplied result is not bound to the runtime input/configuration."""


class UncertifiedSelectionError(QSEncodeInsightError):
    """Preparation was requested from an uncertified audit selection."""


class InternalInvariantError(QSEncodeInsightError):
    """An internal product invariant was violated."""


__all__ = [
    "QSEncodeInsightError",
    "InputValidationError",
    "ConfigurationError",
    "BaselineConstructionError",
    "ResourceAuditError",
    "SerializationError",
    "ResultBindingError",
    "UncertifiedSelectionError",
    "InternalInvariantError",
]
