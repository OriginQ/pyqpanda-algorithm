"""Qualification manifest: closed schema validation and credential sanitization.

The release gate treats a manifest as the only durable evidence a
candidate release is qualified: it pins the exact commit, wheel digest,
device, and per-algorithm task/verdict records.  Two rules make the
artifact credential-safe:

* :func:`validate_manifest` enforces the closed ``schema.json``
  structure -- unknown keys are rejected at every level and every
  required field must be present and well-formed.
* :func:`sanitize_payload` recursively removes credential-shaped keys
  (``api_key``, ``token``, ``password``, ...) before anything is
  written, and :func:`contains_credentials` lets callers assert that a
  payload is clean.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

#: Schema version every manifest must declare.
SCHEMA_VERSION = "1"

#: Case-insensitive key fragments that mark a dict entry as a credential.
_CREDENTIAL_KEY_PARTS = (
    "api_key",
    "token",
    "password",
    "secret",
    "authorization",
    "bearer",
    "access_key",
)

_SCHEMA = json.loads(
    Path(__file__).with_name("schema.json").read_text(encoding="utf-8")
)


class ManifestValidationError(ValueError):
    """Raised when a manifest dict fails closed-schema validation."""


def validate_manifest(manifest_dict: dict) -> dict:
    """Validate ``manifest_dict`` against the closed manifest schema.

    Returns the validated dict unchanged on success; raises
    :class:`ManifestValidationError` describing the first violation.
    """
    try:
        jsonschema.validate(manifest_dict, _SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ManifestValidationError(
            f"invalid qualification manifest at {exc.json_path}: {exc.message}"
        ) from exc
    return manifest_dict


def sanitize_payload(payload: Any) -> Any:
    """Return ``payload`` with credential-shaped keys removed recursively.

    Dicts, lists, and tuples are copied; leaves are returned unchanged,
    so the caller's object is never mutated.  A key is treated as a
    credential when its lowercased form contains any fragment of
    ``_CREDENTIAL_KEY_PARTS`` (``api_key``, ``access_token``,
    ``bearer``, ...), so no API key or token can survive into an
    artifact.
    """
    if isinstance(payload, dict):
        return {
            key: sanitize_payload(item)
            for key, item in payload.items()
            if not _is_credential_key(key)
        }
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(sanitize_payload(item) for item in payload)
    return payload


def contains_credentials(payload: Any) -> bool:
    """Return True when any key of ``payload`` is credential-shaped.

    Recurses through nested dicts, lists, and tuples with the same
    case-insensitive substring matching as :func:`sanitize_payload`, so
    an ``api_key`` buried in a parsed result is still detected.
    """
    if isinstance(payload, dict):
        return any(
            _is_credential_key(key) or contains_credentials(item)
            for key, item in payload.items()
        )
    if isinstance(payload, (list, tuple)):
        return any(contains_credentials(item) for item in payload)
    return False


@dataclass(frozen=True)
class AlgorithmQualification:
    """Qualification record of one fixed algorithm case.

    ``raw_result_digest`` is the SHA-256 digest of the raw service
    response and ``parsed_result`` holds the digestable interpretation;
    ``verdict`` is the outcome against the committed threshold
    (``passed`` or ``failed``).  ``transpiled`` records the preflight
    device-transpilation evidence.
    """

    algorithm: str
    execution_mode: str
    task_ids: tuple[str, ...]
    shots: int
    threshold: float
    raw_result_digest: str
    parsed_result: dict
    verdict: str
    transpiled: bool = False

    def to_dict(self) -> dict:
        """Return the JSON-safe case record."""
        return {
            "algorithm": self.algorithm,
            "execution_mode": self.execution_mode,
            "task_ids": list(self.task_ids),
            "shots": self.shots,
            "threshold": self.threshold,
            "raw_result_digest": self.raw_result_digest,
            "parsed_result": self.parsed_result,
            "verdict": self.verdict,
            "transpiled": self.transpiled,
        }


@dataclass(frozen=True)
class QualificationManifest:
    """Immutable qualification record for one release candidate.

    Fields mirror the closed ``schema.json`` structure exactly, so
    :meth:`to_dict` output always conforms to the schema.
    """

    version: str
    timestamp: str
    git_commit: str
    wheel: str
    wheel_sha256: str
    python_version: str
    pyqpanda3_version: str
    qpanda3_runtime_version: str
    device: dict
    cases: tuple[AlgorithmQualification, ...]

    def to_dict(self) -> dict:
        """Return the credential-safe, schema-conforming manifest dict.

        The payload passes through :func:`sanitize_payload`, so no
        credential-shaped key can reach an artifact even if one slipped
        into a parsed result.
        """
        payload = {
            "version": self.version,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "wheel": self.wheel,
            "wheel_sha256": self.wheel_sha256,
            "python_version": self.python_version,
            "pyqpanda3_version": self.pyqpanda3_version,
            "qpanda3_runtime_version": self.qpanda3_runtime_version,
            "device": dict(self.device),
            "cases": [case.to_dict() for case in self.cases],
        }
        return sanitize_payload(payload)


def _is_credential_key(key: Any) -> bool:
    normalized = (key.lower() if isinstance(key, str) else str(key)).replace("-", "_")
    return any(part in normalized for part in _CREDENTIAL_KEY_PARTS)
