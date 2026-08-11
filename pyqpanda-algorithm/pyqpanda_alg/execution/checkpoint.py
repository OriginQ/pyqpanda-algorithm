"""JSON checkpoint serialization for resumable algorithm tasks.

A checkpoint is the only durable artifact of an
:class:`~pyqpanda_alg.execution.algorithm_task.AlgorithmTask`: it holds
the format version, algorithm name, serializable state, backend task
IDs, backend identity, and user metadata.  Two security rules are
enforced here so checkpoints never leak credentials:

* credential-named keys (``api_key``, ``token``, ``password``, ...)
  are filtered recursively out of every dict, and
* values that are not JSON primitives (live ``RuntimeService`` or
  ``QDevice`` handles, circuits, ...) are rejected before anything is
  written, so no live object can ever end up in a checkpoint.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .backend_task import TaskStatus
from .errors import AlgorithmInputError, TaskRecoveryError

CHECKPOINT_FORMAT_VERSION = 1

#: Case-insensitive key fragments that mark a dict entry as a credential.
_CREDENTIAL_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "password",
    "passwd",
    "token",
    "secret",
    "credential",
    "authorization",
    "authorisation",
)


@dataclass(frozen=True)
class AlgorithmCheckpoint:
    """Decoded contents of one algorithm-task checkpoint file."""

    format_version: int
    algorithm: str
    status: str
    state: Any
    result: Any
    backend_task_ids: tuple[str, ...]
    backend_identity: Any
    metadata: dict[str, Any]


def redact_credentials(value: Any) -> Any:
    """Return ``value`` with credential-named keys removed recursively.

    Dicts, lists, and tuples are copied; leaves are returned unchanged,
    so the caller's object is never mutated.  A key is treated as a
    credential when it contains any of the case-insensitive fragments
    in ``_CREDENTIAL_KEY_PARTS`` (for example ``api_key`` or
    ``access_token``).
    """
    if isinstance(value, dict):
        return {
            key: redact_credentials(item)
            for key, item in value.items()
            if not _is_credential_key(key)
        }
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_credentials(item) for item in value)
    return value


def write_checkpoint(path: Union[str, Path], checkpoint: AlgorithmCheckpoint) -> Path:
    """Write ``checkpoint`` to ``path`` as redacted, JSON-serializable text.

    The write is atomic: the payload goes to a temporary sibling file
    first and is moved into place with :func:`os.replace`, so a reader
    never observes a half-written checkpoint.
    """
    payload = {
        "format_version": checkpoint.format_version,
        "algorithm": checkpoint.algorithm,
        "status": checkpoint.status,
        "state": _as_json_primitives(redact_credentials(checkpoint.state)),
        "result": _as_json_primitives(redact_credentials(checkpoint.result)),
        "backend_task_ids": list(checkpoint.backend_task_ids),
        "backend": _as_json_primitives(redact_credentials(checkpoint.backend_identity)),
        "metadata": _as_json_primitives(redact_credentials(checkpoint.metadata)),
    }
    target = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True)
    temp = target.with_name(f"{target.name}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target


def read_checkpoint(path: Union[str, Path]) -> AlgorithmCheckpoint:
    """Read and validate a checkpoint file.

    Raises :class:`~pyqpanda_alg.execution.errors.TaskRecoveryError`
    for unreadable, malformed, or version-incompatible files.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskRecoveryError(f"could not read checkpoint at {path}") from exc
    if not isinstance(payload, dict):
        raise TaskRecoveryError(
            f"malformed checkpoint at {path}: expected a JSON object"
        )
    try:
        version = payload["format_version"]
        algorithm = payload["algorithm"]
        status = payload["status"]
        state = payload["state"]
        result = payload["result"]
        backend_task_ids = payload["backend_task_ids"]
        backend_identity = payload["backend"]
        metadata = payload["metadata"]
    except KeyError as exc:
        raise TaskRecoveryError(
            f"malformed checkpoint at {path}: missing field {exc.args[0]!r}"
        ) from exc
    if version != CHECKPOINT_FORMAT_VERSION:
        raise TaskRecoveryError(
            f"unsupported checkpoint format version {version!r}; "
            f"this build supports version {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(algorithm, str) or not algorithm:
        raise TaskRecoveryError("checkpoint algorithm name must be a non-empty string")
    if not _is_task_status(status):
        raise TaskRecoveryError(f"invalid checkpoint status {status!r}")
    if not isinstance(backend_task_ids, list) or not all(
        isinstance(task_id, str) for task_id in backend_task_ids
    ):
        raise TaskRecoveryError("checkpoint backend_task_ids must be a list of strings")
    if not isinstance(metadata, dict):
        raise TaskRecoveryError("checkpoint metadata must be an object")
    return AlgorithmCheckpoint(
        format_version=version,
        algorithm=algorithm,
        status=status,
        state=state,
        result=result,
        backend_task_ids=tuple(backend_task_ids),
        backend_identity=backend_identity,
        metadata=metadata,
    )


def _is_credential_key(key: Any) -> bool:
    lowered = key.lower() if isinstance(key, str) else str(key)
    lowered = lowered.replace("-", "_")
    return any(part in lowered for part in _CREDENTIAL_KEY_PARTS)


def _as_json_primitives(value: Any) -> Any:
    """Return ``value`` converted to JSON primitives.

    Tuples become lists.  Anything else that JSON cannot represent
    (live objects such as RuntimeService or QDevice handles) raises
    :class:`~pyqpanda_alg.execution.errors.AlgorithmInputError` instead
    of being silently dropped or half-serialized.  Dict keys are
    validated too: JSON can only stringify primitive keys, so a tuple or
    live-object key raises instead of leaking a raw ``TypeError`` out of
    :func:`json.dumps`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise AlgorithmInputError(
                    "checkpoint dict keys must be JSON-serializable, "
                    f"got {type(key).__name__}"
                )
            converted[key] = _as_json_primitives(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_as_json_primitives(item) for item in value]
    raise AlgorithmInputError(
        "checkpoint values must be JSON-serializable, "
        f"got {type(value).__name__}"
    )


def _is_task_status(status: Any) -> bool:
    try:
        TaskStatus(status)
    except ValueError:
        return False
    return True
