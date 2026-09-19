"""OriginQ QCloud execution of locally optimized QAOA parameters.

Circuit preparation is entirely offline, including without pyqpanda3 installed.
Only ``run_hardware`` reads credentials or accesses the cloud. Results retain
unamended integer shot counts; probabilities are never resampled into counts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import hashlib
import math
from numbers import Integral, Real
import os
import re
import time
from typing import Any

import numpy as np

from .model import IrrigationQUBO
from .solvers import MAX_STATEVECTOR_QUBITS, _angles, _check_size, _positive_integer


TOKEN_ENV_VAR = "QCLOUD_API_KEY"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class HardwareError(RuntimeError):
    """A hardware operation failed; SDK exception text is deliberately omitted."""


def prepare_qaoa_circuit(
    model: IrrigationQUBO,
    gammas: np.ndarray,
    betas: np.ndarray,
    *,
    max_qubits: int = MAX_STATEVECTOR_QUBITS,
) -> dict:
    """Export measured OriginIR with the exact conventions of ``solve_qaoa``.

    The nonconstant Ising terms are divided by their largest absolute coefficient.
    ``RZ(2 gamma h)`` and CNOT/RZ/CNOT implement ``exp(-i gamma H)``;
    the mixer is ``RX(+2 beta)``. Measurements map q[i] directly to c[i].
    ``depth`` is the logical gate depth before the cloud maps to chip topology.
    """
    n = _check_size(model, max_qubits)
    gammas, betas = _angles(gammas, betas)
    h, j, offset = model.to_ising()
    h, j, offset = np.asarray(h, dtype=float), np.asarray(j, dtype=float), float(offset)
    if h.shape != (n,) or j.shape != (n, n):
        raise ValueError("to_ising must return h[n], upper-triangular J[n,n], and offset")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(j)) or not np.isfinite(offset):
        raise ValueError("Ising coefficients must be finite")
    if np.any(np.tril(j) != 0):
        raise ValueError("Ising J must be strictly upper triangular")
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(j)))) or 1.0
    lines = [f"QINIT {n}", f"CREG {n}"]
    counts: Counter[str] = Counter()
    wire_depth = [0] * n

    def gate(name: str, qubits: tuple[int, ...], angle: float | None = None) -> None:
        instruction = name + " " + ",".join(f"q[{q}]" for q in qubits)
        if angle is not None:
            if not math.isfinite(angle):
                raise ValueError("QAOA gate angles must be finite")
            # Fixed-point avoids parser-dependent scientific-notation handling.
            instruction += f",({angle:.17f})"
        if name == "MEASURE":
            instruction += f",c[{qubits[0]}]"
        lines.append(instruction)
        counts[name] += 1
        depth = max(wire_depth[q] for q in qubits) + 1
        for q in qubits:
            wire_depth[q] = depth

    for q in range(n):
        gate("H", (q,))
    for gamma, beta in zip(gammas, betas):
        for q in range(n):
            if h[q] != 0:
                gate("RZ", (q,), float(2 * gamma * (h[q] / scale)))
        for q in range(n):
            for other in range(q + 1, n):
                if j[q, other] != 0:
                    gate("CNOT", (q, other))
                    gate("RZ", (other,), float(2 * gamma * (j[q, other] / scale)))
                    gate("CNOT", (q, other))
        for q in range(n):
            gate("RX", (q,), float(2 * beta))
    for q in range(n):
        gate("MEASURE", (q,))
    originir = "\n".join(lines) + "\n"
    return {
        "originir": originir,
        "n_qubits": n,
        "normalization": scale,
        "depth": max(wire_depth),
        "gate_counts": dict(sorted(counts.items())),
        "sha256": hashlib.sha256(originir.encode("utf-8")).hexdigest(),
    }


def decode_counts(raw_counts: object, n_qubits: int, shots: int) -> dict[int, int]:
    """Strictly decode |c[n-1] ... c[0]> keys, preserving observed frequencies."""
    n_qubits = _positive_integer(n_qubits, "n_qubits")
    shots = _positive_integer(shots, "shots")
    if not isinstance(raw_counts, Mapping) or not raw_counts:
        raise HardwareError("QCloud returned no integer shot counts")
    counts = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or len(key) != n_qubits or any(c not in "01" for c in key):
            raise HardwareError("QCloud returned an invalid binary count key")
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise HardwareError("QCloud returned noninteger or negative shot counts")
        if value:
            counts[int(key, 2)] = int(value)
    if sum(counts.values()) != shots:
        raise HardwareError("QCloud count total differs from requested shots; no counts were synthesized")
    return dict(sorted(counts.items()))


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a nonempty identifier using letters, digits or ._:-")
    return value


def _chip_name(value: object) -> str:
    name = _identifier(value, "chip")
    if any(word in name.lower() for word in ("amplitude", "simulator", "qvm")):
        raise ValueError("chip must name a real QPU, not a simulator")
    return name


def _poll_settings(timeout: float, poll_interval: float) -> tuple[float, float]:
    for name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if timeout < 0 or poll_interval <= 0:
        raise ValueError("timeout must be nonnegative and poll_interval must be positive")
    return float(timeout), float(poll_interval)


def _load_sdk() -> Any:
    try:
        from pyqpanda3 import qcloud
    except (ImportError, OSError):
        raise HardwareError("QCloud requires pyqpanda3; install this example's '.[qcloud]' extra") from None
    return qcloud


def _parse_originir(originir: str) -> Any:
    try:
        from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog
        return convert_originir_string_to_qprog(originir)
    except (ImportError, OSError):
        raise HardwareError("OriginIR import requires pyqpanda3; install '.[qcloud]'") from None
    except Exception:
        raise HardwareError("pyqpanda3 rejected the exported OriginIR circuit") from None


def _service(sdk: Any, token_env: str) -> Any:
    if not isinstance(token_env, str) or _ENV_NAME.fullmatch(token_env) is None:
        raise ValueError("token_env must be an environment variable name")
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise HardwareError(f"Set {token_env} to your OriginQ API key before cloud access")
    try:
        service = sdk.QCloudService(api_key=token)
        # The SDK can log request metadata. Discard its logs instead of
        # exposing credentials or persisting them in the experiment directory.
        service.setup_logging(sdk.LogOutput.FILE, os.devnull)
        return service
    except Exception:
        raise HardwareError("QCloud authentication or service initialization failed") from None


def list_hardware_backends(*, token_env: str = TOKEN_ENV_VAR) -> dict[str, bool]:
    """Query current backend names; performs authenticated cloud access."""
    sdk = _load_sdk()
    service = _service(sdk, token_env)
    try:
        available = service.backends()
    except Exception:
        raise HardwareError("QCloud backend discovery failed") from None
    if not isinstance(available, Mapping):
        raise HardwareError("QCloud returned invalid backend information")
    result = {}
    for name, ready in available.items():
        try:
            name = _chip_name(name)
        except ValueError:
            continue
        if type(ready) is not bool:
            raise HardwareError("QCloud returned invalid backend availability")
        result[name] = ready
    return result


def run_hardware(
    circuit: Mapping,
    *,
    chip: str,
    shots: int = 1024,
    token_env: str = TOKEN_ENV_VAR,
    timeout: float = 300,
    poll_interval: float = 5,
    job_id: str | None = None,
    on_submitted: Callable[[str], None] | None = None,
) -> dict:
    """Explicitly submit one circuit, or poll an existing job without resubmitting.

    New jobs require ``on_submitted(job_id)`` to persist the job identifier before
    polling begins. ``timeout=0`` performs one query. Timeout returns ``pending``;
    the job continues in QCloud and can be resumed using its persisted identifier.
    The timeout bounds our polling loop; individual SDK network requests retain
    the SDK's own transport timeout. ``query()`` never waits for job completion.
    """
    chip = _chip_name(chip)
    shots = _positive_integer(shots, "shots")
    timeout, poll_interval = _poll_settings(timeout, poll_interval)
    if not isinstance(circuit, Mapping):
        raise ValueError("circuit must be the prepared circuit metadata")
    n = _positive_integer(circuit.get("n_qubits"), "n_qubits")
    originir = circuit.get("originir")
    if not isinstance(originir, str) or not originir.startswith(f"QINIT {n}\nCREG {n}\n"):
        raise ValueError("circuit OriginIR register sizes do not match its metadata")
    if circuit.get("sha256") != hashlib.sha256(originir.encode("utf-8")).hexdigest():
        raise ValueError("circuit OriginIR checksum does not match its metadata")
    if job_id is not None:
        job_id = _identifier(job_id, "job_id")
    elif not callable(on_submitted):
        raise ValueError("New submissions require an on_submitted callback to persist the job id")
    sdk = _load_sdk()
    prog = _parse_originir(originir) if job_id is None else None
    # Construction initializes SDK authentication also for QCloudJob(id).
    service = _service(sdk, token_env)
    if job_id is None:
        try:
            available = service.backends()
            if not isinstance(available, Mapping) or available.get(chip) is not True:
                raise HardwareError("Selected QCloud chip is unavailable or not listed for this account")
            backend = service.backend(chip)
            # chip_info is only supported for physical QPU backends. Cloud
            # mapping chooses physical qubits; no logical index is a chip id.
            usable = backend.chip_info().available_qubits()
            if len(set(usable)) < n:
                raise HardwareError("Selected QCloud chip has too few available qubits")
            options = sdk.QCloudOptions()
            options.set_mapping(True)
            options.set_optimization(True)
            options.set_amend(False)
            options.set_is_prob_counts(True)
            job = backend.run(prog, shots, options)
            job_id = _identifier(job.job_id(), "job_id")
        except HardwareError:
            raise
        except Exception:
            raise HardwareError("QCloud chip validation or submission failed; check the cloud task list before retrying") from None
        try:
            on_submitted(job_id)
        except Exception:
            raise HardwareError(f"Cloud job {job_id} was submitted but could not be saved; retain this id and do not resubmit") from None
    else:
        try:
            job = sdk.QCloudJob(job_id)
        except Exception:
            raise HardwareError("QCloud could not restore the saved job") from None

    response = {
        "job_id": job_id,
        "chip": chip,
        "shots": shots,
        "status": "pending",
        "counts": {},
        "raw_counts": {},
        "amend": False,
        "request_counts": True,
    }
    deadline = time.monotonic() + timeout
    while True:
        try:
            result = job.query()
            state = result.job_status()
            status = state.name
        except Exception:
            raise HardwareError(f"QCloud query failed for saved job {job_id}; resume this job to retry") from None
        if status not in {"WAITING", "QUEUING", "COMPUTING", "FINISHED", "FAILED"}:
            raise HardwareError("QCloud returned an unsupported job status")
        response["cloud_status"] = status
        if status == "FAILED":
            response["status"] = "failed"
            return response
        if status == "FINISHED":
            try:
                raw_counts = result.get_counts(sdk.DataBase.Binary)
            except Exception:
                raise HardwareError("QCloud finished without accessible integer counts; inspect the saved job in the cloud console") from None
            response["counts"] = decode_counts(raw_counts, n, shots)
            response["raw_counts"] = {key: int(value) for key, value in sorted(raw_counts.items())}
            response["status"] = "finished"
            return response
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return response
        time.sleep(min(poll_interval, remaining))
