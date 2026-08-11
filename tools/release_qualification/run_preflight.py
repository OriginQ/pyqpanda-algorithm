"""FakeBackend preflight runner: credential-free qualification evidence.

The preflight runner is the credential-free half of release
qualification.  It queries an explicitly selected device through a
``qpanda3-runtime`` ``RuntimeService`` (already logged in by the
caller), constructs the device's ``FakeBackend`` -- the noise-injected
simulator the device itself exposes -- and runs every fixed case from
:mod:`tools.release_qualification.cases` against it.  Each submitted
circuit is transpiled through the fake backend and each raw result is
recorded as a SHA-256 digest only: raw responses themselves are never
written, and everything that is written passes through the manifest
sanitizer, so no credential can reach an artifact.

:func:`run_preflight` takes the service as an argument so the release
tests drive it with the credential-free service stand-in.  The
``--chip-id``/``--wheel``/``--output-dir`` CLI in :func:`main` obtains
the API key exclusively from the ``QPANDA3_API_KEY`` environment
variable, logs in, and delegates to :func:`run_preflight` under a real
``if __name__ == "__main__":`` entry (the real fake backend's
transpile path spawns processes, so the guard is required).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pyqpanda3.core import QCircuit, QProg

from pyqpanda_alg.execution import (
    BackendCapabilities,
    CompletedBackendTask,
    DeviceCapabilityError,
    EstimateBatchResult,
    ExecutionOptions,
    SampleBatchResult,
)
from pyqpanda_alg.Shor.circuit import build_order_finding_circuit

from .cases import QUALIFICATION_CASES, TRANSPILATION_CASES
from .manifest import (
    SCHEMA_VERSION,
    AlgorithmQualification,
    QualificationManifest,
    validate_manifest,
)

#: Default wheel name when the CLI does not name one yet.
_DEFAULT_WHEEL = "pyqpanda_alg-2.1.0-py3-none-any.whl"
#: Placeholder digest used when the wheel file is not available yet.
_UNKNOWN_DIGEST = "0" * 64

#: The committed Shor case's modulus and RNG seed (see ``cases.py``).
_SHOR_MODULUS = 15
_SHOR_SEED = 42


class _PreflightBackend:
    """``ExecutionBackend`` adapter that drives the device's FakeBackend.

    Submissions are executed synchronously by the fake backend -- the
    credential-free simulator the device exposes -- and every submitted
    circuit and raw result is recorded, so the runner can transpile the
    case's circuits and digest its raw results afterwards.  The fake
    backend surface mirrors a device that samples and estimates but has
    no variational-session or statevector path; those capabilities are
    advertised as unavailable and requesting them raises
    :class:`~pyqpanda_alg.execution.DeviceCapabilityError`, which is
    the honest signal algorithms must respect.
    """

    capabilities = BackendCapabilities(
        sampling=True,
        estimation=True,
        variational_session=False,
        statevector=False,
        tomography=False,
    )

    def __init__(self, fake: Any) -> None:
        self._fake = fake
        #: ``(circuit, observable_or_None)`` pairs in submission order.
        self.submissions: list = []
        #: Raw fake-backend responses (counts dicts and estimate floats).
        self.raw_results: list = []

    def submit_sample(self, circuit: Any, *, options: ExecutionOptions):
        counts = self._fake.sample(circuit, shots=options.shots)
        self.submissions.append((circuit, None))
        self.raw_results.append(counts)
        return CompletedBackendTask(
            SampleBatchResult(counts=(counts,), shots=options.shots),
            task_id="preflight-sample",
        )

    def submit_estimate(
        self, circuit_and_observable: Any, *, options: ExecutionOptions
    ):
        circuit, observable = circuit_and_observable
        value = self._fake.estimate(circuit, observable, shots=options.shots)
        self.submissions.append((circuit, observable))
        self.raw_results.append(value)
        return CompletedBackendTask(
            EstimateBatchResult(values=(value,)), task_id="preflight-estimate"
        )

    def submit_statevector(self, circuit: Any, *, options: ExecutionOptions):
        raise DeviceCapabilityError(
            "the fake backend has no statevector path; "
            "statevector evidence is recorded by the QPU runner"
        )

    def create_variational_session(
        self, ansatz: Any, observable: Any, *, options: ExecutionOptions
    ):
        raise DeviceCapabilityError(
            "the fake backend has no variational-session path; "
            "variational evidence is recorded by the QPU runner"
        )


def run_preflight(
    service: Any,
    chip_id: str,
    output_dir: Any = None,
    *,
    wheel: str | None = None,
    wheel_sha256: str | None = None,
) -> QualificationManifest:
    """Run every fixed case against ``chip_id``'s FakeBackend.

    Queries the device through ``service``, constructs its fake
    backend, qualifies every ``QUALIFICATION_CASES`` and
    ``TRANSPILATION_CASES`` entry, and returns the sanitized manifest.
    When ``output_dir`` is given, the sanitized manifest JSON is
    written to ``output_dir/manifest.json``.  ``wheel``/``wheel_sha256``
    name the candidate artifact; the digest is the real file digest
    when the CLI computed one, the placeholder otherwise.
    """
    device = service.device(chip_id)
    fake = device.fake_backend()
    cases = tuple(
        _qualify(case, fake)
        for case in (*QUALIFICATION_CASES, *TRANSPILATION_CASES)
    )
    manifest = QualificationManifest(
        version=SCHEMA_VERSION,
        timestamp=_utc_now(),
        git_commit=_git_commit(),
        wheel=wheel or _DEFAULT_WHEEL,
        wheel_sha256=wheel_sha256 or _UNKNOWN_DIGEST,
        python_version=platform.python_version(),
        pyqpanda3_version=_package_version("pyqpanda3"),
        qpanda3_runtime_version=_package_version("qpanda3-runtime"),
        device=_device_record(device, chip_id),
        cases=cases,
    )
    payload = manifest.to_dict()
    validate_manifest(payload)  # fail loudly rather than write a broken record
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _qualify(case: Any, fake: Any) -> AlgorithmQualification:
    """Run one fixed case against the fake backend and record evidence.

    Callable builders run their algorithm against the recording
    adapter; circuit-shaped builders are transpiled directly.  Every
    captured circuit is transpiled through the fake backend and the raw
    results are recorded as a SHA-256 digest; the verdict is ``passed``
    exactly when the run completed and every captured circuit
    transpiled.
    """
    request = case.builder()
    error = None
    if isinstance(request, (QCircuit, QProg)):
        circuits, raw_results, submitted = [request], [], 0
    elif isinstance(request, tuple) and len(request) == 2:
        circuits, raw_results, submitted = [request[0]], [], 0
    else:
        backend = _PreflightBackend(fake)
        try:
            request(backend)
        except Exception as exc:  # record the failure; transpile anyway
            error = exc
        submitted = len(backend.submissions)
        circuits = [circuit for circuit, _ in backend.submissions]
        raw_results = backend.raw_results
        if error is None and not circuits and case.algorithm == "Shor":
            # The committed Shor draw (seed 42, modulus 15) resolves
            # classically -- gcd(12, 15) = 3 -- so the solver submits
            # nothing; transpile the order-finding circuit the solver
            # would build on its first coprime draw instead.
            circuits = [_shor_order_finding_program()]

    transpiled_list, failed_list = fake.transpile(circuits) if circuits else ([], [])
    transpiled = bool(transpiled_list) and not failed_list
    passed = error is None and transpiled
    return AlgorithmQualification(
        algorithm=case.algorithm,
        execution_mode="preflight" if case.mode != "transpile" else "transpile",
        task_ids=(),
        shots=case.shots,
        threshold=case.threshold,
        raw_result_digest=_digest(raw_results),
        parsed_result={
            "submissions": submitted,
            "transpiled_circuits": len(transpiled_list),
            "outcome": "ok" if error is None else f"{type(error).__name__}: {error}",
        },
        verdict="passed" if passed else "failed",
        transpiled=transpiled,
    )


def _shor_order_finding_program() -> QProg:
    """The committed Shor case's first quantum-order-finding circuit.

    Mirrors ``cases._shor_invoke``: bases are drawn from
    ``random.Random(42)`` with ``randrange(2, modulus - 1)`` exactly as
    the solver draws them, and the first base coprime to 15 selects the
    quantum path (the solver redraws per attempt).  The build is a
    17-qubit circuit (8 phase + 2*4 value/ancilla qubits) for modulus
    15.  ``build_order_finding_circuit`` lives one module below the
    package's sanctioned surface; the runner drives it deliberately so
    the qualification record can attest the case's circuit transpiles.
    """
    rng = random.Random(_SHOR_SEED)
    base = rng.randrange(2, _SHOR_MODULUS - 1)
    while math.gcd(base, _SHOR_MODULUS) != 1:
        base = rng.randrange(2, _SHOR_MODULUS - 1)
    return build_order_finding_circuit(base, _SHOR_MODULUS).program


def _device_record(device: Any, chip_id: str) -> dict:
    """Build the schema-conforming device record from the device surface.

    ``summary`` is the device's human-readable name when the surface
    provides one; ``calibration_timestamp`` is the configuration
    snapshot time (UTC), because the runtime exposes no calibration
    timestamp accessor.
    """
    try:
        chip = str(device.chip_id())
    except Exception:
        chip = chip_id
    name = getattr(device, "name", None)
    summary = str(name()) if callable(name) else f"device {chip}"
    return {
        "chip_id": chip,
        "summary": summary,
        "calibration_timestamp": _utc_now(),
    }


def _digest(raw_results: list) -> str:
    """SHA-256 hex digest of the raw results, never the results itself."""
    payload = json.dumps(raw_results, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "0" * 40  # placeholder; the QPU workflow pins the real commit


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Credentialed CLI entry: login and run the preflight qualification."""
    parser = argparse.ArgumentParser(
        description="Run the fixed qualification cases against a device's "
        "FakeBackend and write the sanitized preflight manifest."
    )
    parser.add_argument(
        "--chip-id", required=True, help="explicitly selected device, e.g. WK_C180"
    )
    parser.add_argument(
        "--wheel", default=None, help="candidate wheel filename (optional)"
    )
    parser.add_argument(
        "--output-dir", default=".", help="directory for the manifest (default: .)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("QPANDA3_API_KEY")
    if not api_key:
        sys.exit(
            "QPANDA3_API_KEY is required; refusing to run preflight without it"
        )

    wheel_sha256 = None
    if args.wheel:
        wheel_path = Path(args.wheel)
        if wheel_path.is_file():
            wheel_sha256 = _sha256_file(wheel_path)

    from qpanda3_runtime import RuntimeService  # kept out of the import path

    service = RuntimeService()
    service.login(api_key)
    manifest = run_preflight(
        service,
        args.chip_id,
        output_dir=args.output_dir,
        wheel=args.wheel,
        wheel_sha256=wheel_sha256,
    )
    passed = sum(1 for case in manifest.cases if case.verdict == "passed")
    failed = [case.algorithm for case in manifest.cases if case.verdict != "passed"]
    print(
        f"preflight {args.chip_id}: {passed}/{len(manifest.cases)} cases passed"
        f"{', failed: ' + ', '.join(failed) if failed else ''}"
    )
    print(f"manifest written to {Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
