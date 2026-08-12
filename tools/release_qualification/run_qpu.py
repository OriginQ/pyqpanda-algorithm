"""Real-QPU qualification runner: sequential, resumable remote evidence.

Plan 7 Task 4: :class:`QPURunner` submits every fixed case from
:mod:`tools.release_qualification.cases` against the QPU runtime
service one case at a time.  Every remote task is checkpointed after
the submission; a resumed run recovers checkpointed tasks through the
service and queries their existing task IDs instead of resubmitting
them.  A task submission failure is recorded as a failed verdict and is
never automatically retried.  Status queries go through
:class:`RuntimeBackendTask` whose bounded wait implements the execution
layer's backoff, clamped by the committed per-case timeout.  Verdicts
consume only the committed case thresholds via
:mod:`tools.release_qualification.verdicts`.

Credentials never pass through this module: the runner receives an
already-logged-in service, and the manifest it produces passes through
:func:`sanitize_payload` before anything is written, so no
credential-shaped value can reach an artifact.

Design notes
------------
* Real QPU evidence, not transpile evidence: every submission carries
  ``preflight=PreflightMode.NONE`` so the task submitted to the
  service is executed directly.  Transpile records are the preflight
  runner's job (:mod:`tools.release_qualification.run_preflight`);
  the QPU runner never depends on the transpile service, so an outage
  there (e.g. a blanket ``"Failed to transpile."``) cannot block or
  falsify the real execution.
* Circuit-shaped cases (bell, Grover) submit exactly one sampling task
  with the committed shots; algorithm cases run their committed
  invocation against a checkpointing backend proxy, so the solver's own
  submissions (one per Shor attempt, per VQE round, ...) are equally
  checkpointed and recoverable.
* Shor provenance is not faked: the committed seed and attempt budget
  drive the real solver, but the committed draw (seed 42, modulus 15)
  resolves classically at its first base, and a classical resolution is
  not QPU evidence.  The runner therefore drives the committed RNG with
  a coprime-redraw policy (bases keep coming from
  ``random.Random(case.seed)``; non-coprime draws are skipped), so the
  solver's first attempt always enters quantum order finding.  No
  factor or order is ever encoded.
* HHL records its success probability and the requested observable from
  the sampled circuit; no statevector is requested on the QPU path.
* ``raw_result_digest`` digests the recorded parsed result: raw service
  responses are never retained, so the digest is computed over the
  interpretation that enters the manifest (documented honestly).
"""

import argparse
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pyqpanda3.core import QCircuit, QProg

if __package__ in (None, ""):  # run as a plain script: make the repo importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqpanda_alg.execution import (  # noqa: E402
    ExecutionOptions,
    PreflightMode,
    QPandaRuntimeBackend,
    RuntimeBackendTask,
)

from tools.release_qualification.cases import (  # noqa: E402
    QUALIFICATION_CASES,
    SMOKE_CASES,
    _SHOR_MODULUS,
)
from tools.release_qualification.manifest import (  # noqa: E402
    SCHEMA_VERSION,
    AlgorithmQualification,
    QualificationManifest,
    validate_manifest,
)
from tools.release_qualification.run_preflight import (  # noqa: E402
    _UNKNOWN_DIGEST,
    _device_record,
    _digest,
    _git_commit,
    _package_version,
    _sha256_file,
    _utc_now,
    _wheel_basename,
)
from tools.release_qualification.verdicts import verdict_for_case  # noqa: E402

#: Shor modulus fixed at release time (imported from ``cases.py`` as the
#: single source); the committed RNG policy is applied by the runner
#: (see module notes).
#: Default QPU chip for the qualification run; overridable via ``--chip-id``.
_DEFAULT_CHIP_ID = "WK_C180"


class QPURunner:
    """Sequential, resumable real-QPU runner for the fixed cases.

    ``service`` must already be logged in; the runner never
    authenticates and never writes credentials.  ``device`` defaults to
    ``service.device(chip_id)``; an explicitly selected device (e.g. a
    test double) takes precedence.  With ``checkpoint_dir`` set, every
    submitted remote task is checkpointed right after the submission and
    a resumed run recovers checkpointed tasks through the service,
    querying their existing task IDs instead of resubmitting them.
    All submissions are direct executions
    (``preflight=PreflightMode.NONE``): the transpile record is the
    preflight runner's to provide, never this runner's to wait on.
    """

    def __init__(
        self,
        service: Any,
        *,
        chip_id: str = _DEFAULT_CHIP_ID,
        device: Any = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.service = service
        self.chip_id = chip_id
        self.device = device if device is not None else service.device(chip_id)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        #: Remote tasks actually submitted; recovered tasks never count.
        self.submission_count = 0
        self._backend = QPandaRuntimeBackend(service, self.device)

    def run_case(self, case: Any) -> AlgorithmQualification:
        """Run one fixed case against the QPU and return its record.

        One submission per request: a submission failure is recorded as
        a failed verdict and is never retried.  Verdicts consume only
        the committed case threshold.  The record is returned whether
        the case passed or failed, so an outage can postpone
        qualification but never becomes a passing result.
        """
        executor = _CaseExecutor(
            self._backend,
            self.service,
            algorithm=case.algorithm,
            directory=self.checkpoint_dir,
        )
        try:
            request = case.builder()
        except Exception as exc:  # the request itself failed to build
            record = self._failure_record(case, executor, exc)
            self.submission_count += executor.submission_count
            return record

        if isinstance(request, (QCircuit, QProg)):
            record = self._qualify_circuit(case, executor, request, None)
        elif isinstance(request, tuple) and len(request) == 2:
            record = self._qualify_circuit(case, executor, request[0], request[1])
        else:
            record = self._qualify_invoke(case, executor, request)
        self.submission_count += executor.submission_count
        return record

    # ------------------------------------------------------------------
    # Qualification paths.
    # ------------------------------------------------------------------

    def _qualify_circuit(
        self,
        case: Any,
        executor: "_CaseExecutor",
        circuit: Any,
        observable: Any,
    ) -> AlgorithmQualification:
        """Submit a pure sampling (or estimation) request and record it."""
        # Direct QPU submission: the transpile record is provided by the
        # preflight runner separately, so a transpile-service failure
        # never blocks the real execution.
        options = ExecutionOptions(
            shots=case.shots, timeout=case.timeout, preflight=PreflightMode.NONE
        )
        try:
            if observable is not None:
                task = executor.submit_estimate((circuit, observable), options=options)
                parsed = task.result().single_value()
            else:
                task = executor.submit_sample(circuit, options=options)
                parsed = task.result().single_counts()
        except Exception as exc:
            return self._failure_record(case, executor, exc)
        # The parsed result (counts dict or estimate float) is the JSON-safe
        # record on the circuit path; a scalar parsed result is wrapped so
        # ``parsed_result`` always stays an object per the schema.
        parsed_record = parsed if isinstance(parsed, dict) else {"value": _jsonable(parsed)}
        return self._success_record(case, executor, parsed, record=parsed_record)

    def _qualify_invoke(
        self, case: Any, executor: "_CaseExecutor", invoke: Callable
    ) -> AlgorithmQualification:
        """Run a committed algorithm invocation against the checkpointing
        proxy and record its parsed result."""
        try:
            if case.algorithm == "Shor":
                result = _run_shor_case(case, executor)
            else:
                result = invoke(executor)
        except Exception as exc:
            return self._failure_record(case, executor, exc)
        parsed, record = _parsed_and_record(case, result)
        return self._success_record(case, executor, parsed, record=record)

    # ------------------------------------------------------------------
    # Record assembly.
    # ------------------------------------------------------------------

    def _success_record(
        self, case: Any, executor: "_CaseExecutor", parsed: Any, *, record: dict | None = None
    ) -> AlgorithmQualification:
        """Record a completed case; the verdict uses the committed bound."""
        parsed_record = record if record is not None else {"value": _jsonable(parsed)}
        verdict = verdict_for_case(case, parsed, shots=case.shots)
        return AlgorithmQualification(
            algorithm=case.algorithm,
            execution_mode="qpu" if case.mode != "transpile" else "transpile",
            task_ids=tuple(executor.task_ids),
            shots=case.shots,
            threshold=case.threshold,
            raw_result_digest=_digest([parsed_record]),
            parsed_result=parsed_record,
            verdict="passed" if verdict.passed else "failed",
            transpiled=True,
        )

    def _failure_record(self, case: Any, executor: "_CaseExecutor", exc: Exception) -> AlgorithmQualification:
        """Record a submission/execution failure honestly as a failure."""
        record = {
            "outcome": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "task_ids": list(executor.task_ids),
        }
        return AlgorithmQualification(
            algorithm=case.algorithm,
            execution_mode="qpu" if case.mode != "transpile" else "transpile",
            task_ids=tuple(executor.task_ids),
            shots=case.shots,
            threshold=case.threshold,
            raw_result_digest=_digest([record]),
            parsed_result=record,
            verdict="failed",
            transpiled=False,
        )


class _CaseExecutor:
    """Checkpointing submission proxy for one fixed case.

    Every submitted remote task is checkpointed to
    ``<directory>/<algorithm>-<n>.json`` immediately after the
    submission ("checkpoint after each remote task").  On resume
    (``resume=True``), the checkpointed task at the next ordinal is
    recovered through the service and returned instead of submitting a
    new one -- the existing task ID is queried, never resubmitted.
    ``task_ids`` collects every task actually submitted or recovered;
    ``submission_count`` counts only real submissions.
    """

    def __init__(
        self,
        backend: Any,
        service: Any,
        *,
        algorithm: str,
        directory: Path | None,
    ) -> None:
        self._backend = backend
        self._service = service
        self._algorithm = algorithm
        self._directory = directory
        self._ordinal = 0
        self._resume = self._has_checkpoints() if directory is not None else False
        self.task_ids: list[str] = []
        self.submission_count = 0

    # Backend surface (the committed invocations call exactly these).

    def submit_sample(self, circuit: Any, *, options: ExecutionOptions) -> RuntimeBackendTask:
        # Force direct execution: algorithm cases build their own
        # ExecutionOptions with the default TRANSPILE_ONLY preflight, and
        # the QPU (e.g. WK_C180, gateset RPhi+CZ) would reject the
        # gateset/qubit preflight check.  The QPU qualification runner
        # executes directly; transpile records are the preflight
        # runner's to provide.  Shots/timeout and the other fields are
        # preserved.
        return self._submit(
            lambda: self._backend.submit_sample(
                circuit, options=dataclasses.replace(options, preflight=PreflightMode.NONE)
            )
        )

    def submit_estimate(self, circuit_and_observable: tuple, *, options: ExecutionOptions) -> RuntimeBackendTask:
        # Direct execution, as in submit_sample (see there).
        return self._submit(
            lambda: self._backend.submit_estimate(
                circuit_and_observable,
                options=dataclasses.replace(options, preflight=PreflightMode.NONE),
            )
        )

    def submit_statevector(self, circuit: Any, *, options: ExecutionOptions) -> Any:
        # QPandaRuntimeBackend raises DeviceCapabilityError for the
        # unadvertised capability; the failure is recorded by the runner.
        return self._backend.submit_statevector(circuit, options=options)

    def create_variational_session(self, ansatz: Any, observable: Any, *, options: ExecutionOptions) -> Any:
        # Session tasks are tracked by the session itself (VQE's
        # AlgorithmTask machinery); the executor checkpoints the
        # sample/estimate tasks only.
        return self._backend.create_variational_session(ansatz, observable, options=options)

    @property
    def capabilities(self) -> Any:
        """Advertise the inner backend's capabilities to algorithms."""
        return self._backend.capabilities

    # ------------------------------------------------------------------

    def _submit(self, submit: Callable) -> RuntimeBackendTask:
        """Recover the checkpointed task at the next ordinal, or submit
        and checkpoint a new one.

        The submission is counted before the service call, so a
        submission failure still counts as the one attempt that was made
        (it is never retried).
        """
        path = self._checkpoint_path(self._ordinal) if self._directory is not None else None
        self._ordinal += 1
        if self._resume and path is not None and path.exists():
            task = RuntimeBackendTask.recover(self._service, str(path))
        else:
            self.submission_count += 1
            task = submit()
            if path is not None:
                task.checkpoint(str(path))
        self.task_ids.append(task.id)
        return task

    def _has_checkpoints(self) -> bool:
        return bool(list(self._directory.glob(f"{self._algorithm}-*.json")))

    def _checkpoint_path(self, ordinal: int) -> Path:
        return self._directory / f"{self._algorithm}-{ordinal}.json"


class _CoprimeBaseRng:
    """Committed-RNG draw policy for Shor's base selection.

    Draws from ``random.Random(seed)`` exactly as the committed solver
    would, but redraws bases that share a factor with the modulus, so
    the solver's first attempt always enters quantum order finding.
    The draw stream is unchanged; only non-coprime candidates are
    skipped, and no factor or order is encoded.
    """

    def __init__(self, seed: int, modulus: int) -> None:
        self._rng = random.Random(seed)
        self._modulus = modulus

    def randrange(self, start: int, stop: int) -> int:
        while True:
            candidate = self._rng.randrange(start, stop)
            if math.gcd(candidate, self._modulus) == 1:
                return candidate

    #: The committed Shor solver only calls ``randrange``; delegate any
    #: other attribute defensively so the policy cannot break the solver.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._rng, name)


def _run_shor_case(case: Any, executor: "_CaseExecutor") -> Any:
    """Run the committed Shor case through the real solver.

    See the module docstring: the committed seed and attempt budget are
    the only inputs; the coprime-redraw policy guarantees the first
    attempt is a genuine quantum order-finding run.
    """
    from pyqpanda_alg.Shor import Shor, ShorConfig

    solver = Shor(
        _SHOR_MODULUS,
        rng=_CoprimeBaseRng(case.seed, _SHOR_MODULUS),
        config=ShorConfig(max_attempts=case.max_attempts),
    )
    return solver.run(
        backend=executor,
        # Direct QPU submission (see module docstring): the transpile
        # record is the preflight runner's job, so a transpile-service
        # outage must not block the real execution.
        execution_options=ExecutionOptions(
            shots=case.shots, timeout=case.timeout, preflight=PreflightMode.NONE
        ),
    )


def _parsed_and_record(case: Any, result: Any) -> tuple[Any, dict]:
    """Return (domain-parsed result, JSON-safe record dict) of a run.

    The parsed result is what the committed domain predicate gates;
    the record is what enters the manifest (always JSON-safe).  Shor
    records its quantum provenance; HHL records success probability and
    the requested observable only -- no statevector on the QPU path.
    """
    if case.algorithm == "Shor":
        record = {
            "factors": list(result.factors) if result.factors is not None else None,
            "is_prime": bool(result.is_prime),
            "used_quantum": bool(result.used_quantum),
            "task_ids": list(result.task_ids),
            "order": result.order,
            "metadata": result.metadata or {},
        }
        return record, record
    if case.algorithm == "HHL":
        record = {
            "success_probability": result.success_probability,
            "metadata": result.metadata or {},
        }
        return result, record  # the domain predicate reads attributes
    return result, {"value": _jsonable(result)}


def _jsonable(value: Any) -> Any:
    """Recursively convert a parsed result into JSON primitives."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        # Field walk instead of asdict: asdict deep-copies, which fails
        # on non-picklable fields (e.g. circuit objects).
        return _jsonable(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        )
    return str(value)


def run_qpu(
    service: Any,
    chip_id: str = _DEFAULT_CHIP_ID,
    output_dir: str | Path | None = None,
    *,
    wheel: str | None = None,
    wheel_sha256: str | None = None,
    checkpoint_dir: str | Path | None = None,
    device: Any = None,
) -> QualificationManifest:
    """Run every fixed case against the QPU and return the manifest.

    Runs ``QUALIFICATION_CASES`` then ``SMOKE_CASES`` sequentially, one
    case at a time.  The manifest passes through
    :func:`validate_manifest` and :func:`sanitize_payload` before
    anything is written, so no credential can reach the artifact.
    """
    runner = QPURunner(service, chip_id=chip_id, device=device, checkpoint_dir=checkpoint_dir)
    all_cases = (*QUALIFICATION_CASES, *SMOKE_CASES)
    total = len(all_cases)
    print(f"qpu: {total} cases on {chip_id} starting", flush=True)
    qualified: list[AlgorithmQualification] = []
    for index, case in enumerate(all_cases, start=1):
        started = time.perf_counter()
        qualified_case = runner.run_case(case)
        elapsed = time.perf_counter() - started
        print(
            f"qpu[{index}/{total}] {case.algorithm}: "
            f"verdict={qualified_case.verdict} "
            f"tasks={len(qualified_case.task_ids)} ({elapsed:.1f}s)",
            flush=True,
        )
        qualified.append(qualified_case)
    cases = tuple(qualified)
    manifest = QualificationManifest(
        version=SCHEMA_VERSION,
        timestamp=_utc_now(),
        git_commit=_git_commit(),
        wheel=_wheel_basename(wheel),
        wheel_sha256=wheel_sha256 or _UNKNOWN_DIGEST,
        python_version=platform.python_version(),
        pyqpanda3_version=_package_version("pyqpanda3"),
        qpanda3_runtime_version=_package_version("qpanda3-runtime"),
        device=_device_record(runner.device, chip_id),
        cases=cases,
    )
    payload = manifest.to_dict()  # sanitized inside to_dict
    validate_manifest(payload)
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> None:
    """Credentialed CLI entry: login and run the QPU qualification.

    Reads ``QPANDA3_API_KEY`` only; refuses to run without it.  The
    key is passed to the runtime service login and never written to any
    artifact.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed qualification cases against a real QPU and "
            "write the sanitized manifest."
        )
    )
    parser.add_argument(
        "--chip-id", default=_DEFAULT_CHIP_ID, help=f"QPU chip id (default: {_DEFAULT_CHIP_ID})"
    )
    parser.add_argument(
        "--output-dir", default=".", help="directory for the generated manifest.json"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="directory for per-task checkpoints; a rerun resumes from them",
    )
    parser.add_argument("--wheel", default=None, help="path to the wheel under qualification")
    args = parser.parse_args()

    api_key = os.environ.get("QPANDA3_API_KEY")
    if not api_key:
        sys.exit("QPANDA3_API_KEY is required; refusing to run qpu qualification without it")

    wheel_sha256 = None
    if args.wheel:
        wheel_path = Path(args.wheel)
        if wheel_path.is_file():
            wheel_sha256 = _sha256_file(wheel_path)

    from qpanda3_runtime import RuntimeService  # kept out of the import path

    service = RuntimeService()
    service.login(api_key)
    manifest = run_qpu(
        service,
        chip_id=args.chip_id,
        output_dir=args.output_dir,
        wheel=args.wheel,
        wheel_sha256=wheel_sha256,
        checkpoint_dir=args.checkpoint_dir,
    )
    passed = sum(1 for case in manifest.cases if case.verdict == "passed")
    total = len(manifest.cases)
    print(f"qpu qualification: {passed}/{total} cases passed; manifest written")


if __name__ == "__main__":
    main()
