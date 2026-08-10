# Capability-Based Execution Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one execution abstraction that supports CPU, qpanda3-runtime, synchronous calls, resumable algorithm tasks, and explicit device capability failures.

**Architecture:** Backend methods always submit work and return task objects. `LocalBackend` returns completed tasks, while `QPandaRuntimeBackend` adapts QTaskManager and VQSession without leaking authentication into algorithms. `AlgorithmTask` is a serializable local state machine that coordinates one or more backend tasks.

**Tech Stack:** Python dataclasses, typing Protocol/ABC, pyqpanda3, optional qpanda3-runtime, pytest.

## Global Constraints

- Runtime support is optional and imported lazily.
- `QPandaRuntimeBackend(service, device)` receives an already logged-in service and explicit device.
- No runtime failure silently falls back to LocalBackend.
- Task checkpoints never contain API keys, tokens, RuntimeService objects, or QDevice objects.
- Submission is not automatically retried; only idempotent status queries use bounded backoff.
- Standard tests use a runtime stub and need no network credentials.

---

### Task 1: Define errors, options, and capabilities

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/errors.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/options.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/capabilities.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/results.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/__init__.py`
- Create: `test/execution/test_value_objects.py`

**Interfaces:**
- Produces: `ExecutionOptions`, `PreflightMode`, `BackendCapabilities`, `SampleBatchResult`, `EstimateBatchResult`, `StatevectorBatchResult`, and the public exception hierarchy

- [ ] **Step 1: Write failing value-object tests**

```python
import pytest
from pyqpanda_alg.execution import ExecutionOptions, PreflightMode


def test_execution_options_reject_nonpositive_shots():
    with pytest.raises(ValueError, match="shots"):
        ExecutionOptions(shots=0)


def test_default_options_are_safe_for_runtime():
    options = ExecutionOptions()
    assert options.shots == 1000
    assert options.timeout == 1800.0
    assert options.preflight is PreflightMode.TRANSPILE_ONLY
    assert options.is_mapping is True
```

- [ ] **Step 2: Run and verify import failure**

Run: `python -m pytest test/execution/test_value_objects.py -v`

Expected: FAIL because `pyqpanda_alg.execution` does not exist.

- [ ] **Step 3: Implement immutable dataclasses and exceptions**

Define these exception classes exactly:

```python
class AlgorithmExecutionError(RuntimeError):
    pass

class AlgorithmInputError(AlgorithmExecutionError):
    pass

class MissingRuntimeDependencyError(AlgorithmExecutionError):
    pass

class BackendUnavailableError(AlgorithmExecutionError):
    pass

class DeviceCapabilityError(AlgorithmExecutionError):
    pass

class TranspilationError(AlgorithmExecutionError):
    pass

class TaskSubmissionError(AlgorithmExecutionError):
    pass

class TaskTimeoutError(AlgorithmExecutionError):
    pass

class TaskRecoveryError(AlgorithmExecutionError):
    pass

class ResultDecodingError(AlgorithmExecutionError):
    pass
```

Implement `ExecutionOptions` as a frozen dataclass with `shots=1000`, timeout expressed publicly in seconds as `timeout=1800.0`, tuple-valued optional `specified_block`, and booleans matching qpanda3-runtime names. Reject nonpositive shots/timeouts. Implement `PreflightMode` values `NONE`, `TRANSPILE_ONLY`, and `FAKE_EXECUTE`.

`SampleBatchResult` stores `counts: tuple[dict[str, int], ...]`, `shots`, and sanitized `raw_metadata`; `single_counts()` requires exactly one entry. `EstimateBatchResult` stores `values: tuple[float, ...]` and sanitized `raw_metadata`; `single_value()` requires exactly one value. `StatevectorBatchResult` stores immutable copied complex arrays and `single_statevector()` requires exactly one entry. These wrappers remove local/runtime result-shape ambiguity without discarding transport diagnostics. `BackendCapabilities` explicitly declares sampling, estimation, variational-session, state-vector, and tomography support.

- [ ] **Step 4: Run tests**

Run: `python -m pytest test/execution/test_value_objects.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution/test_value_objects.py
git commit -m "feat: define execution value objects"
```

### Task 2: Define backend and backend-task protocols

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/backend.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/backend_task.py`
- Create: `test/execution/test_backend_protocol.py`

**Interfaces:**
- Consumes: `ExecutionOptions`, `BackendCapabilities`
- Produces: `ExecutionBackend`, `BackendTask[T]`, `CompletedBackendTask[T]`, `TaskStatus`

- [ ] **Step 1: Write failing completed-task tests**

```python
from pyqpanda_alg.execution import CompletedBackendTask, TaskStatus


def test_completed_task_has_stable_result():
    task = CompletedBackendTask({"0": 10}, task_id="local-1")
    assert task.status() is TaskStatus.SUCCEEDED
    assert task.result() == {"0": 10}
    assert task.id == "local-1"
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/execution/test_backend_protocol.py -v`

Expected: FAIL because task types are absent.

- [ ] **Step 3: Implement protocols with one result contract**

Define `TaskStatus` values `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, and `TIMED_OUT`. `BackendTask` exposes `id`, `status()`, `try_result()`, `result(timeout=None)`, and `checkpoint(path=None)`. `ExecutionBackend` exposes `submit_sample`, `submit_estimate`, `submit_statevector`, `create_variational_session`, and `capabilities` with keyword-only `ExecutionOptions`. Backends that do not advertise state-vector support raise `DeviceCapabilityError` before submission.

- [ ] **Step 4: Run tests and type-check public signatures**

Run: `python -m pytest test/execution/test_backend_protocol.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution/test_backend_protocol.py
git commit -m "feat: define backend task protocols"
```

### Task 3: Implement LocalBackend sampling and estimation

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/local.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/result_normalization.py`
- Create: `test/execution/test_local_backend.py`

**Interfaces:**
- Consumes: `ExecutionBackend`, pyqpanda3 `QProg`, `Hamiltonian`
- Produces: `LocalBackend.submit_sample() -> BackendTask[SampleBatchResult]`, `LocalBackend.submit_estimate() -> BackendTask[EstimateBatchResult]`, `LocalBackend.submit_statevector() -> BackendTask[StatevectorBatchResult]`, and `resolve_backend()`

- [ ] **Step 1: Write Bell sampling and expectation tests**

```python
import numpy as np
from pyqpanda3.core import CNOT, H, QProg, measure
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend, resolve_backend


def bell_program():
    prog = QProg()
    prog << H(0) << CNOT(0, 1) << measure([0, 1], [0, 1])
    return prog


def test_local_sample_returns_counts():
    task = LocalBackend().submit_sample(
        bell_program(), options=ExecutionOptions(shots=200)
    )
    counts = task.result().single_counts()
    assert sum(counts.values()) == 200
    assert set(counts).issubset({"00", "11"})


def test_local_estimate_returns_float():
    prog = QProg()
    prog << H(0)
    task = LocalBackend().submit_estimate(
        (prog, Hamiltonian({"X0": 1.0})),
        options=ExecutionOptions(shots=1),
    )
    assert abs(task.result().single_value() - 1.0) < 1e-9


def test_local_statevector_returns_normalized_state():
    prog = QProg()
    prog << H(0)
    state = LocalBackend().submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
    assert np.allclose(np.abs(state) ** 2, [0.5, 0.5])


def test_resolve_backend_defaults_only_when_none():
    local = resolve_backend(None)
    supplied = LocalBackend()
    assert isinstance(local, LocalBackend)
    assert resolve_backend(supplied) is supplied
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/execution/test_local_backend.py -v`

Expected: FAIL because LocalBackend is absent.

- [ ] **Step 3: Implement the minimal local adapter**

Sampling must preserve counts and shots in `SampleBatchResult`. Estimation must reject circuits containing measurements, call pyqpanda3 expectation APIs, and return `EstimateBatchResult`. State-vector execution must reject measured circuits and return a copied, read-only complex array. Implement `resolve_backend(backend)` as `LocalBackend()` only when `backend is None`; never catch a runtime error and replace the supplied backend. Result normalization owns bit-string ordering and probability conversion helpers; algorithms must not duplicate them.

- [ ] **Step 4: Run focused and existing CPU tests**

Run: `python -m pytest test/execution/test_local_backend.py test/QAOA -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution/test_local_backend.py
git commit -m "feat: add local execution backend"
```

### Task 4: Implement resumable AlgorithmTask state machines

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/algorithm_task.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/checkpoint.py`
- Create: `test/execution/test_algorithm_task.py`
- Create: `test/execution/test_checkpoint_security.py`

**Interfaces:**
- Consumes: `BackendTask`, `TaskStatus`, `TaskTimeoutError`, `TaskRecoveryError`
- Produces: `AlgorithmTask[T]`, `AlgorithmStep`, checkpoint format version `1`

- [ ] **Step 1: Write a two-step state-machine test**

```python
from pyqpanda_alg.execution import AlgorithmTask, CompletedBackendTask, TaskStatus


def test_poll_advances_two_backend_steps():
    values = iter([2, 3])

    def advance(state):
        value = next(values)
        state["total"] += value
        return CompletedBackendTask(value), state["total"] >= 5

    task = AlgorithmTask(algorithm="test-counter", initial_state={"total": 0}, advance=advance)
    assert task.poll() is TaskStatus.RUNNING
    assert task.poll() is TaskStatus.SUCCEEDED
    assert task.result() == 5
```

- [ ] **Step 2: Write a credential-redaction checkpoint test**

```python
def test_checkpoint_does_not_serialize_credentials(tmp_path):
    task = AlgorithmTask(
        algorithm="test-counter",
        initial_state={"total": 0},
        metadata={"api_key": "secret", "nested": {"token": "secret-token"}},
        advance=lambda state: (CompletedBackendTask(0), True),
    )
    path = task.checkpoint(tmp_path / "task.json")
    payload = path.read_text(encoding="utf-8")
    assert "secret" not in payload
    assert "token" not in payload
```

- [ ] **Step 3: Implement deterministic polling and JSON checkpoints**

`AlgorithmTask.id` is a generated algorithm ID; `backend_task_ids` stores every remote ID. The constructor accepts `algorithm`, `initial_state`, `metadata`, and a process-local `advance` callback; the callback itself is never serialized. Checkpoints contain format version, algorithm name, serializable state, task IDs, backend identity, and user metadata after recursive credential-key filtering. `AlgorithmTask.resume(path, backend=backend)` reconstructs the callback through a registered algorithm factory keyed by `algorithm`; it never deserializes arbitrary Python objects.

- [ ] **Step 4: Run state, timeout, recovery, and redaction tests**

Run: `python -m pytest test/execution/test_algorithm_task.py test/execution/test_checkpoint_security.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution
git commit -m "feat: add resumable algorithm tasks"
```

### Task 5: Implement the optional qpanda3-runtime adapter

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/runtime.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/runtime_task.py`
- Create: `test/execution/fakes.py`
- Create: `test/conftest.py`
- Create: `test/execution/test_runtime_backend_contract.py`
- Create: `test/execution/test_optional_runtime_import.py`

**Interfaces:**
- Consumes: `ExecutionBackend`, `ExecutionOptions`, `BackendCapabilities`
- Produces: `QPandaRuntimeBackend(service, device)`, `RuntimeBackendTask`

- [ ] **Step 1: Write a credential-free runtime contract test**

```python
def test_runtime_sample_forwards_exact_options(fake_runtime_service, fake_device, bell_program):
    backend = QPandaRuntimeBackend(fake_runtime_service, fake_device)
    options = ExecutionOptions(shots=321, specified_block=(0, 1))
    task = backend.submit_sample(bell_program, options=options)
    assert fake_runtime_service.sample_calls[0]["shots"] == 321
    assert fake_runtime_service.sample_calls[0]["specified_block"] == [0, 1]
    assert task.result().single_counts() == {"00": 160, "11": 161}
```

- [ ] **Step 2: Write a missing-extra test**

Simulate an import environment without `qpanda3_runtime` and assert that importing `pyqpanda_alg` succeeds while constructing `QPandaRuntimeBackend` raises `MissingRuntimeDependencyError` with the exact install command.

- [ ] **Step 3: Implement lazy imports and task adaptation**

In `fakes.py`, define `RecordingBackend`, `FakeRuntimeService`, `FakeQTaskManager`, and `FakeDevice`; `RecordingBackend` records sample, estimate, and state-vector calls. Expose repository-wide `recording_backend`, `fake_runtime_service`, `fake_device`, and `runtime_backend` pytest fixtures from `test/conftest.py`. Also define small deterministic `bell_program`, `five_qubit_prog`, `ansatz`, and `observable` fixtures there for Tasks 5-7. Map `sample()` and `estimate()` options exactly to qpanda3-runtime. Adapt `QTaskManager.id()`, `try_get_result()`, `get_result_sync()`, `get_result_async()`, `check_point()`, and task-state recovery into the batch result wrappers. `QPandaRuntimeBackend.submit_statevector()` always raises `DeviceCapabilityError` before service submission. Convert only transport failures into the public exception hierarchy and retain the original exception as `__cause__`.

- [ ] **Step 4: Run contract tests without network access**

Run: `python -m pytest test/execution/test_runtime_backend_contract.py test/execution/test_optional_runtime_import.py -v -m runtime_contract`

Expected: PASS without API credentials.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution
git commit -m "feat: adapt qpanda3 runtime tasks"
```

### Task 6: Add device capabilities and preflight modes

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/execution/runtime.py`
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/preflight.py`
- Create: `test/execution/test_runtime_capabilities.py`
- Create: `test/execution/test_preflight.py`

**Interfaces:**
- Consumes: QDevice `available_qubits`, `basic_gates`, `chip_topo_edges`, `fake_backend`
- Produces: static validation, `TRANSPILE_ONLY`, and `FAKE_EXECUTE` behavior

- [ ] **Step 1: Write qubit-capacity and gate-set failure tests**

```python
import pytest
from pyqpanda_alg.execution import DeviceCapabilityError


def test_preflight_rejects_circuit_larger_than_device(runtime_backend, five_qubit_prog):
    runtime_backend.device.available_qubits.return_value = [0, 1]
    with pytest.raises(DeviceCapabilityError, match="requires 5 qubits"):
        runtime_backend.submit_sample(five_qubit_prog, options=ExecutionOptions())
```

- [ ] **Step 2: Write the multiprocessing guidance test**

Force FakeBackend execution to report the qpanda3-runtime multiprocessing entry error. Assert the public exception names `fake_execute` and shows `if __name__ == "__main__":`.

- [ ] **Step 3: Implement capability checks before submission**

Check circuit qubits, observable qubits, device availability, gate set, topology, and specified block. `TRANSPILE_ONLY` calls the supported FakeBackend/runtime transpile path. `FAKE_EXECUTE` runs the fake task and records its task/result metadata before the real submission; it never invokes `run_on_real_device()` for VQSession because that API is not guaranteed.

- [ ] **Step 4: Run capability and preflight tests**

Run: `python -m pytest test/execution/test_runtime_capabilities.py test/execution/test_preflight.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution
git commit -m "feat: validate runtime device capabilities"
```

### Task 7: Add variational-session abstraction

**Files:**
- Create: `pyqpanda-algorithm/pyqpanda_alg/execution/variational.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/execution/local.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/execution/runtime.py`
- Create: `test/execution/test_variational_session.py`

**Interfaces:**
- Consumes: pyqpanda3 `VQCircuit`, Hamiltonian; qpanda3-runtime VQSession
- Produces: `VariationalSession.run(parameters) -> BackendTask[float]`, context-manager release semantics

- [ ] **Step 1: Write release-on-error tests**

```python
import pytest


def test_runtime_variational_session_releases_on_error(runtime_backend, ansatz, observable):
    session = runtime_backend.create_variational_session(ansatz, observable, options=ExecutionOptions())
    with pytest.raises(RuntimeError):
        with session:
            session.raw_session.raise_on_run = RuntimeError("failure")
            session.run([0.1, 0.2])
    assert session.raw_session.release_calls == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/execution/test_variational_session.py -v`

Expected: FAIL because the abstraction is absent.

- [ ] **Step 3: Implement local and runtime sessions**

Both variants expose the same context manager. Runtime uses `service.vqsession(ansatz, device, shots, life_time, observable)` and releases idempotently. Local evaluates concrete parameters through pyqpanda3. Checkpoint state stores parameters/history but never a live session ID as a durable guarantee.

- [ ] **Step 4: Run session tests**

Run: `python -m pytest test/execution/test_variational_session.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/execution test/execution/test_variational_session.py
git commit -m "feat: add variational execution sessions"
```

### Task 8: Publish and verify the execution API

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/execution/__init__.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/__init__.py`
- Create: `Tutorials/source/execution.rst`
- Modify: `Tutorials/source/index.rst`
- Create: `test/execution/test_public_api.py`

**Interfaces:**
- Consumes: Tasks 1-7
- Produces: stable imports documented for all later algorithm plans

- [ ] **Step 1: Write public export tests**

```python
from pyqpanda_alg.execution import (
    AlgorithmTask,
    BackendCapabilities,
    ExecutionOptions,
    LocalBackend,
    QPandaRuntimeBackend,
    StatevectorBatchResult,
    resolve_backend,
)


def test_execution_public_types_are_importable():
    assert all((AlgorithmTask, BackendCapabilities, ExecutionOptions, LocalBackend, QPandaRuntimeBackend, StatevectorBatchResult, resolve_backend))
```

- [ ] **Step 2: Run and verify any missing exports**

Run: `python -m pytest test/execution/test_public_api.py -v`

Expected: FAIL until every symbol is exported.

- [ ] **Step 3: Export only the approved public surface and document usage**

Document CPU default, explicit runtime construction, optional dependency installation, sample/estimate/state-vector behavior, FakeBackend constraints, checkpoint recovery, and the no-silent-fallback rule. State-vector examples are LocalBackend-only; runtime attempts fail with `DeviceCapabilityError`. Examples use environment variables in application code and never embed credentials.

- [ ] **Step 4: Run all execution tests, enforce coverage, and build docs**

Run: `python -m pytest test/execution -q --cov=pyqpanda_alg.execution --cov-branch --cov-report=term-missing --cov-fail-under=85`

Expected: PASS.

Run: `python -m coverage report --include="*/execution/algorithm_task.py,*/execution/result_normalization.py,*/execution/errors.py" --fail-under=100`

Expected: critical task-state, result-decoding, and exception modules have no uncovered executable branches.

Run: `python -m sphinx -W -b html Tutorials/source Tutorials/build/html`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg Tutorials test
git commit -m "docs: publish execution backend API"
```
