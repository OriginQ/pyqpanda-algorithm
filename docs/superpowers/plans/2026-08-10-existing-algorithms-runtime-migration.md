# Existing Algorithms Runtime Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all 13 existing public modules a real execution-backend path while preserving valid CPU calls and result types.

**Architecture:** Circuit builders remain backend-free. Executable algorithms resolve `backend=None` to LocalBackend, submit sampling/estimation through the execution package, and parse transport results into existing domain results. Multi-round algorithms use AlgorithmTask rather than embedding RuntimeService calls.

**Tech Stack:** pyqpanda3 circuits/Hamiltonian, pyqpanda_alg.execution, qpanda3-runtime contract stubs, pytest.

## Global Constraints

- No algorithm module imports `qpanda3_runtime` or authenticates directly.
- No runtime failure falls back to CPU.
- Existing valid CPU positional calls and result shapes remain compatible.
- New `backend` and `execution_options` arguments are keyword-only.
- Every executable algorithm gets `run()` and `submit()`; circuit-only utilities remain pure builders.
- Each migrated algorithm has LocalBackend and credential-free runtime-contract tests.

---

### Task 1: Lock circuit-only components to backend-free behavior

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/plugin.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QCmp/QCmp.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QAOA/default_circuits.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QAOA/dstate.py`
- Create: `test/runtime_migration/test_circuit_components.py`

**Interfaces:**
- Consumes: pyqpanda3 `QCircuit`, `QProg`
- Produces: circuit builders with no backend argument and no execution side effects

- [ ] **Step 1: Write circuit purity tests**

```python
from pyqpanda3.core import QCircuit
from pyqpanda_alg.QCmp import int_comparator
from pyqpanda_alg.QAOA.default_circuits import xy_mixer


def test_circuit_components_return_circuits_without_backend():
    assert isinstance(int_comparator(2, 1, [0, 1, 2]), QCircuit)
    assert isinstance(xy_mixer([0, 1], 0.25), QCircuit)
```

- [ ] **Step 2: Run focused tests before edits**

Run: `python -m pytest test/runtime_migration/test_circuit_components.py test/QAOA -q`

Expected: purity tests expose any signature/API drift and existing active tests remain green.

- [ ] **Step 3: Remove execution from helper paths only where present**

Move demonstration-only CPU execution out of callable helper bodies. Keep circuit construction and mathematical helpers in place; do not introduce a class hierarchy for pure functions.

- [ ] **Step 4: Verify components transpile through a runtime stub**

Add a contract assertion that each representative QProg reaches `QPandaRuntimeBackend.submit_sample()` after the test attaches measurements. Run: `python -m pytest test/runtime_migration/test_circuit_components.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/plugin.py pyqpanda-algorithm/pyqpanda_alg/QCmp pyqpanda-algorithm/pyqpanda_alg/QAOA test/runtime_migration/test_circuit_components.py
git commit -m "refactor: keep circuit components backend free"
```

### Task 2: Migrate QAOA, QUBO, and QmRMR to estimator execution

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QAOA/qaoa.py:297-996`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QUBO/QUBO.py:430-536`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QmRMR/QmRMR_core.py:171-278`
- Create: `test/runtime_migration/test_qaoa_backend.py`
- Create: `test/runtime_migration/test_qubo_backend.py`
- Create: `test/runtime_migration/test_qmrmr_backend.py`

**Interfaces:**
- Consumes: `resolve_backend`, `ExecutionOptions`, `ExecutionBackend.submit_estimate`, `AlgorithmTask`
- Produces: keyword-only backend support and resumable `submit()` for all three algorithm families

- [ ] **Step 1: Write a recording-estimator QAOA test**

```python
import sympy as sp
from pyqpanda_alg.QAOA.qaoa import QAOA
from test.execution.fakes import RecordingBackend


def test_qaoa_runtime_uses_estimator_without_cpu_fallback():
    x0 = sp.Symbol("x0")
    backend = RecordingBackend(expectations=[0.5, -0.25, -0.5])
    model = QAOA(x0)
    model.run(layer=1, optimizer_option={"options": {"maxiter": 1}}, backend=backend)
    assert backend.estimate_call_count > 0
    assert backend.sample_call_count == 0
```

- [ ] **Step 2: Run migration tests and verify signature failures**

Run: `python -m pytest test/runtime_migration/test_qaoa_backend.py test/runtime_migration/test_qubo_backend.py test/runtime_migration/test_qmrmr_backend.py -v`

Expected: FAIL because the algorithms do not accept backend arguments.

- [ ] **Step 3: Separate cost construction from execution**

QAOA builds a measurement-free QProg and Hamiltonian, then submits estimation. QUBO_QAOA delegates to the migrated QAOA API. QUBO_GAS remains for the sampling task. QmRMR converts its objective evaluations to the same estimator boundary. `run()` drives `submit().result()` for runtime and preserves existing LocalBackend return types.

- [ ] **Step 4: Run focused CPU and contract tests**

Run: `python -m pytest test/QAOA test/QAlgBase/Test_QUBO_*.py test/QAlgBase/Test_QmRMR_all_code_Feature_Selection.py test/runtime_migration/test_qaoa_backend.py test/runtime_migration/test_qubo_backend.py test/runtime_migration/test_qmrmr_backend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/QAOA pyqpanda-algorithm/pyqpanda_alg/QUBO pyqpanda-algorithm/pyqpanda_alg/QmRMR test
git commit -m "feat: run optimization algorithms through estimators"
```

### Task 3: Migrate Grover, QAE, QARM, and QUBO_GAS to sampling state machines

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/Grover/Grover_core.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QAE/QAE.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QARM/qarm.py:245-455`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QUBO/QUBO.py:350-440`
- Create: `test/runtime_migration/conftest.py`
- Create: `test/runtime_migration/test_grover_backend.py`
- Create: `test/runtime_migration/test_qae_backend.py`
- Create: `test/runtime_migration/test_qarm_backend.py`

**Interfaces:**
- Consumes: `submit_sample`, count normalization, `AlgorithmTask`
- Produces: checkpointable repeated-sampling algorithms and deprecated QARM QCloud entry

- [ ] **Step 1: Write a QARM regression test that forbids QCloudResult APIs**

```python
def test_qarm_runtime_uses_execution_backend(qarm_fixture, recording_backend):
    result = qarm_fixture.run(backend=recording_backend)
    assert result
    assert recording_backend.sample_call_count > 0
    assert "get_prob_dict" not in recording_backend.accessed_result_attributes
```

Define `qarm_fixture` in `test/runtime_migration/conftest.py` from the package's `QARM/dataset/data2.txt` fixture using the same support/confidence values as the active QARM test. Keep this fixture credential-free and deterministic. Reuse the root `recording_backend` fixture from `test/conftest.py`.

- [ ] **Step 2: Write iterative QAE/Grover task tests**

Use deterministic count sequences and assert that `submit()` returns before all rounds, `poll()` advances one round, checkpoint preserves the current round, and resume continues without replaying completed remote task IDs.

- [ ] **Step 3: Implement sampling adapters and compatibility warnings**

Replace internal `CPUQVM()` construction with `resolve_backend`. QAE and Grover use normalized counts. QARM retains `machine_type="QCloud"` for one release, emits `DeprecationWarning`, and requires an explicit backend rather than reading `api_key`. QUBO_GAS delegates to migrated GroverAdaptiveSearch.

- [ ] **Step 4: Run focused suites**

Run: `python -m pytest test/QAlgBase/Test_grover_* test/QARM test/runtime_migration/test_grover_backend.py test/runtime_migration/test_qae_backend.py test/runtime_migration/test_qarm_backend.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/Grover pyqpanda-algorithm/pyqpanda_alg/QAE pyqpanda-algorithm/pyqpanda_alg/QARM pyqpanda-algorithm/pyqpanda_alg/QUBO test
git commit -m "feat: migrate search algorithms to sampling tasks"
```

### Task 4: Migrate QKmeans, QPCA, QSVM, QSVR, and QSEncode

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QKmeans/QuantumKmeans.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QPCA/QPCA.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QSVM/quantum_kernel_svm.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QSVR/QSVR.py`
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QSEncode/QSEncode.py`
- Create: `test/runtime_migration/test_sampling_ml_algorithms.py`

**Interfaces:**
- Consumes: `submit_sample`, normalized probabilities, keyword-only execution options
- Produces: sampling-based distance, kernel, phase-distribution, and encoding results

- [ ] **Step 1: Write public-entry-point backend-use tests**

```python
import numpy as np
from pyqpanda_alg.QKmeans import QuantumKmeans
from pyqpanda_alg.QPCA import qpca
from pyqpanda_alg.QSEncode import QSpare_Code
from pyqpanda_alg.QSVM import QuantumKernel_vqnet
from pyqpanda_alg.QSVR import Quantum_SVR


def assert_sampled(callable_, recording_backend):
    before = recording_backend.sample_call_count
    callable_()
    assert recording_backend.sample_call_count > before


def test_public_sampling_entry_points_use_supplied_backend(recording_backend):
    points = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])

    assert_sampled(lambda: QuantumKmeans(k=2).fit(points, backend=recording_backend), recording_backend)
    assert_sampled(lambda: qpca(points, 1, backend=recording_backend), recording_backend)
    assert_sampled(
        lambda: QuantumKernel_vqnet(n_qbits=2).evaluate(x, backend=recording_backend),
        recording_backend,
    )
    assert_sampled(lambda: Quantum_SVR(x, y).get_res(backend=recording_backend), recording_backend)
    assert_sampled(
        lambda: QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(backend=recording_backend),
        recording_backend,
    )
```

- [ ] **Step 2: Run and verify backend-argument failures**

Run: `python -m pytest test/runtime_migration/test_sampling_ml_algorithms.py -v`

Expected: FAIL for unsupported backend parameters or internal CPU execution.

- [ ] **Step 3: Implement hardware-measurable paths**

Add keyword-only `backend` and `execution_options` to the existing execution entry points rather than inventing a common method: `QuantumKmeans.fit`, `qpca`, `QuantumKernel_vqnet.evaluate`, `Quantum_SVR.get_res`/`show_res`, and `QSpare_Code.Quantum_Res`. QKmeans/QSVM/QSVR derive distance or kernel values from ancilla/overlap counts. QPCA returns the sampled phase-estimation distribution. QSEncode uses `submit_statevector` only when the backend advertises that capability; its runtime path returns counts/probabilities or explicitly requested observables and otherwise raises `DeviceCapabilityError` before submission.

- [ ] **Step 4: Run active CPU tests and contract tests**

Run: `python -m pytest test/QPCA test/QSVM test/QAlgBase/Test_class_qsvr_Quantum_SVR.py test/QAlgBase/Test_class_basic_sparecode_QSpare_Code.py test/runtime_migration/test_sampling_ml_algorithms.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/QKmeans pyqpanda-algorithm/pyqpanda_alg/QPCA pyqpanda-algorithm/pyqpanda_alg/QSVM pyqpanda-algorithm/pyqpanda_alg/QSVR pyqpanda-algorithm/pyqpanda_alg/QSEncode test
git commit -m "feat: add runtime sampling to data algorithms"
```

### Task 5: Migrate QSVD to observable-based optimization

**Files:**
- Modify: `pyqpanda-algorithm/pyqpanda_alg/QSVD/QSVD.py`
- Create: `test/runtime_migration/test_qsvd_backend.py`

**Interfaces:**
- Consumes: `submit_estimate`, `VariationalSession`
- Produces: QSVD cost evaluation without runtime state-vector access

- [ ] **Step 1: Write a runtime estimator test**

```python
def test_qsvd_runtime_cost_uses_observable_estimates(recording_backend):
    solver = SVD([[1.0, 0.0], [0.0, 0.5]])
    solver.QSVD_min(backend=recording_backend, maxiter=1)
    assert recording_backend.estimate_call_count > 0
    assert recording_backend.statevector_call_count == 0
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest test/runtime_migration/test_qsvd_backend.py -v`

Expected: FAIL because QSVD constructs CPUQVM and StateVector internally.

- [ ] **Step 3: Express the overlap cost as measurable circuits/observables**

Keep the existing CPU result contract. Runtime optimization consumes estimator results and records statistical uncertainty. Reject requests for exact singular vectors when the backend has no state-vector or tomography capability.

- [ ] **Step 4: Run QSVD correctness and backend tests**

Run: `python -m pytest test/QAlgBase/Test_QSVD_SVD.py test/runtime_migration/test_qsvd_backend.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg/QSVD test/runtime_migration/test_qsvd_backend.py
git commit -m "feat: evaluate qsvd through observables"
```

### Task 6: Restore tests and enforce the no-internal-QVM rule

**Files:**
- Modify or delete: `test/legacy_disabled/*`
- Create: `test/meta/test_backend_boundary.py`
- Create: `test/coverage-baseline.json`
- Create: `tools/check_coverage_baseline.py`
- Modify: `Tutorials/source/index.rst`
- Create: `Tutorials/source/runtime_algorithms.rst`

**Interfaces:**
- Consumes: Tasks 1-5
- Produces: complete migration evidence for all current modules

- [ ] **Step 1: Write the architecture-boundary test**

```python
from pathlib import Path


def test_algorithms_do_not_construct_cpuqvm_directly():
    source_root = Path("pyqpanda-algorithm/pyqpanda_alg")
    offenders = []
    for path in source_root.rglob("*.py"):
        if "execution" in path.parts:
            continue
        if "CPUQVM()" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert offenders == []
```

- [ ] **Step 2: Restore or delete every legacy-disabled test**

For each file listed in `test/legacy_disabled/README.md`, restore it under the active module directory if it asserts a public behavior; delete it if it duplicates another test or asserts obsolete pyqpanda2 text output. Update the README until its list is empty, then remove the directory.

- [ ] **Step 3: Add one LocalBackend and one runtime-contract test per executable public algorithm**

The parameterized test inventory must name QAOA, QARM, QKmeans, QPCA, QSVM, QUBO_QAOA, QUBO_GAS, QAE, QSVD, QSVR, Grover, GroverAdaptiveSearch, QmRMR, and QSEncode. QCmp and circuit helpers belong to the transpilation inventory.

Generate JSON coverage for every pre-existing algorithm package and store its measured line/branch percentages in `test/coverage-baseline.json`. `check_coverage_baseline.py` compares a new coverage JSON file with that committed mapping and exits nonzero when any existing module decreases; new VQE/HHL/Shor and execution modules use their explicit plan thresholds instead.

- [ ] **Step 4: Run the entire suite and boundary audit**

Run: `python -m pytest test -q`

Expected: PASS with no `legacy_disabled` directory and no direct algorithm-side `CPUQVM()` construction.

Run: `python -m pytest test --cov=pyqpanda_alg --cov-branch --cov-report=json:test/coverage-current.json && python tools/check_coverage_baseline.py test/coverage-baseline.json test/coverage-current.json`

Expected: PASS with no legacy module below its committed baseline.

- [ ] **Step 5: Commit**

```bash
git add pyqpanda-algorithm/pyqpanda_alg test Tutorials tools/check_coverage_baseline.py
git commit -m "test: complete existing algorithm migration"
```
