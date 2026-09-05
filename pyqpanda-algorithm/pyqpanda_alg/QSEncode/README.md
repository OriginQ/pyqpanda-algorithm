# QSEncode-Insight

QSEncode-Insight decides whether a probability distribution should actually be
compressed before it is represented by a quantum program. It combines an
explicit fidelity budget, several state-preparation strategies, five-repeat
compiled-resource auditing, and a refusal path when compression is not useful.

It is a new API alongside the existing `QSpare_Code`; the legacy class and its
behavior remain unchanged.

## Contribution boundary

QSEncode-Insight is an innovation-application tool, not a claim of a newly
invented low-level state-preparation decomposition. The circuit constructors
`amplitude_encode`, `sparse_isometry`, and `ds_quantum_state_preparation` are
provided by PyQPanda3. This package contributes the end-to-end decision layer:
canonical input handling, fast transforms, fidelity-budgeted `k*`, capability
filtering, repeated compiled-resource auditing, deterministic selection,
semantic verification, resource attribution, and explicit refusal.

```text
probability distribution
        ↓
canonicalization and padding
        ↓
explicit Walsh or Fourier mode
        ↓
fidelity target → minimal k*
        ↓
fixed candidate neighborhood
        ↓
preparation capability filter
        ↓
five-repeat compiled-resource audit
        ↓
frozen resource selector
        ↓
COMPRESS / DO_NOT_COMPRESS
        ↓
standard / audit verification
```

## Why it exists

The original QSEncode interface can truncate transformed coefficients with a
fixed cut length. A fixed cut does not express a fidelity requirement, and a
smaller coefficient vector does not guarantee a smaller compiled circuit.
State-preparation methods also have different qubit, ancilla, two-qubit-gate,
and depth costs. Some inputs are simply not worth compressing.

QSEncode-Insight instead follows this sequence:

1. derive the minimal retained coefficient count `k*` from a target fidelity;
2. evaluate the frozen neighborhood `{k*-1, k*, k*+1}`;
3. filter incompatible preparation methods;
4. compare real compiled resources under one fixed compiler profile;
5. recommend compression, or explicitly return `do_not_compress`.

This is compiled-resource evidence. It is not a claim of quantum speedup,
hardware acceleration, or guaranteed savings for every distribution.

## Source-checkout setup

This repository is a source checkout. From the repository root, install the
QSEncode-Insight lock file and expose the inner source directory to Python:

```bash
python -m pip install -r pyqpanda-algorithm/requirements-qseencode-insight.txt
```

PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\pyqpanda-algorithm).Path
```

Bash:

```bash
export PYTHONPATH="$PWD/pyqpanda-algorithm${PYTHONPATH:+:$PYTHONPATH}"
```

Validated contest environment:

- Python 3.14.2
- PyQPanda3 0.3.5
- NumPy 2.4.6
- SciPy 1.17.1
- SymPy 1.14.0
- Matplotlib 3.10.9

Submission compatibility was independently rechecked in a clean Python 3.12.10
environment with the same pinned scientific stack. The QSEncode-Insight suite
passed 237 tests, the full repository suite passed 254 tests, a wheel was built
and installed, and the API, CLI, and notebook Run-All smoke checks passed. The
benchmark claim itself remains tied to the frozen Python 3.14.2 environment
described below.

For the exact test environment, also install:

```bash
python -m pip install -r pyqpanda-algorithm/requirements-qseencode-insight-test.txt
```

The repository's historical `test/pytest.ini` enables Allure command-line
options. QSEncode-Insight provides a dependency-light test configuration that
does not require Allure:

```powershell
$env:PYTHONPATH = (Resolve-Path .\pyqpanda-algorithm).Path
python -m pytest -c pyqpanda-algorithm/pytest-qseencode.ini
```

```bash
export PYTHONPATH="$PWD/pyqpanda-algorithm${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -c pyqpanda-algorithm/pytest-qseencode.ini
```

## Quick start

```python
import numpy as np

from pyqpanda_alg.QSEncode import QSEncodeInsight

probabilities = np.array([
    0.0006917643261373052,
    0.015724004731018214,
    0.1261730210273901,
    0.3574112099154543,
    0.3574112099154544,
    0.1261730210273902,
    0.01572400473101823,
    0.0006917643261373052,
])

engine = QSEncodeInsight(
    basis="fourier",
    fidelity_target=0.99,
)
result = engine.analyze(probabilities)

print(result.selection.decision.value)
print(result.selection.selected_candidate_id)

artifact = engine.prepare(probabilities, result=result)
print(artifact.output_qubits)
```

For this fixed N=8 example, Fourier mode recommends
`compressed__k4__sparse_isometry`. The same distribution in Walsh mode returns
`do_not_compress`, demonstrating that truncation is not automatically useful.

## Reading a result

`InsightResult` is a deterministic, JSON-serializable snapshot. Its main
sections are:

- `InputSummary`: normalization, padding, and input hashes;
- `TransformDiagnostics`: basis convention and Parseval checks;
- `ErrorBudgetResult`: target fidelity, `k*`, and candidate neighborhood;
- `CapabilityReport`: method compatibility and failure reasons;
- `ResourceAudit`: five compiled attempts, medians, ranges, and hashes;
- `SelectionResult`: winner or refusal with primary-resource comparison;
- `SemanticVerification`: standard status or five-repeat audit evidence;
- `EvidenceScope`: whether the run is inside the validated contest scope.

Use `result.to_dict()` or `result.to_json(indent=2)` for structured reporting.
The result does not embed QProg or full OriginIR text; runnable programs are
returned separately by `prepare()`.

## Standard and audit verification

`verification="standard"` is the default. It performs input, transform,
error-budget, logical-preparation, capability, compilation, topology, basis,
resource, and selection checks. It deliberately does **not** perform a compiled
statevector semantic sweep, and reports `not_run_by_standard`.

`verification="audit"` reuses the same five compiled attempts for the actual
recommendation and certifies all five semantically. Only 5/5 passes produce a
valid audited recommendation. A failed audit does not select a different
method or `k`; `prepare()` blocks the uncertified recommendation unless the
caller explicitly requests a documented dense-baseline fallback.

## Evidence scope

`validated_default` currently requires:

- `fidelity_target=0.99`;
- an explicitly selected `walsh` or `fourier` basis;
- `N` in `{8, 16, 32, 64}`;
- the exact default method order and frozen selector policy;
- PyQPanda3 0.3.5 and the frozen five-repeat compiler profile.

Other supported configurations can still run, but are labeled
`outside_validated_scope`; they must not be described as benchmark-validated.
There is no automatic Walsh/Fourier selector in v1.

## CLI

Analyze an inline JSON list:

```bash
python -m pyqpanda_alg.QSEncode.cli analyze \
  --basis fourier \
  --fidelity-target 0.99 \
  --verification standard \
  --input-json '[0.1,0.2,0.3,0.4]'
```

Use `--input-file probabilities.json` instead of `--input-json` for a UTF-8
JSON file, and `--pretty` for indented output.

## Reproducible demo and evidence

- Notebook: `example/QAlgBase/QSEncode_Insight_Demo.ipynb`
- Benchmark summary: [BENCHMARK_EVIDENCE.md](BENCHMARK_EVIDENCE.md)
- Visual benchmark summary: [BENCHMARK_VISUAL_SUMMARY.md](BENCHMARK_VISUAL_SUMMARY.md)

The notebook uses only the small sealed N=8 example and does not run the
480-cell benchmark.

## Limitations

- The confirmed evidence is limited to the preregistered N<=64 scope.
- Dirichlet-family inputs were substantially weaker than the other main
  families in the generalization test.
- Some preparation constructors reject otherwise valid candidates; capability
  filtering and fallback behavior are therefore part of the product.
- Compiled gate/depth reductions are not evidence of end-to-end hardware
  speedup.
