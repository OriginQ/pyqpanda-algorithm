# QAE circuit construction benchmark

`QAE.run()` needs `2**m - 1` controlled Grover applications for `m` phase
estimation ancillas. Previously it rebuilt each application from Python,
calling the preparation callback `2**(m+1) - 1` times. Reusing the Grover
circuit within each run reduces preparation callback calls to three, and
builds one controlled variant per ancilla. Quantum gate count and oracle
query count are unchanged. Each new run rebuilds the circuit.

The callback must describe a fixed state-preparation operator within a run.
Changing its parameters between runs remains supported. This optimization
does not address the separate target-index issues discussed in PR #27.

## Reproduce

From the repository root, with runtime dependencies installed:

```sh
python benchmarks/qae_construction.py --baseline --label upstream --output before.json
python benchmarks/qae_construction.py --label reuse --output after.json
python benchmarks/qae_construction.py --baseline --build-only --ancillas 8 10 12 --label upstream-build --output build-before.json
python benchmarks/qae_construction.py --build-only --ancillas 8 10 12 --label reuse-build --output build-after.json
```

The baseline is loaded from upstream commit
`5f973efccb84bc193157d1ccebe32137e307293b` using `git show`, so that commit must
be present in the clone. This does not change the checkout. The test case is
a three-qubit preparation with five layers of RY/RZ rotations and a CNOT
chain. Results use a warmup and the median of three runs. JSON includes raw
timings and, for full runs, all ancilla probabilities. `--build-only` stops
at the simulator boundary and reports no simulation result.

## Local measurement, 2026-09-23

Windows x64, Python 3.12.14, pyqpanda3 0.4.1, NumPy 2.5.3, SciPy 1.18.1.

| Ancillas | Callback calls before/after | Build ms before/after | Total ms before/after |
|---|---|---|---|
| 4 | 31 / 3 | 5.50 / 0.75 | 9.44 / 4.55 |
| 6 | 127 / 3 | 20.81 / 1.61 | 38.43 / 18.22 |
| 7 | 255 / 3 | 41.60 / 2.67 | 79.97 / 44.90 |

All saved ancilla probability entries were identical between baseline and
optimized full runs. Construction-only medians at 8, 10 and 12 ancillas
were 77.00 / 4.24 ms, 327.97 / 14.87 ms and 1445.11 / 145.98 ms,
respectively. These larger cases do not establish end-to-end speedups.

The full simulation at 10 ancillas was stopped because of runtime cost.
No completion or accuracy result is claimed for that interrupted run.
Numbers depend on the preparation circuit and machine; simulator time can
dominate at larger widths. No quantum speedup or hardware result is claimed.

## Regression checks

Set `PYTHONPATH` to the repository's `pyqpanda-algorithm` package directory,
then run:

```sh
python -m pytest -o addopts= -q test/QAlgBase/Test_QAE_circuit_reuse.py
python -m pytest -o addopts= -q test
```

The new tests cover 1, 2, 4 and 6 ancillas, several single/multi-bit targets,
the full simulator probability distribution, known probabilities including
0 and 1, construction counts and input changes between runs. The complete
local suite passed: 56 tests (38 new cases and 18 existing cases).
