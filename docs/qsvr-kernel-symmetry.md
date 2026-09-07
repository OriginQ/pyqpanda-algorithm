# QSVR training-kernel symmetry

## Release note

`Quantum_SVR.k_kernel(X, Y)` now evaluates only one triangle when `X` and
`Y` contain the same samples in the same order. The fidelity kernel satisfies
`K(x, y) = |<phi(y)|phi(x)>|^2 = K(y, x)`, so the other triangle can reuse
those values. Equal array copies and equal nested lists also benefit.

For `n` training samples, simulator calls decrease from `n^2` to
`n(n+1)/2`. Diagonal entries are still evaluated through `dist`; they are
not replaced by constants. Distinct datasets use the original full pairwise
evaluation, including square matrices with different sample order. There is
no persistent cache, and the public method signatures are unchanged.

The optimization applies to the existing symmetric fidelity kernel.
Custom `dist` overrides must preserve that symmetry. Differences from the
old full loop can occur at floating-point roundoff because one mirrored
entry replaces a separately simulated evaluation.

## Validation

Base: `OriginQ/pyqpanda-algorithm` develop
`5f973efccb84bc193157d1ccebe32137e307293b`.

- Focused tests before the change: 7 passed, 4 failed. The four failures
  detected 16 simulator calls where 10 suffice for four training samples.
- Focused tests after the change: 11 passed.
- Repository pytest run: 29 passed in 23.78 seconds.
- Kernel entries agree within absolute/relative tolerance `1e-12` with
  independent NumPy state vectors constructed from RX, RY, and CZ matrices.
- Cases cover same-object inputs, equal copies, equal lists, duplicate
  points, reordered samples, rectangular matrices, singletons, and empty
  inputs. An SVR integration case compares against an independently
  constructed precomputed kernel, allowing libsvm's stopping tolerance.

Test filenames follow the repository's `Test_*.py` discovery convention.
The prose contribution guide's dotted `feature.test.py` name cannot be
imported by pytest's current default import mode.

## Reproduce

From the repository root, with the package's dependencies plus pytest
installed (including pandas and scikit-learn used by existing imports):

```bash
PYTHONPATH=pyqpanda-algorithm MPLBACKEND=Agg python -m pytest \
  -c test/pytest.ini -o addopts='' test/QAlgBase/Test_QSVR_kernel.py -q

PYTHONPATH=pyqpanda-algorithm MPLBACKEND=Agg python -m pytest \
  -c test/pytest.ini -o addopts='' test -q

PYTHONPATH=pyqpanda-algorithm MPLBACKEND=Agg python \
  benchmarks/benchmark_qsvr_kernel.py
```

`-o addopts=''` omits the optional Allure-reporting plugin flags.

## Local benchmark

Python 3.12.13, PyQPanda3 0.4.1, NumPy 2.3.5, Linux x86_64.
Three repetitions per size, fixed random seeds, median elapsed time,
alternating evaluation order after a simulator warm-up:

| Samples | Old calls | New calls | Old median | New median | Speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 136 | 12.90 ms | 6.74 ms | 1.91x |
| 32 | 1,024 | 528 | 50.71 ms | 26.17 ms | 1.94x |
| 64 | 4,096 | 2,080 | 244.51 ms | 116.94 ms | 2.09x |

Maximum absolute matrix difference: `1.1102230246251565e-15`.
These are local simulator measurements. They do not establish quantum
advantage or guarantee the same elapsed-time improvement on other systems.
