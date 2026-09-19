# Budgeted adaptive QKMeans assignment (counts only)

This optional intermediate algorithm makes **one point-to-candidate decision**.
It does not change the existing `QuantumKmeans` API or `fit` loop, update centers,
normalize data, restart a clustering run, or silently resolve an uncertain tie.

```python
from pyqpanda_alg.QKmeans import (
    AssignmentResult, adaptive_assign, build_assignment_program,
)

result = adaptive_assign(
    point, centroids, run_counts,
    batch_shots=64, max_shots=4096, alpha=0.05,
)
```

## Objective and circuit

`point` and every centroid contain exactly two finite real coordinates `(x, y)`
in `[-1, 1]`; booleans, strings and complex numbers are rejected. Candidate
iterables must be nonempty. All coordinates are copied to immutable float
snapshots before any runner call, preserving original candidate indices.

The legacy angle encoding is retained:

- `theta = (y + 1) * pi / 2`, `phi = (x + 1) * pi / 2`;
- q0 holds the point and q1 holds the centroid, each prepared by
  `U3(theta, phi, 0)`;
- q2 is the SWAP-test ancilla, measured **only into c0**.

In an ideal circuit the objective is
`p_i = (1 - |<psi(point)|psi(centroid_i)>|^2) / 2`.
The assignment minimizes the observed ancilla-one probability. This encoding
is **not a general Euclidean-distance oracle** and can map distinct coordinates
to the same state. With a noisy runner the target is its stationary measured
probability, which need not equal the ideal objective and can exceed 0.5.

`build_assignment_program(point, centroid)` returns a fresh caller-owned
three-qubit `QProg`. Its gate vocabulary is `U3`, `H`, `T` (including dagger),
and `CNOT`, plus measurement. It uses the standard exact Fredkin identity
`CX(right,left); CCX(control,left,right); CX(right,left)`, with CCX expanded
into the usual H/T/CNOT gates. There are no new gates, multi-control nodes,
backend objects, state reads or probability queries in this module.
This supports external counts runners implementing those standard operations;
it is not a claim of compatibility with every possible backend.

The tests compare the full eight-basis-state amplitude matrix of the
Fredkin decomposition against the SDK's independent `SWAP(...).control(...)`
implementation, including phases. They also compare full encoded circuits.
State-vector access is restricted to those independent **test references**,
never used to make an assignment decision.

## External counts runner contract

`run_counts(program, shots)` is supplied explicitly by the caller. It must:

1. Prepare a fresh `|000>` state and execute the supplied circuit on every shot.
2. Return a mapping such as `{'0': 49, '1': 15}` for **c0 only**. Keys must be
   single-bit strings `'0'` or `'1'`; an absent outcome is zero. Empty, multibit,
   integer-key and probability-valued results are rejected.
3. Return nonnegative integral counts (Python or NumPy integers, not booleans)
   summing **exactly** to the requested shots. Do not renormalize, postselect,
   discard failed shots, or substitute a probability estimate.
4. For the statistical guarantee, deliver independent Bernoulli samples of a
   fixed probability for each candidate throughout this assignment. Batches
   must not recycle earlier observations. Independent candidate streams are
   sufficient; the union-bound step itself requires no cross-candidate
   independence. Each candidate's full stream must retain its stationary
   Bernoulli law under the adaptive sampling protocol.

Each callback receives a newly built program: mutating or clearing it cannot
alter a later batch's program. A runner that changes a circuit *before executing
that batch* is responsible for the changed measurement law. There is no cache
of programs or observations across `adaptive_assign` calls. Invalid counts and
runner exceptions propagate immediately, without retries or fallback execution.
The algorithm caps requested shots; it cannot police a runner's hidden work,
ignored shot requests, or physical execution failures.

## Budget, scheduling, and results

`batch_shots` is a positive integer; `max_shots` is a nonnegative **global** shot
budget; neither accepts booleans. `alpha` is finite and strictly between 0 and 1.
Active candidates are visited in original-index round-robin order, one full
batch at a time. After every batch all previous counts for that candidate are
retained and its interval is updated. Elimination occurs immediately; eliminated
candidates are never sampled again. If fewer than `batch_shots` remain, stop:
there is no partial final batch and no overspending. With a small budget,
later candidates may remain completely unmeasured. Increasing a budget also
increases the look allowance in the confidence radius, so early stopping need
not be monotone in the configured budget.

`AssignmentResult` is a frozen dataclass with immutable tuple fields:

| Field | Meaning |
| --- | --- |
| `status` | `single_candidate`, `separated`, or `undecided` |
| `winner` | Original candidate index, or `None` when undecided |
| `means` | Cumulative empirical probabilities; `None` for unmeasured entries |
| `intervals` | Last intervals, clipped to `[0, 1]`, **not** to `[0, 0.5]` |
| `shots_per_centroid` | Cumulative shots, including eliminated candidates |
| `total_shots` | Sum of per-candidate shots, no greater than `max_shots` |
| `active` | Tuple of surviving original indices, not a ranking |

An unmeasured interval is `[0, 1]`. One input candidate returns
`single_candidate`, index 0, and zero shots: this is a trivial choice, not
measured evidence. Otherwise only a single survivor yields `separated`.
Budget exhaustion with multiple survivors returns `undecided` and `winner=None`,
even when an empirical argmin exists. Identical centers are not special-cased:
they can exhaust the budget without separation. Under the stated sampling
assumptions, an incorrect separation remains possible on the failure event.

## Finite-look simultaneous bound

Let `K` be the original candidate count and
`L = floor(max_shots / batch_shots)`. Candidate i's j-th look has the fixed
sample size `n = j * batch_shots`. At any sampled look use

```text
r(n) = sqrt(log(2 * K * max(1, L) / alpha) / (2*n))
interval_i = [max(0, mean_i-r(n)), min(1, mean_i+r(n))]
```

The implementation evaluates the logarithm in log space to avoid overflow.
Hoeffding's two-sided bound allocates at most `alpha / (K*L)` failure probability
to each potential candidate/look pair when `L > 0`. Union over all `K*L` pairs
covers the data-dependent subset actually visited. This is conservative: `L`
is a global maximum batch count but is allowed for every candidate in the
union bound. `L=0` causes no sampling and no nontrivial separation.

Eliminate i only when `lower_i > min(upper_j for j in active)` (strict `>`).
On the simultaneous coverage event, a true minimum cannot satisfy this
inequality, and any returned separated winner is the unique measured-probability
minimizer. Therefore, **for one entire assignment, including all its looks**,

```text
Pr(returning an incorrect separated winner) <= alpha.
```

This is **not** a general conditional posterior statement
`Pr(incorrect | a separated winner was returned) <= alpha`.
There is no promise that the algorithm will return a separated winner.
The bound assumes the stationary independent-shot measurement model; it is not
proof of hardware fidelity, absence of drift/correlations, lack of readout bias,
or agreement with the ideal state overlap. An arbitrary counts provider cannot
be certified statistically merely by passing schema validation.

For multiple points or clustering iterations, callers must allocate an overall
error budget (for example, predetermined per-call alpha values whose sum is
at most the desired global error probability). Repeatedly rerunning undecided
calls with fresh alpha is not free error control. No full confidence-sequence
framework or iterative fit policy is provided here.

## Scope and provenance

Adaptive sampling and successive candidate elimination are established methods,
not a new best-arm-identification theorem. The repository-specific contribution
is their concrete decision-level integration with this legacy SWAP encoding:
a counts-only injectable runner, fixed-batch global budget, simultaneous
finite-look intervals, isolated program ownership, and an explicit undecided
outcome instead of an unconditional empirical argmin.

Well-separated candidates may save shots compared with fixed allocation.
Close candidates, conservative union bounds, circuit construction overhead,
and a small budget can remove that benefit. No universal shot, variance,
wall-time or quantum-speedup improvement is claimed. Deterministic counts
fixtures validate bookkeeping, not a physical sampling distribution or empirical
coverage. Software measurement backend examples are not hardware experiments.

## Run the public example and tests

The example explicitly constructs a `CPUQVM` software measurement backend and
returns only its measured counts. Backend selection stays outside the algorithm.
From the repository root, with the project dependencies and pytest installed:

```bash
env -u PYTHONHOME OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=pyqpanda-algorithm \
  python examples/qkmeans_adaptive_assignment.py

env -u PYTHONHOME OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=pyqpanda-algorithm python -m pytest -o addopts= -p no:cacheprovider \
  test/QAlgBase/test_qkmeans_adaptive_assignment.py
```

Sample counts and stopping times can vary. The example handles `undecided`
without replacing it with an empirical winner.

## Measured resource comparison

See [the public benchmark and its limits](../benchmarks/README_qkmeans.md) for
two actual runs of each of five fixtures, including close candidates, ties and
a non-identical winning centroid. The script takes an explicit counts runner.

Alpha must also remain strictly inside (0,1) after conversion to float. Positive
Real values that round to zero, or values below one that round to one, are
rejected before any runner call.
