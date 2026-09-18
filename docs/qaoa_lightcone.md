# QAOA local expectation sampling

`QAOALightCone` is an **opt-in, sampling-only mean-objective executor**. It
constructs a smaller circuit for each nonzero nonconstant Ising term by tracing
its backward light cone. It does not change `QAOA`, choose an optimizer, or
instantiate a backend. Backward light cones are a standard circuit-locality
technique, not a new QAOA ansatz.

## Supported circuit and API

```python
from pyqpanda_alg.QAOA import QAOALightCone

executor = QAOALightCone(
    n_qubits=12,
    terms={(): 0.5, (1,): -0.3, (1, 4): 0.7, (4, 9): -0.2},
    depth=1,
)
jobs = executor.build_programs(gammas=[0.27], betas=[-0.18])
for job in jobs:
    print(job.plan.target, job.plan.global_qubits, job.parity_cbits)
    print(job.program.originir())
```

The Hamiltonian is

\[
H_C = c + \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j.
\]

Starting from `|+>^n`, each forward layer executes `exp(-i gamma[l] H_C)`
then `RX(-2 beta[l])` on every qubit. In particular, the mixer sign matches the
**actual default implementation** in `qaoa.py`: it implements
`exp(+i beta[l] sum_i X_i)`. The local cost gates are `RZ(2 gamma h)` and
`CNOT(i,j); RZ(j, 2 gamma J); CNOT(i,j)`. The constant is an energy offset,
not a gate. Angles are in radians; the two parameter sequences are in forward
layer order and must each have length `depth`.

### `QAOALightCone(n_qubits, terms, depth)`

- `n_qubits` and `depth` are nonnegative integers, excluding booleans.
- `terms` is a mapping from tuples to real coefficients: `()` for a constant,
  `(i,)` for a field, `(i,j)` for an interaction. Explicit `n_qubits` avoids
  confusing the number of used indices with the register width. Sparse labels
  and disconnected components are supported; every index must be in
  `range(n_qubits)`.
- A pair may be supplied in either order. Repeated indices, keys of length
  greater than two, and two keys that canonicalize to the same pair are errors.
  A Python mapping cannot preserve overwritten identical keys: callers must
  combine repeated contributions before constructing it.
- Coefficients and angles must be `numbers.Real` scalars convertible to finite
  floats. Complex values (even `1+0j`), booleans, strings, NaN and infinity are
  rejected. Ordinary NumPy integer/real scalars and fractions are supported.
  Zero coefficients are validated and then omitted. Conversion uses floating
  point; arbitrary-precision or symbolic arithmetic is not provided.
- Weights are copied. `executor.terms`, `executor.plans`, and their nested
  records are immutable tuples/frozen records. Later input mutations have no
  effect. Changing weights or depth requires constructing a new executor.
- `depth=0` is supported: each nonconstant target is sampled from a local
  Hadamard preparation. It is **not** replaced with its analytic zero mean.
  A zero or constant-only Hamiltonian has no programs, and `n_qubits=0` is
  allowed for such a Hamiltonian.

### `build_programs(gammas, betas) -> tuple[LightConeProgram, ...]`

There is one program per nonzero nonconstant term, sorted lexicographically by
canonical global support. Targets are not merged, including when their cones
have the same width. Each returned record contains:

| Field / property | Meaning |
| --- | --- |
| `program` | A fresh, measured local `QProg`, not a full-register program |
| `plan.target.qubits`, `.coefficient` | Global target support and copied weight |
| `plan.global_qubits` | Sorted global labels; entry `i` corresponds to local qubit `i` |
| `plan.global_to_local` | Read-only inverse mapping |
| `plan.layers` | Forward-order layers, each with global `cost_terms` and `mixer_qubits` |
| `local_target` | Target support in local qubit indices |
| `parity_cbits` | Cbit indices whose parity gives this target's measurement |
| `measurement_cbits` | All local cbit indices, in increasing order |

Each `QProg` is mutable and owned by the caller. Modifying it does not change
plans or later builds, but invalidates its stated expectation if the gates are
changed. Parameters and all emitted rotation angles are validated before any
runner is called. Calling `build_programs` does not execute anything.

### `estimate(gammas, betas, shots, run_counts) -> LightConeEstimate`

Supply a synchronous callback with exactly the following contract:

```python
# device is explicitly supplied/configured by the application.
# This adapter is suitable only if that device has the displayed API and
# returns counts in the required cbit order. Other APIs need their own adapter.
def run_counts(program, shots):
    device.run(program, shots=shots)
    return device.result().get_counts()

result = executor.estimate([0.27], [-0.18], shots=256, run_counts=run_counts)
print(result.mean, result.total_shots)
```

There is no implicit CPU simulator, statevector, exact-probability path or
fallback. The runner must freshly prepare and sample each program; it is
responsible for resetting the backend and preserving the intended operations.
The executor sees only `runner(program, shots) -> counts`.

**Measurement order is strict:** for a local width `w`, every local qubit `i`
is measured into `c[i]`. Counts keys must be exactly `w` binary characters in
order `c[w-1]...c[1]c[0]`; the **rightmost character is c0**. With global cone
`(7,90)`, key `"01"` means global qubit 7 measured 1 and global qubit 90 measured
0. It gives `Z7=-1`, `Z90=+1`, and `Z7 Z90=-1`.

Counts must be a mapping to nonnegative integer multiplicities (booleans are
not integers here) with total **exactly `shots`**. Zero-frequency outcomes may
be omitted. No automatic reversal, integer-key conversion, padding, whitespace
stripping or probability-to-count conversion is performed. If a provider uses
a different format, adapt it explicitly at the runner boundary. Probabilities,
quasi-counts and normalized frequencies are rejected.

`shots` is a positive integer **per program**, not a global shot budget.
The frozen result has:

- `mean`: constant plus the weighted sum of sampled parity means;
- `term_means`: unweighted means, in `executor.plans` order;
- `shots_per_program`: the requested positive integer;
- `total_shots`: `len(executor.plans) * shots`, counting circuit preparations.

A constant-only objective still requires a callable and valid parameters, but
makes no runner calls and returns `total_shots=0`. Callback exceptions and bad
counts propagate; there is no retry or partial result. A failure on a later
program can occur after earlier programs have already consumed shots.

Every `estimate` call builds and runs fresh programs, **even for repeated
parameters**. Only the immutable structural plans are reused; no expectations,
counts or parameter-bound circuits are cached.

## Why the local circuits are sufficient

Take a target supported on `S`. Traverse the layers backwards:

1. A product of single-qubit mixers can change operators on `S`, but cannot
   enlarge support. Retain mixers on `S` only.
2. At the following backward cost step, select all nonzero Z/ZZ terms that
   touch the **entering** `S`. All other cost factors act disjointly from the
   current observable and commute with it. Since the entire cost layer
   commutes internally, those factors cancel from the conjugation even when
   they share a newly reached qubit with a retained factor.
3. Only **after selection**, enlarge `S` by the supports of the selected
   factors. Carry that new set to the preceding layer.

For a chain `0--1--2--3` and target `Z0`, a depth-one plan may retain `(0,1)`,
but not `(1,2)`. Serial expansion inside that same commuting cost layer would
incorrectly treat the entire connected chain as one backward step. At depth
two, `(1,2)` can enter through the preceding layer. Fields on a newly reached
boundary qubit are likewise not added until an earlier layer reaches them.
The stored layers are reversed again for forward execution.

By induction, the full and retained circuits conjugate every observable on
the target identically. At the input boundary the conjugated observable acts
only on the cone. The full initial state is a product of `|+>` states, so
tracing out its complement leaves **`|+>` on every cone qubit**, not a maximally
mixed state and not a boundary qubit fixed to zero. This proves equality of
the target measurement marginal and hence of its parity mean. Preparing the
local `|+>` state and executing the retained gates is therefore sufficient.
This argument does not require access to a backend's state.

Only the **target marginal** is promised. Other measured cone qubits need not
have their full-circuit marginal, and separate target samples cannot be
assembled into samples of the global output distribution. The construction
is restricted to the stated product initial state, commuting Z/ZZ cost and
single-qubit X mixer. Custom initial states, XY mixers, general Pauli costs,
CVaR and Gibbs losses are not supported.

## Resources and finite shots

On sparse bounded-degree graphs at low depth, a term's cone can be much
smaller than the full register. Qubit labels are remapped to a compact local
register, and only retained gates are emitted. Plans are formed once using
term incidence lists; angle changes reuse these dependencies. `job.program`
can be inspected for actual width and gates before selecting a device.

This trades **smaller individual preparations** for **multiple preparations**.
For `K` nonconstant terms and `N` shots per program, the executor consumes
`K*N` shots, versus `N` shots for one full circuit at the same per-program
setting. Overlapping cones repeat work. At high depth or on dense graphs,
cones may cover most or all qubits; construction, storage, submission and gate
costs can outweigh the benefit. There is no promise of fewer total shots or a
runtime speedup.

For ideal execution with fresh unbiased samples, each parity estimator is
unbiased and so is the weighted mean. If shots and programs are independent,
its variance is

\[
\operatorname{Var}(\widehat E)
= \sum_t a_t^2 (1-\mu_t^2)/N
\leq \sum_t a_t^2/N.
\]

Full-circuit sampling instead shares shots among terms and has covariances;
its accuracy at a fixed total preparation budget cannot be compared by width
alone. Independence is an assumption on the runner, not something counts
validation can establish. The executor does not infer confidence intervals.
Noisy devices, crosstalk, backend reuse without reset and correlated initial
states can invalidate ideal marginal equivalence: deleting gates can change
noise as well as width. This is not a general noise-preserving compilation.

## Example and tests

See [the small executable example](../examples/qaoa_lightcone.py). Without a
runner argument it prints a plan/resource preview and does not execute a
backend:

```sh
PYTHONPATH=pyqpanda-algorithm python examples/qaoa_lightcone.py
PYTHONPATH=pyqpanda-algorithm python examples/qaoa_lightcone.py \
    --runner my_device_adapter:run_counts --shots 256
```

`my_device_adapter` is an application-provided Python module; no backend is
bundled with the example. Its function must obey the counts contract above.
The example uses the same plans for two different angle settings and reports
the shot cost of each separately.

Run the bounded tests and existing QAOA regressions from the repository root:

```sh
PYTHONPATH=pyqpanda-algorithm OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 python -m pytest -o addopts='' test/QAOA
```

The new tests use **CPUQVM only as a test reference**, including tiny exact
full-circuit/target-marginal comparisons and an explicit counts-runner adapter.
Exact probabilities are never passed into the estimator. Other tests inject
known counts to check parsing, parity, freshness and resource accounting;
these are interface tests, not evidence of backend execution performance.

## Counts-only software-backend validation

The production algorithm was also executed on an external quantum-simulation
process through its classical preparation/gate/measurement interface, using
a PyQPanda program adapter. Every shot was actually
prepared and measured; no statevector, exact probability or forced branch was
provided to the estimator. CPUQVM in unit tests is an independent reference,
not this benchmark execution path. This is classical simulation, not QPU data.

Fixture: the eight-qubit weighted ring in
`benchmarks/benchmark_qaoa_lightcone.py`, depth 1, gamma=0.31, beta=0.23.
Two separate runs used 256 shots per submitted program:

- Full circuit: one width-8 program, 256 preparations, 14,336 compiled gate executions.
- Local executor: eight width-4 programs, 2,048 preparations, 38,912 compiled gate executions.
- Logged runner wall times (seconds): full 23.480, 22.663; new 11.849, 12.025.
- Sampled device peak RSS: full 68.30–68.42 MiB; new 60.62–60.62 MiB. This includes the Python/library process baseline and is a sampled, not certified, peak.
- Maximum independently measured term-mean differences: 0.17968750, 0.17187500.

The raw classical records were reconstructed independently and matched the
reported counts. Finite-shot differences were checked with conservative
simultaneous Hoeffding bounds, not asserted equal or counted as exact circuit
proofs. Backend workers were pinned to one CPU with single-thread numerical
libraries. Times include device startup, public protocol, fresh preparation and
measurement, and wire logging; they are not isolated C++ kernel timings. The
public benchmark script separately times each algorithm's circuit construction,
execution and aggregation (including cold planning for QAOA); dataset/fixture
setup is excluded. Its runner may manage startup differently. These fields are
not the same timer as the logged operator harness totals above.

**This is not a shot-saving result:** one full-register sample yields all Z
terms jointly, whereas the local version here separately samples each term.
The timing improvement is specific to this classical graph backend and fixture,
not a quantum speedup or a same-total-shot variance comparison.

To run the same public fixture, configure a counts adapter and use
`--runner MODULE:FUNCTION --shots 256` with the benchmark script. No backend
is silently selected; an adapter error or exhausted budget remains an error.

### Extreme weighted sums

The final objective normally uses `math.fsum`. If its intermediates overflow,
only the already-rounded finite weighted sample means are summed exactly as
binary rationals and converted back to float. This preserves finite cancellation
without rescaling away a tiny residual. A genuinely unrepresentable objective
raises `ValueError` after sampling; the consumed shots cannot be recovered.
This classical reduction fallback does not inspect or change the quantum state.
