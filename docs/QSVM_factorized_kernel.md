# Opt-in counts-only factorized QSVM kernel

`FactorizedQuantumKernel` evaluates the **existing, fixed ideal feature map**
with at most two qubits per submitted program. It is an additional API, not a
replacement for `QuantumKernel_vqnet`; the original implementation and trainer
are unchanged. It does not compute analytic fidelities, request probabilities
or states, create a QVM, or project a sampled matrix onto the PSD cone.

## API and backend boundary

```python
from pyqpanda_alg.QSVM import FactorizedQuantumKernel

# Supply your own measured backend adapter; there is no default backend.
kernel = FactorizedQuantumKernel(
    run_counts, shots=1024, batch_size=100, n_qbits=None
)
train_kernel = kernel.evaluate(train_features)             # (n_train, n_train)
test_kernel = kernel.evaluate(test_features, train_features)  # (n_test, n_train)
```

`run_counts(program, shots)` receives a pyqpanda3 `QProg`, with local qubits and
classical bits numbered from zero and final measurements already included.
It must initialize/reset to all-zero for each preparation and return a mapping
of **observed** binary strings to non-negative integer counts totaling exactly
`shots`. A one-qubit program uses keys `"0"`, `"1"`; a two-qubit program uses
`"00"`, `"01"`, `"10"`, `"11"`. The all-zero event is insensitive to string
endianness, but all bits must be present. Normalize provider-specific hex or
spaced results in the adapter, not by returning probabilities as counts.

The adapter owns device lifecycle, independent sampling and failures. Exceptions
from it propagate unchanged. Invalid counts (wrong width/characters, floats,
bools, negative counts, or incorrect total) raise `ValueError`. An absent zero
key means zero successes, **not** a small epsilon. All non-identity components
are measured, even if an earlier component has zero successes.

- `shots`: positive integer **per distinct non-identity component**.
- `batch_size`: positive integer limiting buffered matrix index pairs. It does
  not change shots, combine backend submissions, or limit the cache. Calls are
  sequential. Booleans and non-integral values are rejected for both parameters.
- `n_qbits`: optional expected feature dimension, integer >=2. If omitted it is
  inferred afresh per `evaluate` call, not fixed on first use. It does not set
  the local program width.
- `x_vec` and `y_vec`: finite real numeric array-likes converted to float64,
  shape `(n, d)`, `d >= 2`. A 1D input is one sample and still returns a 2D
  matrix. Their feature dimensions must agree. Complex, string, bool-only,
  object and ragged arrays are rejected. Gate angles overflowing float64 are
  also rejected before any execution. Very large finite angles remain subject
  to the backend's floating-point argument accuracy.
- `x_vec=None` is an error; `y_vec=None` means `x_vec`. Explicit `(0, d)` inputs
  return the appropriate empty matrix without execution. `[]` is rejected
  because it supplies no valid feature dimension.

The callable can be selected explicitly, without modifying any trainer:

```python
from sklearn.svm import SVC
classifier = SVC(kernel=kernel.evaluate)
classifier.fit(train_features, train_labels)
```

Training/prediction are separate evaluations and obtain fresh statistics.
Repeated calls are stochastic; exact symmetry is enforced only within a call
with equal datasets. In particular, independently sampled `evaluate(X, Y)` and
`evaluate(Y, X).T` need not agree shot-for-shot. No classification-accuracy or
PSD guarantee follows from using this callable with scikit-learn. Users who
need a fixed sampled training matrix may manage precomputed kernels themselves.

## Which map is supported, and why the product is correct

Each forward feature map consists of two layers. A layer appends, in time order:

1. `U2(q, 0, pi)` (Hadamard) then `U1(q, 2*x[q])` on each feature qubit;
2. `CNOT(0, 1)`, `U1(1, 2*(pi-x[0])*(pi-x[1]))`, `CNOT(0, 1)`.

Only qubits 0 and 1 are entangled. Its feature state therefore factors as

\[
|\phi(x)\rangle=|\phi_{01}(x_0,x_1)\rangle\otimes
\bigotimes_{j=2}^{d-1}|\phi_j(x_j)\rangle.
\]

The two-qubit core and each one-qubit component prepare the corresponding
`U(y)^dagger U(x)|0>` overlap and actually measure all local qubits. The builder
keeps the legacy overlap gate order, including its cancelled middle CNOT pair.
For a single component the time order is explicitly
`U2, U1(2x), U2, U1(2x), U1(-2y), U2, U1(-2y), U2`.
Reordering the encoding layers generally changes the answer.

With `p_c` the all-zero probability for component `c`, the **ideal** full-map
kernel is `K(x,y) = product_c p_c`. The estimator is

\[
\widehat K(x,y)=\prod_c\frac{N_{c,0}}{S},\qquad S=\texttt{shots}.
\]

For independent component experiments each factor is an unbiased Bernoulli
frequency, so their product is unbiased for this ideal kernel. This argument
requires independent draws **between component positions** as well as normal
within-run sampling. The library cannot verify the adapter's RNG/reset policy.
A runner that reseeds every call to the same stream can violate independence.

This is standard tensor-factorization of this particular map, not a new general
quantum-kernel algorithm or a first invention of analytic factorization. No
custom maps or automatic topology inference are supported. A map extended with
additional entangling edges needs a different decomposition; do not reuse this
implementation or its equivalence tests as a certificate for that map.

## Identity, stationarity, and cache semantics

The estimator supports **only the ideal fixed mapping**. An exactly equal pair
of component inputs (after float64 conversion, not `allclose`) contributes
exactly 1 and is not executed. This applies to diagonal entries, duplicate
vectors anywhere in square/rectangular matrices, and equal components of
otherwise different vectors. It does not replace unequal components with an
analytic answer.

That shortcut assumes exact cancellation of the same ideal unitary with its
inverse. A physical noisy echo need not return all-zero with probability 1.
There is no calibration procedure, noise mitigation or general noisy-kernel
mode here. Even removing this shortcut would not by itself establish equivalence
to a full noisy circuit: splitting and remapping qubits changes exposure to
noise, crosstalk, timing and device calibration. If attaching real hardware,
calibrate and assess ideal-map validity independently and label the observed
local estimator as such, not as the full device's kernel.

Within **one** `evaluate`, a dictionary reuses the observed frequency for
exactly the same ordered pair of component input tuples at the same component
position/structure. Core `(0, 1)` and individual positions `j >= 2` have separate
roles. In particular, two singleton positions with identical numbers are
measured separately: squaring one shared frequency would in general bias the
product. Reversed ordered pairs are not cache hits; Gram symmetry instead
computes one triangle and mirrors it.

Repeated component values across rows/columns can therefore reduce backend
calls. Fully continuous, all-distinct data may get little cache benefit beyond
symmetry and identities. Batching does not reset the cache. The dictionary is
local and discarded after the evaluation, including on error: there is no reuse
across new datasets, calibration changes or repeated calls. Calibration and
runner behavior must remain stationary **during** a call.

**Reused estimates correlate matrix elements.** A cache hit and a mirrored entry
are not new independent experiments or extra shots. Different ordered
experiments for a duplicated row can still fluctuate separately. Do not infer
matrix-entry independence or a larger effective sample size from the number of
entries. The sampled Gram matrix can have negative eigenvalues even though the
underlying ideal fidelity kernel is PSD; this API deliberately does not project,
clip eigenvalues, or silently alter training/test semantics.

## Resources and what can improve

Without identities or reuse, one input pair submits one width-2 program and
`d-2` width-1 programs, each with `S` shots: `(d-1)*S` separate preparations,
versus `S` preparations of the width-`d` full overlap. Thus **width <= 2 is not a
promise of fewer shots, lower variance at equal total budget, or faster wall
clock time**. At `d=2` there is no factorization width gain.

The current core has 26 unitary gates (6 CNOT, 12 U1, 8 U2), each singleton 8
(4 U1, 4 U2), plus two/one final measurements respectively. With no reuse, the
sum of unitary gate instances per one shot of every component equals the full
map's count; the executions are narrower, not fewer total gates. Different
backend costs and reset/dispatch overhead can dominate this tradeoff. The
matrix uses `O(n_x*n_y)` storage, and the per-call cache can grow with the number
of distinct component pairs; `batch_size` only bounds the index buffer.

A useful comparison records: local widths, submitted programs, requested and
observed shots, preparation/reset count, gate/measurement counts, full elapsed
time including construction/cache work, and error against a separately computed
small-system reference. Report equal-shot and equal-total-budget comparisons
separately. Local simulator samples are not hardware data, and neither local
nor hardware component samples automatically replace observations of a full
noisy circuit.

## Example and verification

[`examples/qsvm_factorized_counts.py`](examples/qsvm_factorized_counts.py) accepts
an external adapter and reports per-evaluation resources. There is intentionally
no hidden CPUQVM/analytic fallback. With the package and an adapter available:

```sh
python docs/examples/qsvm_factorized_counts.py --runner my_backend:run_counts --shots 1024
```

The adapter module supplies the same two-argument function described above.
You can also import `run_example` and call `run_example(your_run_counts, shots)`;
the returned JSON-compatible record contains the actual matrices and calls.

Tests use a small independent forward-map-plus-dagger CPUQVM probability oracle
**only in tests**, checked against the unchanged legacy overlap for `d=2,3,6`.
They inspect nontrivial core/singleton gate order, execute actual CPUQVM shot
measurements, and check cache scope/roles, repeated data, symmetry/rectangular
calls, malformed inputs/counts, exceptions, no implicit simulator, and a
non-PSD sampled example. Scripted counts in protocol tests are explicitly test
doubles, not measurements. External device validation is a separate activity.

From the repository root, with its package directory on `PYTHONPATH`:

```sh
PYTHONPATH=pyqpanda-algorithm python -m pytest -o addopts='' test/QSVM/test_factorized_kernel.py test/QSVM/Test_qsvm.py
```

## Counts-only software-backend validation

The production algorithm was also executed on an external quantum-simulation
process through its classical preparation/gate/measurement interface, using
a PyQPanda program adapter. Every shot was actually
prepared and measured; no statevector, exact probability or forced branch was
provided to the estimator. CPUQVM in unit tests is an independent reference,
not this benchmark execution path. This is classical simulation, not QPU data.

Fixture: the four-row, four-feature repeated-component dataset in
`benchmarks/benchmark_qsvm_factorized.py`. Two separate runs used 256 shots
per submitted program, no PSD projection, and freshly sampled components:

- Full overlaps: six width-4 programs, 1,536 preparations, 89,088 compiled gate executions.
- Factorized evaluator: three width-1/2 programs, 768 preparations, 20,480 compiled gate executions.
- Logged runner wall times (seconds): full 25.252, 25.004; new 6.154, 5.790.
- Sampled device peak RSS: full 60.84–60.95 MiB; new 60.01–60.02 MiB. This includes the Python/library process baseline and is a sampled, not certified, peak.
- Maximum independently measured kernel-entry differences: 0.04296875, 0.05078125.

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

**The savings depend on repeated components in this dataset.** All-distinct
inputs can need more preparations with factorization; this does not establish
universal throughput, lower variance at equal budget, or better classification
accuracy. Full and factorized sampled matrices have different correlations.

To run the same public fixture, configure a counts adapter and use
`--runner MODULE:FUNCTION --shots 256` with the benchmark script. No backend
is silently selected; an adapter error or exhausted budget remains an error.
