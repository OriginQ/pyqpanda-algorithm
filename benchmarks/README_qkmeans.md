# Adaptive-assignment measurement benchmark

Run `benchmark_qkmeans_assignment.py --runner MODULE:FUNCTION` with an explicit
measurement-count adapter. The public script uses the same elementary SWAP-test
circuit for both allocation policies. It does not create a simulator implicitly,
read an internal state or convert analytic probabilities into fake counts.

## Two actual software-measurement runs

The recorded runs used an external device process through preparation, gate and
single-bit measurement requests. Every shot prepared and evolved a new register.
The algorithm could not request a statevector, exact probability or forced
outcome. An independent decoder reconstructed every reported count from the
classical wire records. This is software simulation, not a QPU experiment.

Uniform allocation used **512 shots per candidate**. This is a budget-matched
control, not the legacy helper's default 1,024 shots per candidate. Adaptive
allocation used batches of 64, the same **total** cap as that uniform control,
and alpha=0.05 per assignment. There is no common error guarantee across all
points just because each uses alpha=0.05.

- **Separated case:** uniform 1,536 shots; adaptive 256 in each repeat. Both
  adaptive runs returned candidate 1 (zero-based), consistent with the public
  nominal circuit prediction. Uniform wall times 7.149/7.089 seconds; adaptive
  2.016/2.033 seconds.
- **Interior-input case:** uniform 1,536 shots; adaptive 832/960 shots. Both
  returned candidate 1. Uniform wall times 8.832/8.441 seconds; adaptive
  5.292/5.612 seconds.
- **Non-identical winning centroid:** uniform 1,536 shots; adaptive 768/960.
  Neither the point nor any candidate was identical. Both adaptive runs
  returned candidate 1; its ideal ancilla-one probability is nonzero.
- **Close candidates:** both policies spent 1,536 shots. Adaptive returned
  `undecided` in both repeats, with only candidates 0 and 1 still active. The
  far candidate stopped receiving measurements after its first 64-shot batch.
- **Identical candidates:** both spent 1,024 shots. Adaptive returned
  `undecided` in both repeats, not a claimed unique winner.

All inputs, per-run resource figures and outcomes appear in
[qkmeans_counts_observed.json](qkmeans_counts_observed.json).
Wall times include device startup, protocol exchanges and wire logging. Peak
RSS was sampled and includes interpreter/library overhead; it is not a strict
memory upper bound. The public benchmark instead times circuit construction,
callback execution and aggregation, with startup managed by the supplied
callback. Do not mix the timers as if they measured identical boundaries.

## What these results do and do not establish

The first two fixtures were intentionally chosen to exhibit separable winners;
they are not an unbiased performance survey. Their winners coincide with the
point, giving an easy zero-distance match. There is no special-case identity
shortcut in the implementation: those probabilities were measured. The other
fixtures retain the difficult cases where the budget does not settle a label.

This demonstrates useful early stopping on selected examples and honest
abstention on ambiguous examples. It does **not** estimate empirical confidence
coverage, prove every dataset saves shots, establish better clustering accuracy,
or predict cloud billing savings. The simultaneous probability statement is
proved separately under the stated stationary independent-shot assumptions.
A uniform empirical argmin and a separated adaptive decision offer different
statistical contracts; the comparison does not claim equal-error optimality.

The original angle mapping is preserved, so minimization concerns its
ancilla-one probability, not an arbitrary Euclidean metric. CPUQVM was used
separately as an independent small-circuit reference, including checking the
Fredkin decomposition on every three-qubit basis vector. It did not supply
answers to the measured adaptive decisions.

The resource panels used frozen input-package snapshots. Final changes clarified
the probability-bound docstring and rejected extreme Real alpha inputs that
round to float zero or one. Every retained measurement transcript was replayed
through the final policy with exact public-program and batch-size checks; all
decisions and accounting were identical. This replay is classical verification,
not additional quantum measurements. The final public benchmark entry point was
separately exercised with actual 16-shot batches; that smoke test is not added
to these resource figures or represented as an empirical coverage study.

## Existing methods

Hoeffding intervals, union bounds and successive candidate elimination are
standard tools. The contribution is this repository-specific assignment policy,
explicit finite budget and unresolved-result API, not a claim to have invented
adaptive quantum measurements.

- W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*,
  JASA 58 (1963), 13–30, [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
- [An example of existing adaptive quantum-shot allocation](https://aws.amazon.com/blogs/quantum-computing/getting-the-most-from-your-quantum-measurements-adaptive-shot-allocation-on-amazon-braket/)
  illustrates why the broad idea is not claimed as new here; its expectation
  estimation objective is not this centroid-decision API.
