# IQAE sampling and phase-interval regression notes

`IQAE.run()` estimates a different Bernoulli probability at each Grover power
`k`. Its empirical success fraction and Chernoff interval must therefore use
the same samples: all consecutive batches at the current power. This is the
pooling rule described in Algorithm 1 of [Grinko et al., Iterative quantum
amplitude estimation](https://arxiv.org/html/1912.05559v3).

The implementation now retains the first batch when the power changes and
uses the current power's accumulated shot count for its confidence interval.
`n_sum` reports all measurement shots in the current call and resets on every
`run()`. It does not count Grover oracle applications. The existing per-batch
shot cap remains; it does not truncate the sample size of pooled observations.

For example, six batches of 33 shots at powers `[0, 0, 1, 1, 3, 3]` must use
interval sample sizes `[33, 66, 33, 66, 33, 66]`. The previous implementation
used `[33, 66, 99, 132, 165, 198]`, including measurements of other probabilities.
It also dropped the first batch after switching powers when pooling later
batches. Reusing an estimator could include shots from its previous run.

Corrected sample sizes can widen an interval to a full monotonic half-circle.
The inverse phase mapping uses one period for both bounds, located using the
old interval's midpoint. This handles two endpoint cases:

- In the lower half-circle, a full-period upper endpoint must stay in the
  current period, rather than being used to select the following period.
- In the upper half-circle, `(13 / 90) * 90` rounds to `12.999999999999998`.
  Using that endpoint to select the period would incorrectly select 12.

The public method signatures, power-selection heuristic, shot scheduling,
and existing 20-iteration limit are unchanged. This change fixes the sample
bookkeeping and phase mapping; it does not establish a new confidence-coverage
or convergence guarantee for every input. Correct sample accounting can change
the number of shots and the returned estimate.

## Validation

Tested with Python 3.12.14 and pyqpanda3 0.4.1 on Windows. The 18 regression
cases cover pooled statistics, power changes, repeated runs, both phase-boundary
cases, deterministic adaptive estimates at seven probabilities, and CPUQVM
preparations of the zero/one endpoint states. The deterministic count oracle
uses `round(n * sin((2*k + 1) * asin(sqrt(p))) ** 2)`.
These tests do not measure random confidence-interval coverage.

On upstream commit `5f973efccb84bc193157d1ccebe32137e307293b`, the regression
file has 9 failures and 9 passes. After the fix, all 36 repository tests pass
(18 new cases and 18 existing cases).

From the repository root, with dependencies installed:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'pyqpanda-algorithm'
$env:MPLBACKEND = 'Agg'
python -m pytest -o addopts= -q test/QAlgBase/Test_IQAE_sample_accounting.py
python -m pytest -o addopts= -q test
```

`-o addopts=` overrides the repository's optional Allure reporting arguments.
