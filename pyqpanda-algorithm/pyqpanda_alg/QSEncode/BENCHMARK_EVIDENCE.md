# QSEncode-Insight benchmark evidence

## Confirmatory design

The final generalization benchmark was preregistered before its Locked Test was
run. Its primary data comprised 60 independent distribution-design instances
across five equally weighted families, each evaluated at
`N={8,16,32,64}` and in Walsh and Fourier modes: 480 evaluation cells.

The primary product metric was `selector-all`: `do_not_compress` cells remained
in the analysis with zero resource gain. Walsh and Fourier were evaluated
separately; the benchmark did not test an automatic cross-basis selector.

## Locked Test result

All 480 cells completed under the frozen environment and analysis protocol.

A presentation-oriented view of the same frozen results, including dimension,
family, refusal, and attribution breakdowns, is available in
[BENCHMARK_VISUAL_SUMMARY.md](BENCHMARK_VISUAL_SUMMARY.md). It is descriptive
only and does not replace the preregistered aggregation below.

| Basis | Compiled 2q selector-all | Compiled depth selector-all | Gate |
|---|---:|---:|---|
| Walsh | 45.27% | 48.78% | strong pass |
| Fourier | 71.11% | 69.82% | strong pass |

There were 55 `do_not_compress` decisions among the 480 cells (11.46%). They
were not removed from the main statistic.

The preregistered hierarchical point estimate determined the gate. Cluster
bootstrap intervals described stability and did not replace the gate rule.

## Interpretation

The mechanisms differed by basis:

- Walsh two-qubit savings came predominantly from preparation-strategy choice;
- Fourier savings combined fidelity-budget truncation with an additional
  preparation-strategy contribution.

The Dirichlet family was much weaker than Gaussian, bimodal, exponential, and
step families. This negative evidence is retained: QSEncode-Insight diagnoses
whether an input is worth compressing rather than claiming that every
distribution is compressible.

Candidate incompatibility was also common, particularly for DS preparation.
The final recommendation nevertheless required a compatible, correctness-
checked candidate and five successful compiled attempts. Capability filtering
and explicit refusal are therefore core behavior, not cosmetic reporting.

## Claim boundary

Within the preregistered N<=64 test scope, Walsh and Fourier modes both met the
predefined compiled-resource gate. These results do not establish quantum
advantage, hardware runtime acceleration, behavior at larger dimensions, or an
automatic best-basis selector.
