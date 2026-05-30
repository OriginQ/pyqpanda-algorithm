# CCF2026 Professional Track: MNIST Binary Classification

## Constraint Check

- Qubits: `8`
- Ansatz layers: `2`
- Circuit parameters: `32`
- Classical head parameters: `9`
- Total trainable parameters: `41 <= 100`
- Evaluation noise probability: `0.04`

## Data

A deterministic 8x8 MNIST-like 3-vs-8 binary dataset is generated locally. The full pipeline can be swapped to official MNIST arrays by replacing `make_mnist_like_binary_dataset` while keeping the same compressed 8-feature contract.

| split | samples | class_0_digit3 | class_1_digit8 |
| --- | --- | --- | --- |
| train | 72 | 36 | 36 |
| val | 24 | 12 | 12 |
| test | 24 | 12 | 12 |

## Test Metrics

| model | accuracy | f1 | auc | log_loss |
| --- | --- | --- | --- | --- |
| logistic_8feature | 0.791667 | 0.761905 | 0.923611 | 0.373309 |
| vqc_clean | 0.625 | 0.666667 | 0.75 | 0.668958 |
| vqc_noise_aware | 0.625 | 0.64 | 0.583333 | 0.693091 |

## Noise Robustness

| model | noise_prob | accuracy | f1 | auc |
| --- | --- | --- | --- | --- |
| vqc_clean | 0.0 | 0.625 | 0.666667 | 0.75 |
| vqc_clean | 0.02 | 0.625 | 0.666667 | 0.611111 |
| vqc_clean | 0.04 | 0.5 | 0.5 | 0.444444 |
| vqc_clean | 0.08 | 0.291667 | 0.26087 | 0.201389 |
| vqc_noise_aware | 0.0 | 0.666667 | 0.692308 | 0.777778 |
| vqc_noise_aware | 0.02 | 0.583333 | 0.615385 | 0.763889 |
| vqc_noise_aware | 0.04 | 0.583333 | 0.615385 | 0.618056 |
| vqc_noise_aware | 0.08 | 0.333333 | 0.384615 | 0.4375 |

## Quantum Resources

| model | qubits | depth | total_params | 1q | 2q | shots | backend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_8feature | 0 | 0 | 9 | 0 | 0 | 0 | classical |
| vqc_clean | 8 | 28 | 41 | 56 | 21 | 256 | local_statevector |
| vqc_noise_aware | 8 | 28 | 41 | 56 | 21 | 256 | local_statevector_noise_aware |

## Why Noise-Aware Training

The clean VQC is optimized for ideal statevector features. The noise-aware VQC injects angle jitter plus readout/gate attenuation during training, so its head and variational parameters see the same distribution shift that appears during noisy evaluation. This is intentionally small enough for the 8-qubit/100-parameter professional-track constraint.
