# CCF2026 Professional Track: 8-Qubit Noise-Robust MNIST Binary Classifier

This entry targets the 2026 CCF Quantum Computing Programming Challenge, OriginQ Cup,
professional-track MNIST binary-classification task described as:

- MNIST binary classification
- noise robustness
- 8 qubits
- no more than 100 trainable parameters

The implementation is dependency-free Python so reviewers can run it immediately on a
plain Python 3.11 environment. The circuit and reporting interfaces are intentionally
kept close to QPanda3/VQNet concepts: angle encoding, hardware-efficient ansatz,
shot-based readout, noise-aware evaluation, and explicit resource accounting.

## Method

The model compares three systems on the same train/validation/test split:

| model | purpose |
| --- | --- |
| `logistic_8feature` | classical baseline on the same 8 compressed features |
| `vqc_clean` | 8-qubit variational circuit trained without noise |
| `vqc_noise_aware` | same 8-qubit circuit trained with stochastic angle/readout noise |

The quantum model uses:

- 8 qubits
- 2 ansatz layers
- 32 trainable circuit parameters
- 9 classical-head parameters
- 41 total trainable parameters

This stays comfortably below the 100-parameter limit.

## Quick Start

From this directory:

```powershell
py -3.11 scripts\run_quick_demo.py
py -3.11 -m unittest discover -s tests
```

From the repository root:

```powershell
py -3.11 contest2\CCF2026_Professional_MNIST\scripts\run_quick_demo.py
```

Outputs are written to `results/quick_demo/`:

- `metrics.csv`
- `resource_table.csv`
- `robustness.csv`
- `config.json`
- `report.md`
- `figures/test_accuracy.svg`
- `figures/noise_robustness.svg`

## Package

Create an upload-ready zip:

```powershell
py -3.11 scripts\package_submission.py
```

The zip is written to `dist/CCF2026_Professional_MNIST_submission.zip`.

## Notes for Full QPanda3/VQNet Migration

The default backend is a local statevector simulator because the current review
environment may not include QPanda3 or VQNet. To migrate:

1. Replace `ccf2026_mnist_qml/circuit.py` gate calls with QPanda3 circuit builders.
2. Wrap the parameterized circuit as a VQNet quantum layer.
3. Keep `metrics.csv`, `resource_table.csv`, and `robustness.csv` schemas unchanged.
4. Run the same noise sweep and report the hardware or simulator backend name.

