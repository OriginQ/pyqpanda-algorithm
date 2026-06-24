# QReupload — Data Re-uploading Variational Quantum Learning

`QReupload` adds the first ML estimator to `pyqpanda-algorithm` whose **circuit
parameters are trained against a supervised task loss** (sklearn-style `fit`).
`QSVM`/`QSVR` fit only a classical SVM over a *parameter-free* quantum feature
map; `QReupload` is a **trainable** data re-uploading circuit with a learnable
input-scaling factor, an optional entangling ring, exact adjoint-gradient
training, and two diagnostics that make the model's behaviour inspectable.

## Why

A data re-uploading model interleaves data-encoding rotations with trainable
single-qubit blocks. With one feature per qubit and `n_layers` re-uploads it
realises a **truncated Fourier series of controllable degree** in the inputs
(Schuld, Sweke & Meyer, *PRA* 103, 032430, 2021). A trainable input-scaling
factor lets the model align its accessible frequencies to the data; an
entangling ring lets it represent multi-feature **cross-frequency** terms that a
sum of per-qubit readouts cannot.

## Components

| object | purpose |
|---|---|
| `QReuploadRegressor` | sklearn-style `fit / predict / score` regressor |
| `QReuploadClassifier` | binary classifier (`predict_proba`, sigmoid + BCE) |
| `fourier_spectrum(model)` | recover which Fourier harmonics a fitted 1-feature model learned |
| `trainability_scan(qubit_range)` | barren-plateau diagnostic: `Var[∂⟨Z⟩/∂θ]` vs qubit count |

Training uses `pyqpanda3.vqcircuit.VQCircuit` with `DiffMethod.ADJOINT_DIFF`
(exact analytic gradients), optimised with a NumPy Adam + cosine-annealed LR.

## Usage

```python
import numpy as np
from pyqpanda_alg.QReupload import QReuploadRegressor, fourier_spectrum

X = np.linspace(-np.pi, np.pi, 80)
y = np.sin(2 * X) + 0.5 * np.cos(3 * X)

reg = QReuploadRegressor(n_layers=5).fit(X, y)
print(reg.score(X, y))                 # ~0.9999

k, mag = fourier_spectrum(reg)         # peaks at k = 2 and 3 — the target frequencies
```

See `example/QAlgBase/testeg_qreupload.py` for a full demo (spectrum readout,
entanglement ablation on a cross-frequency target, barren-plateau scan), each
shown next to a cross-validated classical Fourier-ridge baseline.

## Design notes (verified on pyqpanda3 0.3.5)

- **Data enters as a constant**, not a second parameter: `Param(λ) * float(x)`.
  A product of two placeholders (`Param * Param`) returns a *zero* adjoint
  gradient, so each sample is baked into its own circuit.
- **Input-scaling factors are initialised near 1.0**; initialising them near 0
  collapses the encoding to identity and stalls training.
- **Readout uses local observables** `⟨Z_q⟩`; the gradient variance still decays
  roughly exponentially in qubit count (run `trainability_scan` to measure the
  rate for your config — e.g. ~`exp(-0.8 · n)` for the default depth-3 probe at
  `seed=0`), so scale qubits deliberately, not blindly.

## Honesty

On band-limited data a Fourier-aware, properly-regularised classical model is a
strong baseline; the example reports it side by side rather than hiding it. The
value of `QReupload` here is the *trainable encoding*, the *readable spectrum*,
and the *built-in trainability warning* — not beating that baseline on raw error.
