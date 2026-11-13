# Full-statevector variational linear solver

`VQLS` minimizes the global objective

`1 - |<b_normalized, A x(theta)>|^2 / ||A x(theta)||^2`

for a normalized variational state. It accepts finite, non-singular, square
complex matrices with power-of-two dimension, including non-Hermitian matrices.
Numerical rank is checked after removing the matrix's overall scale; a system
that is numerically rank deficient is rejected. This does not improve the
conditioning of an ill-conditioned system.

The default RY/CNOT ansatz is simulated with NumPy and has real amplitudes.
Complex solution directions require a suitable custom `state_fn`. This callable
receives real parameters and returns a complete complex statevector. A native
PyQPanda3 statevector simulator can supply it. Measurement counts alone cannot
provide this input, because they omit relative phases. This implementation
does not provide a shot-based cost estimator or claim a quantum speedup.

`solve()` uses SPSA: two objective calls estimate the gradient, and a third
evaluates the updated parameters. A fixed seed reproduces the initialization
and perturbations. `converged` refers to the objective tolerance, not a bound on
classical solution error; inspect `residual_norm` and system conditioning.

`solution(parameters)` restores the least-squares-optimal complex scale of the
variational state. Its returned solution is classical, not merely normalized.
It removes matrix and right-hand-side scales before taking inner products and
uses binary exponents to reconstruct the final scale. Multiplying both `A`
and `b` by the same nonzero real scalar therefore preserves the objective and
solution up to floating-point roundoff, including small and large scales that
would underflow or overflow a naive squared norm.

Inputs are copied at construction; the public `matrix` and `vector` snapshots
are read-only. To solve a changed system, construct another solver. Custom
statevectors are normalized safely even when their finite components have a
very small or large common scale.

The solution must be representable in complex128; otherwise reconstruction
raises `ValueError`. Subnormal solution components may round to zero. The
residual describes the returned vector, including that rounding. If the norm
of the residual itself exceeds float64 range, it is returned as infinity.

For the existing example

```python
import numpy as np
from pyqpanda_alg.VQLS import VQLS

A = np.array([[1.0, -1.0 / 3.0], [-1.0 / 3.0, 1.0]])
b = np.array([1.0, 0.0])
result = VQLS(A, b, layers=2, max_iterations=400, tolerance=1e-7,
              learning_rate=0.35, perturbation=0.15, seed=7).solve()
print(result.solution, result.cost, result.residual_norm)
```

the exact classical solution is `[1.125, 0.375]`. The variational result at the
configured tolerance is approximate. Scaling both inputs by `1e-200` or
`1e200` leaves the same convergence problem; residual norms scale with `b`.

This module is preserved from tokenjunkielabs PR #79, whose VQLS commits are
authored by woahwhattheheck, with numerical corrections. It is not a replacement
for mnn31's earlier PR #28: that contribution constructs native RY/CZ circuits,
uses Hermitian Pauli decomposition and SciPy optimization, and returns a
normalized state. Reconcile these different APIs with the original author and
maintainers before choosing a single public implementation.
