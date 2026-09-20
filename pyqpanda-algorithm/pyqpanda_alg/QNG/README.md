# QNG: Quantum Natural Gradient

QNG equips variational algorithms with geometry-aware updates instead of plain
Euclidean gradient descent.

For a normalized pure state |psi(theta)>, the module constructs the
Fubini-Study metric

~~~text
g_ij = Re(<d_i psi|d_j psi>
          - <d_i psi|psi><psi|d_j psi>)
~~~

and solves the regularized system

~~~text
(g + lambda I) v = grad C
theta_next = theta - learning_rate * v
~~~

The projection term makes the metric insensitive to global phase.

## Minimal example

~~~python
import numpy as np
from pyqpanda_alg.QNG import natural_gradient_step

theta = np.array([0.4])
state = np.array([
    np.cos(theta[0] / 2),
    -1j * np.sin(theta[0] / 2),
])
state_jacobian = np.array([[
    -0.5 * np.sin(theta[0] / 2),
    -0.5j * np.cos(theta[0] / 2),
]])
cost_gradient = np.array([np.sin(theta[0])])

step = natural_gradient_step(
    theta,
    state,
    state_jacobian,
    cost_gradient,
    learning_rate=0.05,
)
print(step.parameters)
print(step.result.diagnostics)
~~~

In a PyQPanda workflow, state and state_jacobian can come from a local
state-vector circuit path, an analytic circuit differentiator, or a separately
collected derivative workflow. The QNG solver itself is backend-independent.

## Large ansaetze

Pass disjoint parameter groups through blocks to use a block-diagonal
Fubini-Study approximation. This preserves within-block geometry while avoiding
a dense cross-layer solve.

The eigenspace solver reports rank, condition number, eigenvalues and cutoff.
Small null directions are removed with rcond; damping stabilizes nearly singular
metrics without silently accepting a materially indefinite metric.
