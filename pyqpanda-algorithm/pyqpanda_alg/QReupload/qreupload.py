# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data re-uploading variational quantum learning models (QPanda3 / pyqpanda3).

A data re-uploading model interleaves data-encoding rotations with trainable
single-qubit blocks; with one feature per qubit and ``n_layers`` re-uploads the
model realises a controllable-degree truncated Fourier series in the inputs
(Schuld, Sweke & Meyer, PRA 103, 032430). A *trainable input scaling* factor on
each encoding rotation lets the model align its accessible frequency spectrum to
the data. An optional entangling ring couples qubits so multi-feature
cross-frequency terms can be represented.

Training uses pyqpanda3's exact adjoint-differentiation
(``vqcircuit.VQCircuit`` + ``DiffMethod.ADJOINT_DIFF``) — analytic gradients,
no parameter-shift overhead — with a NumPy Adam optimiser.

Design notes (verified against pyqpanda3 0.3.5):
  * Each data point is *baked* into its own circuit as a constant factor of the
    encoding rotation (``Param(lambda) * float(x)``). A product of two
    placeholders (``Param * Param``) returns a zero adjoint gradient, so data
    must enter as a constant, not a second parameter.
  * The input-scaling parameters are initialised near 1.0; initialising them
    near 0 collapses the encoding to identity and stalls training.
"""
import numpy as np

from pyqpanda3.vqcircuit import VQCircuit, DiffMethod
from pyqpanda3.core import RY, RZ, CNOT
from pyqpanda3.hamiltonian import Hamiltonian


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class _QReuploadCore:
    """Shared circuit construction + exact adjoint evaluation.

    Trainable single-qubit block = RZ . RY . RZ (general SU(2), 3 params).
    Per layer, for every qubit: encode ``RY(q, lambda * x_q)`` then the block;
    after the block an optional CNOT ring ``q -> (q+1) mod d`` entangles qubits.
    Readout = sum_q w_q <Z_q> + b  (classical head, trained jointly).

    In-circuit parameter count: ``3*d + n_layers*(d + 3*d)`` (init block + per
    layer: d input-scalings + 3*d block angles).
    """

    def __init__(self, d, n_layers, entangle, input_scaling):
        self.d = d
        self.r = n_layers
        self.entangle = entangle and d > 1
        self.input_scaling = input_scaling
        per_layer = (4 if input_scaling else 3) * d   # (lambda + block) or block only
        self.nq = 3 * d + n_layers * per_layer
        # input-scaling parameter indices (per layer l, qubit q): 3d + l*per_layer + q
        self.lam_idx = ([3 * d + l * per_layer + q for l in range(n_layers) for q in range(d)]
                        if input_scaling else [])
        self.observables = [Hamiltonian({f"Z{q}": 1.0}) for q in range(d)]

    # -- circuit (one data point baked as float) --
    def build(self, x):
        v = VQCircuit()
        v.set_Param([self.nq], ["theta"])
        counter = [0]

        def P():
            i = counter[0]; counter[0] += 1
            return v.Param([i])

        def block(q):
            v << RZ(q, P()); v << RY(q, P()); v << RZ(q, P())

        for q in range(self.d):
            block(q)
        for _ in range(self.r):
            for q in range(self.d):                       # encoding (lambda only allocated if scaling)
                v << RY(q, (P() * float(x[q])) if self.input_scaling else float(x[q]))
            for q in range(self.d):
                block(q)
            if self.entangle:
                for q in range(self.d):
                    v << CNOT(q, (q + 1) % self.d)
        return v

    def circuits(self, X):
        return [self.build(row) for row in X]

    def evaluate(self, circuits, theta, want_grad=True):
        """<Z_q> for every qubit (and d<Z_q>/dtheta) for each pre-built circuit."""
        n = len(circuits)
        Z = np.empty((n, self.d))
        G = np.empty((n, self.d, self.nq)) if want_grad else None
        for i, circ in enumerate(circuits):
            for q in range(self.d):
                res = circ.get_gradients_and_expectation(theta, self.observables[q],
                                                          DiffMethod.ADJOINT_DIFF)
                Z[i, q] = res.expectation_val()
                if want_grad:
                    G[i, q] = np.asarray(res.gradients())
        return (Z, G) if want_grad else Z

    def init_theta(self, rng):
        theta = 0.1 * rng.standard_normal(self.nq)
        if self.input_scaling:
            for j in self.lam_idx:
                theta[j] = 1.0 + 0.05 * rng.standard_normal()
        return theta


class _BaseQReupload:
    """Common fit loop (Adam on exact adjoint gradients, cosine-annealed LR)."""

    def __init__(self, n_layers=5, entangle=True, input_scaling=True,
                 epochs=800, lr=0.05, restarts=2, seed=0, verbose=False):
        if n_layers < 1 or restarts < 1 or epochs < 1:
            raise ValueError("n_layers, restarts and epochs must each be >= 1.")
        self.n_layers = n_layers
        self.entangle = entangle
        self.input_scaling = input_scaling
        self.epochs = epochs
        self.lr = lr
        self.restarts = restarts
        self.seed = seed
        self.verbose = verbose

    _param_names = ("n_layers", "entangle", "input_scaling", "epochs", "lr",
                    "restarts", "seed", "verbose")

    def get_params(self, deep=True):
        """Estimator hyper-parameters as a dict (sklearn-compatible)."""
        return {k: getattr(self, k) for k in self._param_names}

    def set_params(self, **params):
        """Set estimator hyper-parameters in place (sklearn-compatible)."""
        for k, v in params.items():
            if k not in self._param_names:
                raise ValueError(f"Invalid parameter {k!r} for {type(self).__name__}.")
            setattr(self, k, v)
        return self

    # subclasses define the loss interface
    def _d_output(self, output, y):
        raise NotImplementedError      # returns dLoss/dOutput (length-N vector)

    def _loss(self, output, y):
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must be a non-empty 2-D array (n_samples, n_features).")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X and y must contain only finite values.")
        self.n_features_in_ = X.shape[1]
        core = _QReuploadCore(self.n_features_in_, self.n_layers, self.entangle,
                              self.input_scaling)
        circuits = core.circuits(X)
        d, nq = core.d, core.nq
        best_loss, best = np.inf, None
        for rs in range(self.restarts):
            rng = (np.random.default_rng() if self.seed is None
                   else np.random.default_rng(self.seed * 100 + rs))
            theta = core.init_theta(rng)
            w = np.ones(d) / d; b = 0.0
            ntot = nq + d + 1
            m = np.zeros(ntot); v = np.zeros(ntot); b1, b2, eps = 0.9, 0.999, 1e-8
            loss = np.inf
            for t in range(1, self.epochs + 1):
                lr_t = self.lr * 0.5 * (1 + np.cos(np.pi * (t - 1) / self.epochs))
                Z, G = core.evaluate(circuits, theta)
                output = Z @ w + b
                loss = self._loss(output, y)
                d_out = self._d_output(output, y)               # [N]
                g_theta = np.einsum("n,q,nqj->j", d_out, w, G)
                g_w = (d_out[:, None] * Z).sum(0)
                g_b = d_out.sum()
                g = np.concatenate([g_theta, g_w, [g_b]])
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                step = lr_t * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                step = np.nan_to_num(step)                       # guard against divergence
                theta -= step[:nq]; w -= step[nq:nq + d]; b -= step[nq + d]
            if self.verbose:
                print(f"[QReupload] restart {rs}: train loss {loss:.4e}")
            if loss < best_loss:
                best_loss, best = loss, (theta.copy(), w.copy(), float(b))
        if best is None:
            raise RuntimeError("All restarts diverged (loss never became finite); "
                               "try a smaller lr or fewer layers.")
        self._core = core
        self.theta_, self.w_, self.b_ = best
        self.train_loss_ = best_loss
        self.n_params_ = core.nq + d + 1
        return self

    def _decision(self, X):
        if not hasattr(self, "_core"):
            raise RuntimeError("This QReupload model is not fitted yet; call fit() first.")
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            n = X.shape[1] if X.ndim == 2 else X.ndim
            raise ValueError(f"X has {n} feature(s) but the model was fitted with "
                             f"{self.n_features_in_}; pass a 2-D array "
                             f"(n_samples, {self.n_features_in_}).")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values (NaN or inf).")
        Z = self._core.evaluate(self._core.circuits(X), self.theta_, want_grad=False)
        return Z @ self.w_ + self.b_


class QReuploadRegressor(_BaseQReupload):
    """Data re-uploading variational quantum regressor (sklearn-style).

    Parameters
    ----------
    n_layers : int, default 5
        Number of data re-uploading layers (controls Fourier degree reach).
    entangle : bool, default True
        Insert a CNOT ring each layer (enables multi-feature cross-frequencies).
    input_scaling : bool, default True
        Use a trainable scaling factor on each encoding rotation.
    epochs, lr, restarts, seed, verbose : training controls.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.linspace(-np.pi, np.pi, 60)
    >>> y = np.sin(2 * X) + 0.5 * np.cos(3 * X)
    >>> reg = QReuploadRegressor(n_layers=5, epochs=400).fit(X, y)
    >>> float(np.mean((reg.predict(X) - y) ** 2)) < 1e-2
    True
    """

    def _loss(self, output, y):
        return float(np.mean((output - y) ** 2))

    def _d_output(self, output, y):
        return 2 * (output - y) / len(y)

    def predict(self, X):
        return self._decision(X)

    def score(self, X, y):
        """R^2 coefficient of determination (sklearn convention for constant y)."""
        y = np.asarray(y, float).ravel()
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:                     # constant target: 1.0 if exact else 0.0
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1 - ss_res / ss_tot)


class QReuploadClassifier(_BaseQReupload):
    """Binary data re-uploading variational quantum classifier (sklearn-style).

    Labels are mapped to {0, 1}; the readout passes through a sigmoid and is
    trained with binary cross-entropy. ``predict_proba`` returns per-class
    probabilities with columns ordered as ``classes_``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-np.pi, np.pi, (80, 2))
    >>> y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) > 0).astype(int)
    >>> clf = QReuploadClassifier(n_layers=3, epochs=300).fit(X, y)
    >>> clf.predict(X).shape == y.shape
    True
    """

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("QReuploadClassifier supports binary classification only.")
        y01 = (y == self.classes_[1]).astype(float)
        return super().fit(X, y01)

    def _loss(self, output, y):
        p = _sigmoid(output)
        return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))

    def _d_output(self, output, y):
        return (_sigmoid(output) - y) / len(y)

    def predict_proba(self, X):
        """Per-class probabilities; columns ordered as ``classes_``."""
        p1 = _sigmoid(self._decision(X))             # P(class == classes_[1])
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        idx = (self._decision(X) > 0).astype(int)
        return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))
