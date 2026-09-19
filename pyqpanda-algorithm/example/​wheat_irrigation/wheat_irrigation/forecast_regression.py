"""Small, auditable regressors for daily ET0 and soil-water prediction.

The VQC uses angle encoding, trainable rotations, entanglement, and a
classical readout in a four-qubit NumPy statevector simulation. The model
makes no claim of quantum advantage. All scalers are fitted exclusively
on the samples supplied to ``fit``.
"""

import copy
import math

import numpy as np


def _finite_array(value, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _features(value, expected_columns=None):
    array = _finite_array(value, "X")
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise ValueError(f"X must have {expected_columns} feature columns")
    return array


def _target(value, name="y"):
    array = _finite_array(value, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty single-target vector")
    return array


def _checked(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} exceeds the supported floating-point range")
    return value


class _MinMax:
    """Training bounds, with divisor one for constant columns; no clipping."""

    def __init__(self, values):
        self.minimum = np.min(values, axis=0)
        self.maximum = np.max(values, axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            span = _checked(self.maximum - self.minimum, "Training range")
        self.scale = np.where(span == 0, 1.0, span)

    def transform(self, values):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return _checked((values - self.minimum) / self.scale, "Scaled values")

    def inverse(self, values):
        with np.errstate(over="ignore", invalid="ignore"):
            return _checked(values * self.scale + self.minimum, "Predictions")

    def describe(self):
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                "scale": self.scale.tolist(), "range": [0.0, 1.0],
                "clip": False, "fit_scope": "training_samples_only"}


def _ridge_fit(features, target, alpha):
    """Centered ridge via augmented least squares; the intercept is unpenalized."""
    feature_mean = np.mean(features, axis=0)
    target_mean = float(np.mean(target))
    centered = features - feature_mean
    # Augmentation is stable for singular/constant columns and alpha == 0.
    matrix = np.vstack((centered, math.sqrt(alpha) * np.eye(features.shape[1])))
    values = np.concatenate((target - target_mean, np.zeros(features.shape[1])))
    coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
    intercept = target_mean - feature_mean @ coefficients
    return _checked(coefficients, "Ridge coefficients"), float(intercept)


def _apply_single_qubit(states, gate, qubit):
    """Apply a shared 2x2 or per-sample Nx2x2 gate; qubit 0 is the low bit."""
    zero = np.arange(16)[(np.arange(16) & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    low, high = states[:, zero].copy(), states[:, one].copy()
    if gate.ndim == 2:
        states[:, zero] = gate[0, 0] * low + gate[0, 1] * high
        states[:, one] = gate[1, 0] * low + gate[1, 1] * high
    else:
        states[:, zero] = gate[:, 0, 0, None] * low + gate[:, 0, 1, None] * high
        states[:, one] = gate[:, 1, 0, None] * low + gate[:, 1, 1, None] * high


def _apply_cnot(states, control, target):
    indices = np.arange(16)
    permutation = indices ^ (((indices >> control) & 1) << target)
    states[:] = states[:, permutation]


def _vqc_statevector(encoded_angles, parameters):
    """Return batched four-qubit states for RY -> RX/RY/RZ -> ring CNOT."""
    encoded = _finite_array(encoded_angles, "encoded_angles")
    params = _finite_array(parameters, "parameters")
    if encoded.ndim != 2 or encoded.shape[1] != 4 or params.shape != (4, 3):
        raise ValueError("VQC expects encoded angles (n, 4) and parameters (4, 3)")
    states = np.zeros((len(encoded), 16), dtype=complex)
    states[:, 0] = 1
    for qubit in range(4):
        cosine, sine = np.cos(encoded[:, qubit] / 2), np.sin(encoded[:, qubit] / 2)
        gate = np.stack((cosine, -sine, sine, cosine), axis=-1).reshape(-1, 2, 2)
        _apply_single_qubit(states, gate, qubit)
    for qubit, (rx, ry, rz) in enumerate(params):
        cosine, sine = np.cos(rx / 2), np.sin(rx / 2)
        _apply_single_qubit(states, np.array([[cosine, -1j * sine],
                                             [-1j * sine, cosine]]), qubit)
        cosine, sine = np.cos(ry / 2), np.sin(ry / 2)
        _apply_single_qubit(states, np.array([[cosine, -sine], [sine, cosine]]), qubit)
        _apply_single_qubit(states, np.diag([np.exp(-0.5j * rz),
                                            np.exp(0.5j * rz)]), qubit)
    for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
        _apply_cnot(states, control, target)
    return states


def _pauli_z_expectations(states):
    signs = 1 - 2 * ((np.arange(16)[:, None] >> np.arange(4)) & 1)
    return np.square(np.abs(states)) @ signs


class _Regressor:
    def __init__(self, method, seed, maxiter, alpha):
        self.method, self.seed, self.maxiter, self.alpha = method, seed, maxiter, alpha
        self._description = None

    def fit(self, X, y):
        features, target = _features(X), _target(y)
        if len(features) != len(target):
            raise ValueError("X and y must contain the same number of samples")
        self.feature_scaler = _MinMax(features)
        self.target_scaler = _MinMax(target)
        self.n_features = features.shape[1]
        scaled_x = self.feature_scaler.transform(features)
        scaled_y = self.target_scaler.transform(target)
        self._description = None
        extra = {}
        if self.method == "vqc":
            transformed, extra = self._fit_quantum(scaled_x, scaled_y)
        else:
            transformed = scaled_x
        self.coefficients, self.intercept = _ridge_fit(transformed, scaled_y, self.alpha)
        residual = transformed @ self.coefficients + self.intercept - scaled_y
        self._description = {
            "method": self.method, "seed": self.seed, "alpha": self.alpha,
            "n_training_samples": len(features), "n_features": self.n_features,
            "feature_scaler": self.feature_scaler.describe(),
            "target_scaler": self.target_scaler.describe(),
            "coefficients": self.coefficients.tolist(), "intercept": self.intercept,
            "training_mse_scaled": float(np.mean(residual ** 2)),
            "readout": "ridge_with_unpenalized_intercept", **extra,
        }
        return self

    def _encode(self, scaled_x):
        if self.projection is None:
            encoded = np.zeros((len(scaled_x), 4))
            encoded[:, :self.n_features] = scaled_x
        else:
            encoded = scaled_x @ self.projection
        with np.errstate(over="ignore", invalid="ignore"):
            return _checked(np.pi * encoded, "Encoded angles")

    def _fit_quantum(self, scaled_x, scaled_y):
        # Importing/using the ridge baseline requires only NumPy.
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError("The vqc forecasting method requires scipy>=1.9") from exc
        rng = np.random.default_rng(self.seed)
        self.projection = None
        if self.n_features > 4:
            self.projection = rng.normal(size=(self.n_features, 4))
            self.projection /= np.sum(np.abs(self.projection), axis=0)
        initial = rng.normal(scale=0.35, size=(4, 3))
        encoded = self._encode(scaled_x)
        evaluations = 0
        circuit_evaluations = 0
        best = {"loss": math.inf, "angles": initial.copy()}

        def objective(flat):
            nonlocal evaluations, circuit_evaluations
            expectations = _pauli_z_expectations(_vqc_statevector(encoded, flat.reshape(4, 3)))
            coefficients, intercept = _ridge_fit(expectations, scaled_y, self.alpha)
            error = expectations @ coefficients + intercept - scaled_y
            # Penalized mean loss matches the closed-form ridge readout.
            loss = float((error @ error + self.alpha * (coefficients @ coefficients))
                         / len(scaled_y))
            evaluations += 1
            circuit_evaluations += 1
            if loss < best["loss"]:
                best.update(loss=loss, angles=flat.reshape(4, 3).copy())
            # The readout minimizes the penalized loss at this point, so the
            # envelope theorem removes derivatives through the ridge solution.
            # Each angle occurs in one Pauli rotation: parameter shift is exact.
            gradient = np.zeros(12)
            if np.any(coefficients):
                for index in range(12):
                    plus, minus = flat.copy(), flat.copy()
                    plus[index] += np.pi / 2
                    minus[index] -= np.pi / 2
                    forward = _pauli_z_expectations(_vqc_statevector(encoded, plus.reshape(4, 3)))
                    backward = _pauli_z_expectations(_vqc_statevector(encoded, minus.reshape(4, 3)))
                    gradient[index] = error @ ((forward - backward) @ coefficients) / len(scaled_y)
                    circuit_evaluations += 2
            return loss, gradient

        initial_loss, _ = objective(initial.ravel())
        if np.all(scaled_y == scaled_y[0]):
            optimization = {"success": True, "status": 0,
                            "message": "Constant training target; circuit optimization unnecessary",
                            "nit": 0, "nfev": 0}
        else:
            result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                              options={"maxiter": self.maxiter, "ftol": 1e-10})
            optimization = {"success": bool(result.success), "status": int(result.status),
                            "message": str(result.message), "nit": int(result.nit),
                            "nfev": int(result.nfev)}
        # Retaining the best evaluated point also covers a reached iteration limit.
        self.angles = best["angles"]
        optimization.update(initial_loss=initial_loss, training_loss=best["loss"],
                            evaluations=evaluations, maxiter=self.maxiter,
                            circuit_evaluations=circuit_evaluations,
                            gradient="exact_parameter_shift_with_variable_projection",
                            loss="(sum_squared_error + alpha * squared_readout_norm) / n")
        description = {
            "n_qubits": 4, "backend": "numpy_cpu_statevector",
            "encoding": "RY(pi * scaled_features), zero padded to 4; if >4, first project",
            "projection": None if self.projection is None else self.projection.tolist(),
            "projection_rule": "seeded Gaussian columns normalized by L1 norm, fixed during fit",
            "circuit": "RY encoding; trainable RX/RY/RZ per qubit; CNOT 0->1,1->2,2->3,3->0",
            "measurement": "Pauli Z expectation on each qubit; qubit 0 is least-significant bit",
            "initial_angles": initial.tolist(), "angles": self.angles.tolist(),
            "optimization": optimization,
            "scope": "4-qubit trainable regression model for daily ET0 and soil-water prediction",
            "quantum_advantage_claimed": False,
        }
        return _pauli_z_expectations(_vqc_statevector(encoded, self.angles)), description

    def predict(self, X):
        if self._description is None:
            raise ValueError("Regressor must be fitted before prediction")
        features = self.feature_scaler.transform(_features(X, self.n_features))
        if self.method == "vqc":
            features = _pauli_z_expectations(_vqc_statevector(self._encode(features), self.angles))
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = _checked(features @ self.coefficients + self.intercept,
                                 "Scaled predictions")
        return self.target_scaler.inverse(predicted)

    def describe(self):
        if self._description is None:
            raise ValueError("Regressor must be fitted before describing it")
        return copy.deepcopy(self._description)


def make_regressor(method="ridge", seed=7, maxiter=30, alpha=0.01):
    """Create a single-target regressor, without fitting on future observations."""
    if method not in ("ridge", "vqc"):
        raise ValueError("method must be 'ridge' or 'vqc'")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
        raise ValueError("alpha must be a finite nonnegative number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative number")
    return _Regressor(method, int(seed), int(maxiter), alpha)


def regression_metrics(observed, predicted):
    """MAE, RMSE, NSE and original KGE (std ratio, mean ratio, correlation).

    Undefined NSE/KGE values are ``None``, suitable for strict JSON. Constant
    observations make NSE undefined; either constant series or a zero observed
    mean makes KGE undefined. Metrics always use the original target units.
    """
    observed, predicted = _target(observed, "observed"), _target(predicted, "predicted")
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have the same length")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = _checked(predicted - observed, "Metric errors")
        error_scale = float(np.max(np.abs(error)))
        normalized_error = error if error_scale == 0 else error / error_scale
        mae = error_scale * float(np.mean(np.abs(normalized_error)))
        rmse = error_scale * float(np.sqrt(np.mean(normalized_error ** 2)))
        # Normalize both series by a shared scale to keep squared moments finite.
        scale = max(float(np.max(np.abs(observed))), float(np.max(np.abs(predicted))), 1.0)
        obs, pred = observed / scale, predicted / scale
        obs_mean, pred_mean = float(np.mean(obs)), float(np.mean(pred))
        obs_centered, pred_centered = obs - obs_mean, pred - pred_mean
        obs_ss = float(obs_centered @ obs_centered)
        pred_ss = float(pred_centered @ pred_centered)
        nse = None
        kge = None
        if not np.all(observed == observed[0]) and obs_ss > 0:
            delta = pred - obs
            nse = float(1 - (delta @ delta) / obs_ss)
            if not np.all(predicted == predicted[0]) and pred_ss > 0 and obs_mean != 0:
                correlation = float((obs_centered @ pred_centered)
                                    / (math.sqrt(obs_ss) * math.sqrt(pred_ss)))
                correlation = float(np.clip(correlation, -1.0, 1.0))
                std_ratio = math.sqrt(pred_ss / obs_ss)
                mean_ratio = pred_mean / obs_mean
                kge = float(1 - math.hypot(correlation - 1, std_ratio - 1, mean_ratio - 1))
    result = {"mae": mae, "rmse": rmse, "nse": nse, "kge": kge}
    for name, value in result.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} exceeds the supported floating-point range")
    return result
