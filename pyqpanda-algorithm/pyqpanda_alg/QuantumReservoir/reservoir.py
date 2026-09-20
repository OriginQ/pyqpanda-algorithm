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

"""Quantum reservoir computing utilities for PyQPanda3.

The quantum reservoir is deliberately fixed after construction. Input samples
drive a parameterized recurrent-style circuit while only the classical ridge
readout is trained. This keeps the quantum stage inexpensive to reuse and makes
the measurement contract explicit for simulator, cloud, and hardware backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for a fixed quantum reservoir."""

    n_qubits: int
    layers: int = 2
    input_scale: float = 1.0
    reservoir_scale: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.layers < 1:
            raise ValueError("layers must be at least 1")
        if not np.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be a positive finite value")
        if not np.isfinite(self.reservoir_scale) or self.reservoir_scale <= 0:
            raise ValueError("reservoir_scale must be a positive finite value")


@dataclass(frozen=True)
class ReservoirParameters:
    """Fixed random angles used by one reservoir instance."""

    drive: np.ndarray
    bias: np.ndarray
    coupling: np.ndarray


def make_reservoir_parameters(config: ReservoirConfig) -> ReservoirParameters:
    """Create deterministic fixed reservoir parameters from config.seed."""

    rng = np.random.default_rng(config.seed)
    shape = (config.layers, config.n_qubits)
    scale = config.reservoir_scale
    drive = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=shape) * scale
    bias = rng.uniform(-np.pi, np.pi, size=shape) * scale
    coupling = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=shape) * scale
    return ReservoirParameters(drive=drive, bias=bias, coupling=coupling)


def _validate_parameters(
    config: ReservoirConfig,
    parameters: ReservoirParameters,
) -> None:
    expected = (config.layers, config.n_qubits)
    for name, values in (
        ("drive", parameters.drive),
        ("bias", parameters.bias),
        ("coupling", parameters.coupling),
    ):
        array = np.asarray(values, dtype=float)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")


def _normalise_sample(sample: Sequence[float] | float) -> np.ndarray:
    values = np.asarray(sample, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("sample must contain at least one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample must contain only finite values")
    return values


def _ring_edges(n_qubits: int) -> tuple[tuple[int, int], ...]:
    if n_qubits < 2:
        return ()
    if n_qubits == 2:
        return ((0, 1),)
    return tuple((index, (index + 1) % n_qubits) for index in range(n_qubits))


def reservoir_circuit(
    qubits: Sequence[Any],
    sample: Sequence[float] | float,
    config: ReservoirConfig,
    parameters: ReservoirParameters | None = None,
):
    """Build the fixed quantum-reservoir circuit for one input sample.

    Input features are streamed cyclically across qubits and layers with RY
    encoding. Each layer then applies fixed RX/RZ reservoir rotations followed
    by ring ZZ interactions implemented with CNOT-RZ-CNOT.

    The function builds a circuit only; backend execution and expectation-value
    acquisition stay with the caller so the same reservoir works with local
    simulators, cloud backends, and real devices.
    """

    resolved_qubits = list(qubits)
    if len(resolved_qubits) != config.n_qubits:
        raise ValueError(
            f"expected {config.n_qubits} qubits, received {len(resolved_qubits)}"
        )
    for index, qubit in enumerate(resolved_qubits):
        if qubit in resolved_qubits[:index]:
            raise ValueError("qubits must be distinct")

    values = _normalise_sample(sample)
    fixed = parameters or make_reservoir_parameters(config)
    _validate_parameters(config, fixed)

    try:
        from pyqpanda3.core import CNOT, QCircuit, RX, RY, RZ
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ImportError(
            "PyQPanda3 is required to construct a quantum reservoir circuit"
        ) from exc

    circuit = QCircuit()
    for layer in range(config.layers):
        for qubit_index, qubit in enumerate(resolved_qubits):
            feature_index = (
                layer * config.n_qubits + qubit_index
            ) % values.size
            encoded = config.input_scale * values[feature_index]
            circuit << RY(qubit, float(encoded))
            circuit << RX(qubit, float(fixed.drive[layer, qubit_index]))
            circuit << RZ(qubit, float(fixed.bias[layer, qubit_index]))

        for edge_index, (left, right) in enumerate(_ring_edges(config.n_qubits)):
            angle = float(fixed.coupling[layer, edge_index])
            circuit << CNOT(resolved_qubits[left], resolved_qubits[right])
            circuit << RZ(resolved_qubits[right], angle)
            circuit << CNOT(resolved_qubits[left], resolved_qubits[right])

    return circuit


def reservoir_observables(n_qubits: int) -> tuple[str, ...]:
    """Return the measurement contract for reservoir_feature_vector.

    Features are all single-qubit Z expectations followed by nearest-neighbour
    ring ZZ correlations.
    """

    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    labels = [f"Z{index}" for index in range(n_qubits)]
    labels.extend(
        f"Z{left}Z{right}"
        for left, right in _ring_edges(n_qubits)
    )
    return tuple(labels)


def reservoir_feature_vector(
    z_expectations: Sequence[float],
    zz_expectations: Sequence[float] | None = None,
) -> np.ndarray:
    """Pack measured reservoir observables into one finite feature vector."""

    z_values = np.asarray(z_expectations, dtype=float).reshape(-1)
    if z_values.size == 0:
        raise ValueError("z_expectations must not be empty")
    if not np.all(np.isfinite(z_values)):
        raise ValueError("z_expectations must be finite")

    expected_edges = len(_ring_edges(z_values.size))
    if zz_expectations is None:
        if expected_edges:
            raise ValueError(
                "zz_expectations are required when the reservoir has two or more qubits"
            )
        zz_values = np.empty(0, dtype=float)
    else:
        zz_values = np.asarray(zz_expectations, dtype=float).reshape(-1)
        if zz_values.size != expected_edges:
            raise ValueError(
                f"expected {expected_edges} ZZ expectations, "
                f"received {zz_values.size}"
            )
        if not np.all(np.isfinite(zz_values)):
            raise ValueError("zz_expectations must be finite")

    return np.concatenate((z_values, zz_values))


def reservoir_feature_matrix(
    z_expectations: Sequence[Sequence[float]],
    zz_expectations: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Build a feature matrix from per-sample expectation values."""

    z_matrix = np.asarray(z_expectations, dtype=float)
    if z_matrix.ndim != 2 or z_matrix.shape[0] == 0 or z_matrix.shape[1] == 0:
        raise ValueError("z_expectations must be a non-empty 2D array")
    if not np.all(np.isfinite(z_matrix)):
        raise ValueError("z_expectations must be finite")

    n_samples, n_qubits = z_matrix.shape
    n_edges = len(_ring_edges(n_qubits))

    if n_edges == 0:
        if zz_expectations is not None:
            zz_matrix = np.asarray(zz_expectations, dtype=float)
            if zz_matrix.size:
                raise ValueError("single-qubit reservoirs have no ZZ features")
        return z_matrix.copy()

    if zz_expectations is None:
        raise ValueError("zz_expectations are required for multi-qubit reservoirs")

    zz_matrix = np.asarray(zz_expectations, dtype=float)
    expected_shape = (n_samples, n_edges)
    if zz_matrix.shape != expected_shape:
        raise ValueError(
            f"zz_expectations must have shape {expected_shape}, "
            f"received {zz_matrix.shape}"
        )
    if not np.all(np.isfinite(zz_matrix)):
        raise ValueError("zz_expectations must be finite")

    return np.concatenate((z_matrix, zz_matrix), axis=1)


class RidgeReadout:
    """Closed-form ridge readout for quantum-reservoir features."""

    def __init__(self, alpha: float = 1e-6):
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite non-negative value")
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self._single_output = False

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[float] | Sequence[Sequence[float]],
    ) -> "RidgeReadout":
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("features must be a non-empty 2D array")
        if not np.all(np.isfinite(x)):
            raise ValueError("features must be finite")

        y = np.asarray(targets, dtype=float)
        self._single_output = y.ndim == 1
        if self._single_output:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != x.shape[0]:
            raise ValueError("targets must contain one row per feature row")
        if not np.all(np.isfinite(y)):
            raise ValueError("targets must be finite")

        design = np.concatenate(
            (np.ones((x.shape[0], 1), dtype=float), x),
            axis=1,
        )
        penalty = np.eye(design.shape[1], dtype=float) * self.alpha
        penalty[0, 0] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ y
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self

    def predict(
        self,
        features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")

        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"features must have {self.coef_.shape[0]} columns"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("features must be finite")

        prediction = x @ self.coef_ + self.intercept_
        if self._single_output:
            return prediction[:, 0]
        return prediction


class QuantumReservoirRegressor:
    """Reusable fixed quantum reservoir plus trainable classical readout."""

    def __init__(
        self,
        config: ReservoirConfig,
        *,
        alpha: float = 1e-6,
        parameters: ReservoirParameters | None = None,
    ):
        self.config = config
        self.parameters = parameters or make_reservoir_parameters(config)
        _validate_parameters(config, self.parameters)
        self.readout = RidgeReadout(alpha=alpha)

    @property
    def observables(self) -> tuple[str, ...]:
        return reservoir_observables(self.config.n_qubits)

    def circuit(
        self,
        qubits: Sequence[Any],
        sample: Sequence[float] | float,
    ):
        return reservoir_circuit(
            qubits,
            sample,
            self.config,
            self.parameters,
        )

    def fit_measurements(
        self,
        z_expectations: Sequence[Sequence[float]],
        targets: Sequence[float] | Sequence[Sequence[float]],
        *,
        zz_expectations: Sequence[Sequence[float]] | None = None,
    ) -> "QuantumReservoirRegressor":
        features = reservoir_feature_matrix(
            z_expectations,
            zz_expectations,
        )
        expected = len(self.observables)
        if features.shape[1] != expected:
            raise ValueError(
                f"measurement features must contain {expected} observables"
            )
        self.readout.fit(features, targets)
        return self

    def predict_measurements(
        self,
        z_expectations: Sequence[Sequence[float]],
        *,
        zz_expectations: Sequence[Sequence[float]] | None = None,
    ) -> np.ndarray:
        features = reservoir_feature_matrix(
            z_expectations,
            zz_expectations,
        )
        return self.readout.predict(features)


__all__ = [
    "QuantumReservoirRegressor",
    "ReservoirConfig",
    "ReservoirParameters",
    "RidgeReadout",
    "make_reservoir_parameters",
    "reservoir_circuit",
    "reservoir_feature_matrix",
    "reservoir_feature_vector",
    "reservoir_observables",
]
