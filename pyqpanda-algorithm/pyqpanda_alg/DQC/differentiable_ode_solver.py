"""Differentiable Quantum ODE Solver: self-contained state-vector examples.

The file implements a small, reproducible differentiable quantum-circuit (DQC)
solver for three first-order ODE benchmarks.  The training objective contains
only the ODE residual and the initial value.  A classical trajectory is built
after training for an independent error report.

The dependency-light backend is a NumPy state-vector implementation, so this
file can be copied and run without the rest of the OriginQ source tree. The
command-line default is ``auto``: it selects the Torch state-vector path when
PyTorch is installed and otherwise uses NumPy::

    python differentiable_ode_solver.py

If ``pyqpanda3`` is installed, ``--backend pyqpanda3`` evaluates the same
rotation/CNOT circuit with the OriginQ local CPU state-vector backend.  The
optional backend is deliberately isolated; no cloud key or hardware access is
needed for these validation examples.

The three cases are:

* ``lambda20``: a damped, rapidly rotating two-state linear system;
* ``coupled``: a non-diagonal two-state linear system;
* ``riccati``: a nonlinear, explicitly coordinate-dependent scalar equation.

Only NumPy is required for the default path.  Matplotlib is optional and is
used only when ``--plot`` is supplied.

References
----------
The DQC construction follows [1].  The optional native circuit path follows
the public QPanda3 interface documented in [2].  The numbered references are
reproduced in the accompanying README and notebook.

[1] O. Kyriienko, A. E. Paine, and V. E. Elfving, "Solving nonlinear
    differential equations with differentiable quantum circuits," Physical
    Review A 103, 052416 (2021), doi:10.1103/PhysRevA.103.052416.
[2] OriginQ, "QPanda3 documentation," https://github.com/OriginQ/QPanda3-doc.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - gives a useful message on a bare Python install
    raise SystemExit("This example requires NumPy. Install it with: python -m pip install numpy") from exc


Array = np.ndarray
Residual = Callable[[float, Array, Array], Array]
Reference = Callable[[Array], Array]
RHS = Callable[[float, Array], Array]


@dataclass(frozen=True)
class ExperimentConfig:
    """Small default configuration suitable for a local state-vector run."""

    n_qubits: int = 3
    depth: int = 2
    collocation_points: int = 24
    steps: int = 12
    holdout_points: int = 101
    learning_rate: float = 0.03
    spsa_delta: float = 0.05
    finite_difference_step: float = 1.0e-3
    seed: int = 17
    x0: float = 0.0
    x1: float = 0.9

    def __post_init__(self) -> None:
        if not 1 <= self.n_qubits <= 12:
            raise ValueError("n_qubits must be between 1 and 12 for a local state vector")
        if not 1 <= self.depth <= 12:
            raise ValueError("depth must be between 1 and 12")
        if self.collocation_points < 2 or self.holdout_points < 2:
            raise ValueError("at least two coordinates are required")
        if self.steps < 0 or self.learning_rate <= 0 or self.spsa_delta <= 0 or self.finite_difference_step <= 0:
            raise ValueError("steps must be non-negative and optimization scales positive")
        if not self.x1 > self.x0:
            raise ValueError("the interval must be increasing")
        if not -1.0 < self.x0 < 1.0 or not -1.0 < self.x1 < 1.0:
            raise ValueError("the native inverse-cosine feature map requires x in (-1, 1)")


@dataclass(frozen=True)
class ODECase:
    name: str
    initial: tuple[float, ...]
    residual: Residual
    reference: Reference
    rhs: RHS


def _rotation(axis: str, angle: float) -> Array:
    """Return a single-qubit rotation matrix using pyqpanda conventions."""

    half = 0.5 * float(angle)
    c, s = np.cos(half), np.sin(half)
    if axis == "x":
        return np.asarray([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis == "y":
        return np.asarray([[c, -s], [s, c]], dtype=complex)
    if axis == "z":
        return np.asarray([[c - 1j * s, 0.0], [0.0, c + 1j * s]], dtype=complex)
    raise ValueError(f"unknown rotation axis: {axis}")


def _apply_single_qubit(state: Array, matrix: Array, qubit: int, n_qubits: int) -> Array:
    """Apply a 2x2 gate to ``qubit`` without materializing a full unitary."""

    result = state.copy()
    stride = 1 << qubit
    block = stride << 1
    for start in range(0, state.size, block):
        for offset in range(stride):
            i0, i1 = start + offset, start + offset + stride
            a, b = state[i0], state[i1]
            result[i0] = matrix[0, 0] * a + matrix[0, 1] * b
            result[i1] = matrix[1, 0] * a + matrix[1, 1] * b
    if result.size != 1 << n_qubits:  # defensive check for accidental dimension changes
        raise RuntimeError("state-vector dimension changed during gate application")
    return result


def _apply_cnot(state: Array, control: int, target: int) -> Array:
    """Apply a CNOT using qubit zero as the least-significant state bit."""

    result = state.copy()
    target_mask = 1 << target
    for basis in range(state.size):
        if ((basis >> control) & 1) and not ((basis >> target) & 1):
            partner = basis | target_mask
            result[basis], result[partner] = state[partner], state[basis]
    return result


def _feature_angles(x: float, n_qubits: int) -> tuple[Array, Array]:
    """Chebyshev tower ``phi_q(x)=2(q+1) arccos(x)`` and its derivative."""

    value = float(x)
    if not np.isfinite(value) or not -1.0 < value < 1.0:
        raise ValueError("feature coordinates must be finite and strictly inside (-1, 1)")
    degrees = np.arange(1, n_qubits + 1, dtype=float)
    scale = np.sqrt(1.0 - value * value)
    return 2.0 * degrees * np.arccos(value), -2.0 * degrees / scale


def _numpy_statevector(
    x: float,
    theta: Sequence[float],
    n_qubits: int,
    depth: int,
    feature_shift: tuple[int, float] | None = None,
) -> Array:
    """Evaluate the RY tower + RZ-RX-RZ/CNOT ansatz on a CPU state vector."""

    parameters = np.asarray(theta, dtype=float).reshape(-1)
    expected = 3 * n_qubits * depth
    if parameters.size != expected:
        raise ValueError(f"theta has length {parameters.size}; expected {expected}")
    angles, _ = _feature_angles(x, n_qubits)
    if feature_shift is not None:
        index, shift = feature_shift
        if not 0 <= index < n_qubits:
            raise ValueError("feature-shift index is outside the register")
        angles = angles.copy()
        angles[index] += float(shift)

    state = np.zeros(1 << n_qubits, dtype=complex)
    state[0] = 1.0
    for qubit, angle in enumerate(angles):
        state = _apply_single_qubit(state, _rotation("y", angle), qubit, n_qubits)

    cursor = 0
    for layer in range(depth):
        del layer  # the loop index is retained to mirror the circuit notation
        for qubit in range(n_qubits):
            for axis in ("z", "x", "z"):
                state = _apply_single_qubit(state, _rotation(axis, parameters[cursor]), qubit, n_qubits)
                cursor += 1
        for control in range(n_qubits - 1):
            state = _apply_cnot(state, control, control + 1)
    return state


def _magnetization(n_qubits: int) -> Array:
    indices = np.arange(1 << n_qubits, dtype=np.int64)
    return sum(1.0 - 2.0 * ((indices >> qubit) & 1) for qubit in range(n_qubits))


class StatevectorDQC:
    """State-vector DQC model with a NumPy path and an optional pyqpanda3 path."""

    def __init__(self, n_qubits: int = 3, depth: int = 2, backend: str = "numpy") -> None:
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.backend = backend.lower()
        self.parameter_count = 3 * self.n_qubits * self.depth
        self._z_values = _magnetization(self.n_qubits)
        self._qvm = None
        self._vqc_types = None
        if self.backend == "pyqpanda3":
            try:
                from pyqpanda3.core import CNOT, CPUQVM, RX, RY, RZ
                from pyqpanda3.vqcircuit import VQCircuit
            except ImportError as exc:  # pragma: no cover - depends on the user's OriginQ environment
                raise ImportError(
                    "backend='pyqpanda3' needs pyqpanda3; use backend='numpy' or install pyqpanda3"
                ) from exc
            self._qvm = CPUQVM()
            self._vqc_types = (CNOT, RX, RY, RZ, VQCircuit)
        elif self.backend != "numpy":
            raise ValueError("backend must be 'numpy' or 'pyqpanda3'")

    def _native_state(
        self, x: float, theta: Sequence[float], feature_shift: tuple[int, float] | None = None
    ) -> Array:
        CNOT, RX, RY, RZ, VQCircuit = self._vqc_types
        angles, _ = _feature_angles(x, self.n_qubits)
        if feature_shift is not None:
            index, shift = feature_shift
            angles = angles.copy()
            angles[index] += float(shift)
        params = np.asarray(theta, dtype=float).reshape(self.depth, self.n_qubits, 3)
        circuit = VQCircuit()
        circuit.set_Param([self.depth, self.n_qubits, 3])
        for qubit, angle in enumerate(angles):
            circuit << RY(qubit, float(angle))
        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                circuit << RZ(qubit, circuit.Param([layer, qubit, 0]))
                circuit << RX(qubit, circuit.Param([layer, qubit, 1]))
                circuit << RZ(qubit, circuit.Param([layer, qubit, 2]))
            for control in range(self.n_qubits - 1):
                circuit << CNOT(control, control + 1)
        program = circuit(params).qprog_at([0])
        self._qvm.run(program, shots=1)
        return np.asarray(self._qvm.result().get_state_vector(), dtype=complex)

    def state(
        self, x: float, theta: Sequence[float], feature_shift: tuple[int, float] | None = None
    ) -> Array:
        if self.backend == "numpy":
            return _numpy_statevector(x, theta, self.n_qubits, self.depth, feature_shift)
        return self._native_state(x, theta, feature_shift)

    def expectation(
        self, x: float, theta: Sequence[float], feature_shift: tuple[int, float] | None = None
    ) -> float:
        state = self.state(x, theta, feature_shift)
        if state.size != self._z_values.size:
            raise RuntimeError("backend returned a state vector with an unexpected size")
        return float(np.dot(np.abs(state) ** 2, self._z_values))

    def input_derivative(self, x: float, theta: Sequence[float]) -> float:
        """Differentiate the input tower by the exact RY parameter-shift rule."""

        _, angle_derivatives = _feature_angles(x, self.n_qubits)
        derivative = 0.0
        for qubit, scale in enumerate(angle_derivatives):
            plus = self.expectation(x, theta, (qubit, 0.5 * np.pi))
            minus = self.expectation(x, theta, (qubit, -0.5 * np.pi))
            derivative += 0.5 * float(scale) * (plus - minus)
        return float(derivative)


class TorchStatevectorDQC:
    """Autograd mirror of :class:`StatevectorDQC` for environments with PyTorch.

    This optional path keeps the state vector explicit and differentiates the
    circuit directly.  It is useful for longer runs because Adam receives an
    exact gradient through the same RY tower and hardware-efficient ansatz.
    """

    def __init__(self, n_qubits: int = 3, depth: int = 2) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the user's environment
            raise ImportError("backend='torch' needs PyTorch; use backend='numpy' instead") from exc
        self.torch = torch
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.parameter_count = 3 * self.n_qubits * self.depth
        indices = torch.arange(1 << self.n_qubits, dtype=torch.long)
        self.magnetization = sum(1 - 2 * ((indices >> q) & 1) for q in range(self.n_qubits))
        self.cnot_permutations = [
            indices ^ (((indices >> control) & 1) << (control + 1))
            for control in range(self.n_qubits - 1)
        ]

    def _rotate(self, state, qubit: int, angle, axis: str):
        paired = state.reshape(state.shape[0], -1, 2, 1 << qubit)
        a, b = paired[:, :, 0, :], paired[:, :, 1, :]
        module = self.torch
        cosine = module.cos(angle / 2).reshape(-1, 1, 1) if getattr(angle, "ndim", 0) else module.cos(angle / 2)
        sine = module.sin(angle / 2).reshape(-1, 1, 1) if getattr(angle, "ndim", 0) else module.sin(angle / 2)
        if axis == "x":
            out_a, out_b = cosine * a - 1j * sine * b, -1j * sine * a + cosine * b
        elif axis == "y":
            out_a, out_b = cosine * a - sine * b, sine * a + cosine * b
        elif axis == "z":
            out_a, out_b = (cosine - 1j * sine) * a, (cosine + 1j * sine) * b
        else:
            raise ValueError(f"unknown rotation axis: {axis}")
        return module.stack((out_a, out_b), dim=2).reshape_as(state)

    def expectation(self, points, theta):
        torch = self.torch
        points = points.reshape(-1)
        theta = theta.reshape(-1)
        if theta.numel() != self.parameter_count:
            raise ValueError("incorrect Torch parameter length")
        if not bool(torch.isfinite(points).all()) or not bool((points.abs() < 1).all()):
            raise ValueError("feature coordinates must lie strictly inside (-1, 1)")
        state = torch.zeros((points.numel(), 1 << self.n_qubits), dtype=torch.complex128)
        state[:, 0] = 1.0
        degrees = torch.arange(1, self.n_qubits + 1, dtype=torch.float64)
        for qubit, degree in enumerate(degrees):
            state = self._rotate(state, qubit, 2.0 * degree * torch.acos(points), "y")
        cursor = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                for axis in ("z", "x", "z"):
                    state = self._rotate(state, qubit, theta[cursor], axis)
                    cursor += 1
            for permutation in self.cnot_permutations:
                state = state[:, permutation]
        return ((state.real.square() + state.imag.square()) * self.magnetization).sum(dim=1)

    def value_and_derivative(self, points, theta):
        torch = self.torch
        coordinates = points.detach().clone().requires_grad_(True)
        values = self.expectation(coordinates, theta)
        derivatives = torch.autograd.grad(values.sum(), coordinates, create_graph=True)[0]
        return values, derivatives


def _floating_values(
    model: StatevectorDQC,
    x: float,
    theta_rows: Array,
    initial: Array,
    raw_at_x0: Array,
) -> tuple[Array, Array]:
    raw = np.asarray([model.expectation(x, row) for row in theta_rows], dtype=float)
    derivatives = np.asarray([model.input_derivative(x, row) for row in theta_rows], dtype=float)
    return initial + raw - raw_at_x0, derivatives


def _train_case_torch(case: ODECase, config: ExperimentConfig, seed_offset: int = 0) -> dict:
    """Train with PyTorch autograd while retaining the same public result schema."""

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(config.seed + seed_offset)
    points = np.linspace(config.x0, config.x1, config.collocation_points)
    holdout = np.linspace(config.x0, config.x1, config.holdout_points)
    model = TorchStatevectorDQC(config.n_qubits, config.depth)
    parameter_shape = (len(case.initial), model.parameter_count)
    rng = np.random.default_rng(config.seed + seed_offset)
    initial_theta = rng.uniform(-0.15, 0.15, size=parameter_shape)
    theta = torch.tensor(initial_theta, dtype=torch.float64, requires_grad=True)
    points_t = torch.tensor(points, dtype=torch.float64)
    initial_t = torch.tensor(case.initial, dtype=torch.float64)

    def values_and_derivatives():
        raw_at_x0 = []
        outputs, rates = [], []
        zero = torch.zeros(1, dtype=torch.float64)
        for row in theta:
            raw0 = model.expectation(zero, row)[0]
            raw, rate = model.value_and_derivative(points_t, row)
            raw_at_x0.append(raw0)
            outputs.append(raw + initial_t[len(outputs)] - raw0)
            rates.append(rate)
        return torch.stack(outputs, dim=1), torch.stack(rates, dim=1)

    matrix_by_case = {
        "lambda20": torch.tensor([[-2.0, -20.0], [20.0, -2.0]], dtype=torch.float64),
        "coupled": torch.tensor([[3.0, 5.0], [-5.0, -3.0]], dtype=torch.float64),
    }

    def equation_residual(values, rates):
        if case.name in matrix_by_case:
            return rates - values @ matrix_by_case[case.name].T
        x = points_t[:, None]
        return rates - 4.0 * values + 6.0 * values.square() - torch.sin(50.0 * x) - values * torch.cos(25.0 * x) + 0.5

    def objective():
        values, rates = values_and_derivatives()
        return equation_residual(values, rates).square().mean()

    optimizer = torch.optim.Adam([theta], lr=config.learning_rate)
    history: list[float] = []
    best_loss = float("inf")
    best_theta = initial_theta.copy()
    for _step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Torch loss for {case.name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
        optimizer.step()
        value = float(loss.detach())
        history.append(value)
        if value < best_loss:
            best_loss, best_theta = value, theta.detach().numpy().copy()

    if config.steps:
        with torch.no_grad():
            theta.copy_(torch.from_numpy(best_theta))
        polish = torch.optim.LBFGS(
            [theta], lr=0.8, max_iter=min(20, max(1, config.steps // 2)),
            tolerance_grad=1.0e-9, tolerance_change=1.0e-12, line_search_fn="strong_wolfe",
        )

        def closure():
            polish.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        polish.step(closure)
        polished = float(objective().detach())
        history.append(polished)
        if polished < best_loss:
            best_loss, best_theta = polished, theta.detach().numpy().copy()

    theta_rows = best_theta.reshape(parameter_shape)
    initial = np.asarray(case.initial, dtype=float)
    prediction, derivative = [], []
    for coordinate in holdout:
        one_point = torch.tensor([coordinate], dtype=torch.float64)
        values = []
        rates = []
        for index, row in enumerate(theta_rows):
            raw, rate = model.value_and_derivative(torch.cat((torch.zeros(1, dtype=torch.float64), one_point)), torch.tensor(row, dtype=torch.float64))
            values.append(initial[index] + float(raw[1]) - float(raw[0]))
            rates.append(float(rate[1]))
        prediction.append(values)
        derivative.append(rates)
    prediction = np.asarray(prediction, dtype=float)
    derivative = np.asarray(derivative, dtype=float)
    reference = np.asarray(case.reference(holdout), dtype=float).reshape(prediction.shape)
    error = prediction - reference
    residuals = np.asarray([case.residual(float(x), y, dy) for x, y, dy in zip(holdout, prediction, derivative)])
    metrics = {
        "case": case.name,
        "backend": "torch",
        "n_qubits": config.n_qubits,
        "depth": config.depth,
        "collocation_points": config.collocation_points,
        "optimization_steps": config.steps,
        "training_loss": float(history[-1]) if history else float(objective().detach()),
        "best_training_loss": float(min(history)) if history else float(objective().detach()),
        "aggregate_rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "component_rmse": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "holdout_residual_rms": float(np.sqrt(np.mean(residuals * residuals))),
        "training_uses_reference": False,
    }
    return {
        "case": case.name, "x_train": points, "x": holdout,
        "prediction": prediction, "derivative": derivative, "reference": reference,
        "error": error, "theta": best_theta, "history": np.asarray(history), "metrics": metrics,
    }


def _damped_reference(points: Array) -> Array:
    points = np.asarray(points, dtype=float)
    envelope = np.exp(-2.0 * points)
    return np.column_stack((envelope * np.cos(20.0 * points), envelope * np.sin(20.0 * points)))


def _damped_rhs(_x: float, state: Array) -> Array:
    return np.asarray([[-2.0, -20.0], [20.0, -2.0]], dtype=float) @ state


def _coupled_reference(points: Array) -> Array:
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        (0.5 * np.cos(4.0 * points) + 0.375 * np.sin(4.0 * points), -0.625 * np.sin(4.0 * points))
    )


def _coupled_rhs(_x: float, state: Array) -> Array:
    return np.asarray([[3.0, 5.0], [-5.0, -3.0]], dtype=float) @ state


def _riccati_rhs(x: float, state: Array) -> Array:
    value = float(np.asarray(state).reshape(-1)[0])
    return np.asarray([4.0 * value - 6.0 * value * value + np.sin(50.0 * x) + value * np.cos(25.0 * x) - 0.5])


def _riccati_residual(x: float, values: Array, derivatives: Array) -> Array:
    value, derivative = float(values[0]), float(derivatives[0])
    return np.asarray([derivative - 4.0 * value + 6.0 * value * value - np.sin(50.0 * x) - value * np.cos(25.0 * x) + 0.5])


def _linear_residual(matrix: Array) -> Residual:
    matrix = np.asarray(matrix, dtype=float)

    def residual(_x: float, values: Array, derivatives: Array) -> Array:
        return np.asarray(derivatives, dtype=float) - matrix @ np.asarray(values, dtype=float)

    return residual


def _rk4_reference(rhs: RHS, initial: Sequence[float], points: Array, max_step: float = 2.0e-4) -> Array:
    """Independent fixed-step RK4 trajectory evaluated at the requested points."""

    coordinates = np.asarray(points, dtype=float).reshape(-1)
    if coordinates.size == 0 or np.any(np.diff(coordinates) < 0.0):
        raise ValueError("points must be non-empty and sorted")
    state = np.asarray(initial, dtype=float).copy()
    x = 0.0
    output: list[Array] = []
    for target in coordinates:
        distance = float(target - x)
        if distance < -1.0e-14:
            raise ValueError("reference points must start at or after x=0")
        if distance > 0.0:
            steps = max(1, int(np.ceil(distance / max_step)))
            h = distance / steps
            for _ in range(steps):
                k1 = rhs(x, state)
                k2 = rhs(x + 0.5 * h, state + 0.5 * h * k1)
                k3 = rhs(x + 0.5 * h, state + 0.5 * h * k2)
                k4 = rhs(x + h, state + h * k3)
                state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                x += h
        x = float(target)
        output.append(state.copy())
    return np.asarray(output, dtype=float)


def make_cases() -> dict[str, ODECase]:
    """Return the three mature first-order examples used in the release."""

    damped_matrix = np.asarray([[-2.0, -20.0], [20.0, -2.0]])
    coupled_matrix = np.asarray([[3.0, 5.0], [-5.0, -3.0]])

    def riccati_reference(points: Array) -> Array:
        return _rk4_reference(_riccati_rhs, (0.75,), points)

    return {
        "lambda20": ODECase("lambda20", (1.0, 0.0), _linear_residual(damped_matrix), _damped_reference, _damped_rhs),
        "coupled": ODECase("coupled", (0.5, 0.0), _linear_residual(coupled_matrix), _coupled_reference, _coupled_rhs),
        "riccati": ODECase("riccati", (0.75,), _riccati_residual, riccati_reference, _riccati_rhs),
    }


def residual_loss(model: StatevectorDQC, case: ODECase, points: Array, flat_theta: Array) -> float:
    """Compute the unsupervised collocation loss from ODE coefficients only."""

    theta_rows = np.asarray(flat_theta, dtype=float).reshape(len(case.initial), model.parameter_count)
    initial = np.asarray(case.initial, dtype=float)
    raw_at_x0 = np.asarray([model.expectation(0.0, row) for row in theta_rows], dtype=float)
    squared: list[float] = []
    for coordinate in points:
        values, derivatives = _floating_values(model, float(coordinate), theta_rows, initial, raw_at_x0)
        residual = np.asarray(case.residual(float(coordinate), values, derivatives), dtype=float).reshape(-1)
        if residual.size != len(case.initial) or not np.all(np.isfinite(residual)):
            return 1.0e12
        squared.extend((residual * residual).tolist())
    return float(np.mean(squared)) if squared else 1.0e12


def _adam_optimize(
    objective: Callable[[Array], float], initial: Array, config: ExperimentConfig, seed: int
) -> tuple[Array, list[float]]:
    """Deterministic Adam with coordinate differences for small circuits.

    Coordinate differences make the short default examples easier to reproduce
    from a dependency-light NumPy installation.  For larger parameter vectors,
    the routine switches to SPSA so that the number of circuit evaluations does
    not grow linearly with the ansatz size.
    """

    rng = np.random.default_rng(seed)
    parameters = np.asarray(initial, dtype=float).copy()
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    best = parameters.copy()
    best_loss = float(objective(parameters))
    history = [best_loss]
    coordinate_mode = parameters.size <= 40
    for step in range(1, config.steps + 1):
        if coordinate_mode:
            scale = config.finite_difference_step / (step**0.05)
            gradient = np.empty_like(parameters)
            for index in range(parameters.size):
                plus_parameters = parameters.copy()
                minus_parameters = parameters.copy()
                plus_parameters[index] += scale
                minus_parameters[index] -= scale
                gradient[index] = (
                    float(objective(plus_parameters)) - float(objective(minus_parameters))
                ) / (2.0 * scale)
        else:
            direction = rng.choice(np.asarray([-1.0, 1.0]), size=parameters.size)
            scale = config.spsa_delta / (step**0.101)
            plus = float(objective(parameters + scale * direction))
            minus = float(objective(parameters - scale * direction))
            gradient = ((plus - minus) / (2.0 * scale)) * direction
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=10.0, neginf=-10.0)
        gradient = np.clip(gradient, -10.0, 10.0)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        parameters -= config.learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
        current = float(objective(parameters))
        history.append(current)
        if current < best_loss:
            best_loss, best = current, parameters.copy()
    return best, history


def train_case(case: ODECase, config: ExperimentConfig, backend: str = "numpy", seed_offset: int = 0) -> dict:
    """Train one case and return predictions, references, and audit metrics."""

    requested_backend = backend.lower()
    if requested_backend == "auto":
        try:
            import torch  # noqa: F401
        except ImportError:
            requested_backend = "numpy"
        else:
            requested_backend = "torch"
    if requested_backend == "torch":
        return _train_case_torch(case, config, seed_offset=seed_offset)
    if requested_backend not in {"numpy", "pyqpanda3"}:
        raise ValueError("backend must be 'auto', 'numpy', 'torch', or 'pyqpanda3'")

    points = np.linspace(config.x0, config.x1, config.collocation_points)
    holdout = np.linspace(config.x0, config.x1, config.holdout_points)
    model = StatevectorDQC(config.n_qubits, config.depth, backend=requested_backend)
    parameter_shape = (len(case.initial), model.parameter_count)
    rng = np.random.default_rng(config.seed + seed_offset)
    initial_theta = rng.uniform(-0.15, 0.15, size=parameter_shape).reshape(-1)
    objective = lambda parameters: residual_loss(model, case, points, parameters)
    theta, history = _adam_optimize(objective, initial_theta, config, config.seed + 1000 + seed_offset)
    theta_rows = theta.reshape(parameter_shape)
    initial = np.asarray(case.initial, dtype=float)
    raw_at_x0 = np.asarray([model.expectation(0.0, row) for row in theta_rows], dtype=float)
    predictions: list[Array] = []
    derivatives: list[Array] = []
    for coordinate in holdout:
        value, derivative = _floating_values(model, float(coordinate), theta_rows, initial, raw_at_x0)
        predictions.append(value)
        derivatives.append(derivative)
    prediction = np.asarray(predictions, dtype=float)
    derivative = np.asarray(derivatives, dtype=float)
    reference = np.asarray(case.reference(holdout), dtype=float).reshape(prediction.shape)
    error = prediction - reference
    residuals = np.asarray([case.residual(float(x), y, dy) for x, y, dy in zip(holdout, prediction, derivative)])
    metrics = {
        "case": case.name,
        "backend": requested_backend,
        "n_qubits": config.n_qubits,
        "depth": config.depth,
        "collocation_points": config.collocation_points,
        "optimization_steps": config.steps,
        "training_loss": float(history[-1]),
        "best_training_loss": float(min(history)),
        "aggregate_rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "component_rmse": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "holdout_residual_rms": float(np.sqrt(np.mean(residuals * residuals))),
        "training_uses_reference": False,
    }
    return {
        "case": case.name,
        "x_train": points,
        "x": holdout,
        "prediction": prediction,
        "derivative": derivative,
        "reference": reference,
        "error": error,
        "theta": theta,
        "history": np.asarray(history, dtype=float),
        "metrics": metrics,
    }


def run_experiments(
    case: str = "all", config: ExperimentConfig | None = None, backend: str = "auto"
) -> dict[str, dict]:
    """Run one or all examples and print a compact review table."""

    config = config or ExperimentConfig()
    cases = make_cases()
    selected = tuple(cases) if case == "all" else (case,)
    unknown = [name for name in selected if name not in cases]
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(unknown)}")
    results: dict[str, dict] = {}
    for offset, name in enumerate(selected):
        result = train_case(cases[name], config, backend=backend, seed_offset=offset)
        results[name] = result
        metrics = result["metrics"]
        print(
            f"{name:8s} backend={metrics['backend']:9s} loss={metrics['best_training_loss']:.3e} "
            f"RMSE={metrics['aggregate_rmse']:.3e} max|e|={metrics['max_abs_error']:.3e}"
        )
    return results


def plot_results(results: dict[str, dict], output: str | Path | None = None) -> None:
    """Plot DQC predictions against the post-training classical references."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: install matplotlib to enable --plot.")
        return
    figure, axes = plt.subplots(1, len(results), figsize=(5.0 * len(results), 3.8), squeeze=False)
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for axis, (name, result) in zip(axes[0], results.items()):
        for column in range(result["prediction"].shape[1]):
            color = colors[column % len(colors)]
            axis.plot(result["x"], result["reference"][:, column], color=color, label=f"state {column + 1} classical")
            axis.plot(result["x"], result["prediction"][:, column], "--", color=color, label=f"state {column + 1} DQC")
        axis.set_title(f"{name}\nRMSE={result['metrics']['aggregate_rmse']:.2e}")
        axis.set_xlabel("x")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel("solution value")
    figure.tight_layout()
    if output is None:
        plt.show()
    else:
        figure.savefig(output, dpi=160)
        print(f"Saved figure: {output}")
        plt.close(figure)


def _json_ready(results: dict[str, dict]) -> dict:
    return {name: result["metrics"] for name, result in results.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "lambda20", "coupled", "riccati"), default="all")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch", "pyqpanda3"), default="auto")
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--points", type=int, default=24, help="number of collocation coordinates")
    parser.add_argument("--steps", type=int, default=12, help="Adam update steps (increase for a longer audit)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--plot", nargs="?", const="dqc_ode_results.png", help="save a comparison figure")
    parser.add_argument("--json", type=Path, help="write metrics to a JSON file")
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        n_qubits=args.qubits,
        depth=args.depth,
        collocation_points=args.points,
        steps=args.steps,
        seed=args.seed,
    )
    results = run_experiments(args.case, config=config, backend=args.backend)
    if args.plot is not None:
        plot_results(results, args.plot)
    if args.json is not None:
        args.json.write_text(json.dumps(_json_ready(results), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved metrics: {args.json}")
    return 0


__all__ = [
    "ExperimentConfig",
    "ODECase",
    "StatevectorDQC",
    "TorchStatevectorDQC",
    "make_cases",
    "plot_results",
    "residual_loss",
    "run_experiments",
    "train_case",
]


if __name__ == "__main__":
    raise SystemExit(main())

