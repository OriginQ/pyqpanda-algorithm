import math
import random
from typing import Dict, List, Optional, Sequence

from .circuit import circuit_features
from .data import Matrix, Vector
from .metrics import binary_log_loss


class LogisticBaseline:
    def __init__(self, epochs: int, lr: float = 0.08):
        self.epochs = epochs
        self.lr = lr
        self.weights: Vector = []
        self.bias = 0.0

    def fit(self, x_rows: Matrix, y: Sequence[int]):
        self.weights = [0.0] * len(x_rows[0])
        self.bias = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            for row, label in zip(x_rows, y):
                pred = _sigmoid(_dot(self.weights, row) + self.bias)
                err = pred - label
                for idx, value in enumerate(row):
                    grad_w[idx] += err * value
                grad_b += err
            for idx in range(len(self.weights)):
                self.weights[idx] -= self.lr * grad_w[idx] / len(x_rows)
            self.bias -= self.lr * grad_b / len(x_rows)
        return self

    def predict_proba(self, x_rows: Matrix) -> Vector:
        return [_sigmoid(_dot(self.weights, row) + self.bias) for row in x_rows]


class NoiseAwareVQC:
    def __init__(
        self,
        n_qubits: int,
        ansatz_layers: int,
        epochs: int,
        shots: int,
        train_noise_prob: float,
        angle_jitter: float,
        seed: int,
    ):
        self.n_qubits = n_qubits
        self.ansatz_layers = ansatz_layers
        self.epochs = epochs
        self.shots = shots
        self.train_noise_prob = train_noise_prob
        self.angle_jitter = angle_jitter
        self.seed = seed
        self.theta: Vector = []
        self.head_weights: Vector = []
        self.head_bias = 0.0
        self.history: List[Dict[str, float]] = []

    @property
    def total_params(self) -> int:
        return len(self.theta) + len(self.head_weights) + 1

    def fit(self, x_rows: Matrix, y: Sequence[int]):
        rng = random.Random(self.seed)
        self.theta = [rng.uniform(-0.18, 0.18) for _ in range(2 * self.n_qubits * self.ansatz_layers)]
        self.head_weights = [0.0] * self.n_qubits
        self.head_bias = 0.0
        for epoch in range(1, self.epochs + 1):
            features = self._features(x_rows, self.theta, None, self.train_noise_prob, self.angle_jitter, rng)
            probs = [_sigmoid(_dot(self.head_weights, feat) + self.head_bias) for feat in features]
            loss = binary_log_loss(y, probs)
            self._update_head(features, y, probs, lr=0.15)

            delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in self.theta]
            step = 0.08
            theta_plus = [value + step * d for value, d in zip(self.theta, delta)]
            theta_minus = [value - step * d for value, d in zip(self.theta, delta)]
            loss_plus = self._loss(x_rows, y, theta_plus, rng)
            loss_minus = self._loss(x_rows, y, theta_minus, rng)
            grad_scale = (loss_plus - loss_minus) / (2.0 * step)
            for idx, d in enumerate(delta):
                self.theta[idx] -= 0.07 * grad_scale * d
            self.history.append({"epoch": float(epoch), "loss": loss})
        return self

    def predict_proba(self, x_rows: Matrix, noise_prob: Optional[float] = None, angle_jitter: Optional[float] = None):
        rng = random.Random(self.seed + 999)
        features = self._features(
            x_rows,
            self.theta,
            self.shots,
            self.train_noise_prob if noise_prob is None else noise_prob,
            self.angle_jitter if angle_jitter is None else angle_jitter,
            rng,
        )
        return [_sigmoid(_dot(self.head_weights, feat) + self.head_bias) for feat in features]

    def _update_head(self, features: Matrix, y: Sequence[int], probs: Sequence[float], lr: float):
        grad_w = [0.0] * len(self.head_weights)
        grad_b = 0.0
        for feat, label, prob in zip(features, y, probs):
            err = prob - label
            for idx, value in enumerate(feat):
                grad_w[idx] += err * value
            grad_b += err
        for idx in range(len(self.head_weights)):
            self.head_weights[idx] -= lr * grad_w[idx] / len(features)
        self.head_bias -= lr * grad_b / len(features)

    def _loss(self, x_rows: Matrix, y: Sequence[int], theta: Sequence[float], rng: random.Random) -> float:
        features = self._features(x_rows, theta, None, self.train_noise_prob, self.angle_jitter, rng)
        return binary_log_loss(y, [_sigmoid(_dot(self.head_weights, feat) + self.head_bias) for feat in features])

    def _features(self, x_rows, theta, shots, noise_prob, angle_jitter, rng):
        return [
            circuit_features(row, theta, self.n_qubits, self.ansatz_layers, shots, noise_prob, angle_jitter, rng)
            for row in x_rows
        ]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)

