import random
from typing import List, Tuple


Matrix = List[List[float]]
Vector = List[float]


THREE = [
    "01111100",
    "10000010",
    "00000010",
    "00011100",
    "00000010",
    "00000010",
    "10000010",
    "01111100",
]

EIGHT = [
    "00111100",
    "01000010",
    "01000010",
    "00111100",
    "01000010",
    "01000010",
    "01000010",
    "00111100",
]


def make_mnist_like_binary_dataset(n_samples: int, seed: int) -> Tuple[Matrix, List[int]]:
    rng = random.Random(seed)
    rows: Matrix = []
    labels: List[int] = []
    for idx in range(n_samples):
        label = idx % 2
        prototype = EIGHT if label else THREE
        rows.append(_distort(_flatten(prototype), rng))
        labels.append(label)
    order = list(range(n_samples))
    rng.shuffle(order)
    return [rows[i] for i in order], [labels[i] for i in order]


def _flatten(pattern: List[str]) -> Vector:
    return [1.0 if ch == "1" else 0.0 for line in pattern for ch in line]


def _distort(base: Vector, rng: random.Random) -> Vector:
    row = []
    shift = rng.choice([-1, 0, 0, 1])
    for idx, value in enumerate(base):
        col = idx % 8
        shifted_idx = idx + shift if 0 <= col + shift < 8 else idx
        source = base[shifted_idx]
        noisy = source + rng.gauss(0.0, 0.18)
        if rng.random() < 0.035:
            noisy = 1.0 - noisy
        row.append(min(1.0, max(0.0, noisy)))
    return row

