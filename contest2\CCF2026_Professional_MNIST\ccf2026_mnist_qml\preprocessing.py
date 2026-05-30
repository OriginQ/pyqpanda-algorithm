import math
import random
from typing import Dict, List, Sequence, Tuple

from .data import Matrix


def stratified_split(x_rows: Matrix, y: Sequence[int], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for idx, label in enumerate(y):
        by_label[int(label)].append(idx)
    splits = {"train": [], "val": [], "test": []}
    for indices in by_label.values():
        rng.shuffle(indices)
        n_train = int(round(len(indices) * train_ratio))
        n_val = int(round(len(indices) * val_ratio))
        splits["train"].extend(indices[:n_train])
        splits["val"].extend(indices[n_train:n_train + n_val])
        splits["test"].extend(indices[n_train + n_val:])
    return {
        name: ([x_rows[i] for i in indices], [int(y[i]) for i in indices])
        for name, indices in splits.items()
    }


def compress_to_8_features(x_rows: Matrix) -> Matrix:
    compressed = []
    for row in x_rows:
        blocks = [
            _mean(row[0:16]),
            _mean(row[16:32]),
            _mean(row[32:48]),
            _mean(row[48:64]),
            _mean(row[0::8]),
            _mean(row[7::8]),
            _vertical_symmetry(row),
            _center_density(row),
        ]
        compressed.append(blocks)
    return compressed


def fit_standardizer(x_rows: Matrix):
    cols = list(zip(*x_rows))
    mean = [sum(col) / len(col) for col in cols]
    scale = []
    for col, mu in zip(cols, mean):
        var = sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)
        scale.append(math.sqrt(var) or 1.0)
    return mean, scale


def apply_standardizer(x_rows: Matrix, mean, scale) -> Matrix:
    return [[(value - mean[i]) / scale[i] for i, value in enumerate(row)] for row in x_rows]


def angle_encode(x_rows: Matrix) -> Matrix:
    output = []
    for row in x_rows:
        output.append([max(-math.pi, min(math.pi, value * math.pi / 2.5)) for value in row])
    return output


def preprocess_splits(splits) -> Dict[str, Tuple[Matrix, List[int]]]:
    raw = {name: (compress_to_8_features(x), labels) for name, (x, labels) in splits.items()}
    mean, scale = fit_standardizer(raw["train"][0])
    selected = {name: (apply_standardizer(x, mean, scale), labels) for name, (x, labels) in raw.items()}
    angle = {name: (angle_encode(x), labels) for name, (x, labels) in selected.items()}
    return {"selected": selected, "angle": angle, "mean": mean, "scale": scale}


def _mean(values) -> float:
    return sum(values) / max(1, len(values))


def _vertical_symmetry(row) -> float:
    total = 0.0
    for r in range(8):
        for c in range(4):
            total += 1.0 - abs(row[r * 8 + c] - row[r * 8 + (7 - c)])
    return total / 32.0


def _center_density(row) -> float:
    ids = [r * 8 + c for r in range(2, 6) for c in range(2, 6)]
    return _mean([row[i] for i in ids])

