"""Run a compact constraint-preserving QAOA QmRMR example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyqpanda_alg.QmRMR import QmRMRFeatureSelection  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    relevance = rng.random(6)
    quadratic = rng.random((6, 6))
    quadratic = 0.5 * (quadratic + quadratic.T)
    selector = QmRMRFeatureSelection(
        quadratic,
        relevance,
        select_num=3,
        layers=2,
        initializer="amplitude",
        maxiter=8,
        seed=7,
    )
    result = selector.optimize()
    print(result.summary())


if __name__ == "__main__":
    main()


