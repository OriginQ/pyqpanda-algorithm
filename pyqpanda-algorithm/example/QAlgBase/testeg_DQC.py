"""Run the differentiable quantum-circuit ODE example from the package tree.

This is intentionally a short state-vector smoke example.  Increase
``steps`` or select a different backend when a longer numerical audit is
required.
"""

import sys
from pathlib import Path


# Make the example runnable directly from a fresh checkout (without installing
# the package first) while keeping the path independent of the user's machine.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pyqpanda_alg.DQC import ExperimentConfig, run_experiments


def main() -> None:
    config = ExperimentConfig(
        n_qubits=3,
        depth=2,
        collocation_points=16,
        holdout_points=41,
        steps=2,
        seed=17,
    )
    run_experiments("coupled", config=config, backend="numpy")


if __name__ == "__main__":
    main()

