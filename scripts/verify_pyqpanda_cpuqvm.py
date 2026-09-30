"""CPUQVM verification: MWIS QAOA and BF probability engines vs NumPy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "pyqpanda-algorithm"
sys.path.insert(0, str(PACKAGE_ROOT))

from pyqpanda_alg.BFDCQO import BFDCQO, QAOASolver
from pyqpanda_alg.BFDCQO.benchmark.mwis import default_mwis_instances


def main():
    app = default_mwis_instances()[0]
    p = app.to_ising()

    a = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="numpy")
    b = BFDCQO(p, n_trotter=2, dt=0.1, iterations=1, shots=None, backend="pyqpanda")
    pa, _ = a._probabilities(np.zeros(p.n))
    pb, _ = b._probabilities(np.zeros(p.n))
    if not np.allclose(pa, pb, atol=1e-8, rtol=0.0):
        raise SystemExit(f"BF CPUQVM mismatch: max error {np.max(np.abs(pa-pb))}")

    params = np.array([0.27, 0.61])  # p=1 -> [beta,gamma]
    qn = QAOASolver(p, p=1, backend="numpy", shots_per_eval=None, max_evals=2, final_shots=16)
    qp = QAOASolver(p, p=1, backend="pyqpanda", shots_per_eval=None, max_evals=2, final_shots=16)
    pn = qn.probabilities(params)
    pp = qp.probabilities(params)
    if not np.allclose(pn, pp, atol=1e-8, rtol=0.0):
        raise SystemExit(f"QAOA CPUQVM mismatch: max error {np.max(np.abs(pn-pp))}")
    print("PASS: MWIS BF/QAOA PyQPanda3 CPUQVM probabilities agree with NumPy <= 1e-8")


if __name__ == "__main__":
    main()
