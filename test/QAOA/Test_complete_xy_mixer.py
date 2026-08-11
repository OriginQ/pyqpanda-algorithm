"""Restored from ``test/legacy_disabled/QAOA/complete_xy_mixer.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_statevector`` keeps the Hamming-weight-preservation
assertion exact and deterministic instead of sampling-noisy.
"""

import numpy as np
import pytest

from pyqpanda3.core import QProg, RX
from pyqpanda_alg.QAOA import default_circuits
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _state_probs(prog):
    """Exact basis-state probabilities via the local backend."""
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    return [abs(amp) ** 2 for amp in state]


def _hamming_weight_distribution(probs, max_weight):
    weight_probs = {}
    for weight in range(max_weight + 1):
        weight_probs[weight] = sum(
            prob for key, prob in enumerate(probs) if bin(key).count('1') == weight
        )
    return weight_probs


class TestCompleteXYMixer:

    def test_complete_xy_mixer_hamming_weight_preservation(self):
        n_qubits = 4
        prog = QProg(n_qubits)
        qubits = prog.qubits()

        for q in qubits:
            prog << RX(q, np.random.random() * 2 * np.pi)

        original_weight_probs = _hamming_weight_distribution(
            _state_probs(prog), n_qubits
        )

        # apply complete_xy_mixer
        beta = np.pi / 5
        circuit = default_circuits.complete_xy_mixer(qubits, beta)
        prog << circuit

        final_weight_probs = _hamming_weight_distribution(
            _state_probs(prog), n_qubits
        )

        tolerance = 0.05
        for weight in range(n_qubits + 1):
            original_prob = original_weight_probs.get(weight, 0)
            final_prob = final_weight_probs.get(weight, 0)
            assert abs(original_prob - final_prob) < tolerance, \
                f"Hamming weight {weight} not preserved within tolerance: " \
                f"{original_prob:.4f} -> {final_prob:.4f}"
