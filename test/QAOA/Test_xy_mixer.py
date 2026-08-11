"""Restored from ``test/legacy_disabled/QAOA/xy_mixer.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_statevector`` keeps the domain-Hamming-weight parity
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


class TestXYMixer:

    def calculate_domain_hamming_weights(self, probs, domains):
        domain_probs = {}
        for key, prob_value in enumerate(probs):
            # qubit 0 is the least-significant bit of the basis index
            key_bits = format(key, '04b')[::-1]
            domain_weights = []
            for domain in domains:
                weight = sum(1 for idx in domain if key_bits[idx] == '1')
                domain_weights.append(weight)

            weight_key = tuple(domain_weights)
            domain_probs[weight_key] = domain_probs.get(weight_key, 0) + prob_value

        return domain_probs

    def test_xy_mixer_integer_domains_parity(self):
        n_qubits = 4
        k_domains = 2

        prog = QProg(n_qubits)
        qubits = prog.qubits()

        for q in qubits:
            prog << RX(q, np.random.random() * np.pi)

        origin_domain_probs = self.calculate_domain_hamming_weights(
            _state_probs(prog), [[0, 1], [2, 3]]
        )

        circuit = default_circuits.xy_mixer(k_domains, 'PXY')(qubits, np.pi / 2)
        prog << circuit

        final_domain_probs = self.calculate_domain_hamming_weights(
            _state_probs(prog), [[0, 1], [2, 3]]
        )

        tolerance = 0.06
        for weight_combo in origin_domain_probs:
            origin_prob = origin_domain_probs[weight_combo]
            final_prob = final_domain_probs.get(weight_combo, 0)
            assert abs(origin_prob - final_prob) < tolerance, \
                f"Domain weights {weight_combo} not preserved: " \
                f"{origin_prob:.4f} -> {final_prob:.4f}"
