"""Restored from ``test/legacy_disabled/QAOA/init_d_state.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_statevector`` keeps the per-domain Hamming-weight
assertion exact and deterministic.
"""

import numpy as np
import pytest

from pyqpanda3.core import QProg
from pyqpanda_alg.QAOA import default_circuits
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _state_probs(prog):
    """Exact basis-state probabilities via the local backend."""
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    return [abs(amp) ** 2 for amp in state]


class TestInitDState:

    def calculate_domain_hamming_weights(self, result_key, domains):
        domain_weights = []
        for domain in domains:
            weight = sum(1 for idx in domain if result_key[idx] == '1')
            domain_weights.append(weight)
        return domain_weights

    def test_init_d_state_integer_domains(self):
        n_qubits = 6
        k = 2
        domains = 2

        prog = QProg(n_qubits)
        qubits = prog.qubits()

        init_circuit_func = default_circuits.init_d_state(domains, k)
        init_circuit = init_circuit_func(qubits)
        prog << init_circuit

        results = _state_probs(prog)

        domain_size = n_qubits // domains
        domain_list = [
            list(range(i * domain_size, (i + 1) * domain_size))
            for i in range(domains)
        ]

        valid_states = 0
        for key, prob in enumerate(results):
            if prob > 0.001:
                # qubit 0 is the least-significant bit of the basis index
                key_reversed = format(key, f'0{n_qubits}b')[::-1]
                domain_weights = self.calculate_domain_hamming_weights(
                    key_reversed, domain_list
                )

                for weight in domain_weights:
                    assert weight == k, \
                        f"Domain weight should be {k}, got {weight} for " \
                        f"state {key_reversed}"

                valid_states += 1

        assert valid_states > 0, "No valid Dicke states found"
