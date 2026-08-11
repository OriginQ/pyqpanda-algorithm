"""Restored from ``test/legacy_disabled/QAOA/default_circuits_prepare_dicke_state.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_list`` to the execution layer:
``LocalBackend.submit_statevector`` keeps the exact probability assertions
(only Hamming-weight-k basis states carry probability, total 1.0).
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


class TestPrepareDickeState:

    def test_prepare_dicke_state_basic(self):
        n = 4
        k = 2
        prog = QProg(n)
        qubits = prog.qubits()

        prog << default_circuits.prepare_dicke_state(qubits, k)

        results = _state_probs(prog)

        total_prob = 0.0
        valid_states_count = 0

        for key in range(2 ** n):
            prob = results[key]
            key_hmw = bin(key).count('1')

            if key_hmw == k:
                assert prob > 0, \
                    f"state {bin(key)[2:].zfill(n)} should have probability, got 0"
                total_prob += prob
                valid_states_count += 1
            else:
                assert abs(prob) < 1e-10, \
                    f"state {bin(key)[2:].zfill(n)} probability should be 0, got {prob}"

        assert abs(total_prob - 1.0) < 1e-10, \
            f"total probability should be 1, got {total_prob}"
        assert valid_states_count > 0, "no valid Dicke states found"
