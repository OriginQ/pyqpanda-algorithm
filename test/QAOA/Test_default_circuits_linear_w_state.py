"""Restored from ``test/legacy_disabled/QAOA/default_circuits_linear_w_state.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_list`` to the execution layer:
``LocalBackend.submit_statevector`` keeps the exact probability assertions
(only Hamming-weight-1 basis states, each with probability 1/n).
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


class TestLinearWState:
    """linear_w_state interface test class"""

    def test_linear_w_state_basic_compressed(self):
        n = 3
        prog = QProg(n)
        qubits = prog.qubits()

        # prepare the W state (compressed mode)
        prog << default_circuits.linear_w_state(qubits, compress=True)

        results = _state_probs(prog)

        # verify: only Hamming-weight-1 basis states carry probability,
        # each with probability 1/n
        total_prob = 0.0
        valid_states_count = 0
        expected_prob = 1.0 / n

        for key in range(2 ** n):
            prob = results[key]
            key_hmw = bin(key).count('1')

            if key_hmw == 1:
                assert prob > 0, \
                    f"state {bin(key)[2:].zfill(n)} should have probability, got 0"
                total_prob += prob
                valid_states_count += 1
                assert abs(prob - expected_prob) < 1e-10, \
                    f"state {bin(key)[2:].zfill(n)} probability should be " \
                    f"{expected_prob}, got {prob}"
            else:
                assert abs(prob) < 1e-10, \
                    f"state {bin(key)[2:].zfill(n)} probability should be 0, got {prob}"

        assert abs(total_prob - 1.0) < 1e-10, \
            f"total probability should be 1, got {total_prob}"
        assert valid_states_count == n, \
            f"valid state count should be {n}, got {valid_states_count}"
