"""Restored from ``test/legacy_disabled/QAlgBase/grover_amp_operator.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_sample`` executes the amplitude-amplification
operator circuit and the sampled counts sum to the requested shots.
"""

import pytest

from pyqpanda3.core import QCircuit, QProg, TOFFOLI, Z

from pyqpanda_alg.Grover import amp_operator
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _create_mark_operator():
    def mark(qubits):
        cir = QCircuit()
        cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
        cir << Z(qubits[2])
        cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
        return cir

    return mark


class Test_grover_amp_operator:

    def test_amp_operator_basic(self):
        mark_operator = _create_mark_operator()
        qubits = QProg(3).qubits()

        circuit = amp_operator(
            q_input=qubits[:2],
            q_flip=qubits,
            q_zero=qubits[:2],
            flip_operator=mark_operator
        )

        assert isinstance(circuit, QCircuit), \
            "amp_operator should return a QCircuit"
        assert circuit is not None, "returned circuit should not be None"
        assert len(str(circuit)) > 0, "circuit should have content"

        prog = QProg()
        prog << circuit
        counts = LocalBackend().submit_sample(
            prog, options=ExecutionOptions(shots=1000)
        ).result().single_counts()
        assert sum(counts.values()) == 1000
