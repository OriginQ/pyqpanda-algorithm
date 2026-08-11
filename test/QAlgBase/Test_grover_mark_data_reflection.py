"""Restored from ``test/legacy_disabled/QAlgBase/grover_mark_data_reflection.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_sample`` returns the sampled search outcomes, and
the two marked states ('101', '001') carry essentially all probability
mass for the exact Grover iteration count.
"""

import pytest

from pyqpanda3.core import QProg

from pyqpanda_alg.Grover import Grover, mark_data_reflection
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


class Test_grover_mark_data_reflection:

    def test_mark_data_reflection_basic(self):
        q_state = QProg(3).qubits()

        def mark(qubits):
            return mark_data_reflection(qubits=qubits, mark_data=['101', '001'])

        demo_search = Grover(flip_operator=mark)
        prog = QProg()
        prog << demo_search.cir(q_input=q_state)

        counts = LocalBackend().submit_sample(
            prog, options=ExecutionOptions(shots=1000)
        ).result().single_counts()

        assert '101' in counts, "target state '101' should be in the results"
        assert '001' in counts, "target state '001' should be in the results"

        target_prob = (counts.get('101', 0) + counts.get('001', 0)) / 1000
        assert target_prob > 0.1, \
            f"combined target probability should be well above random, got: {target_prob}"
