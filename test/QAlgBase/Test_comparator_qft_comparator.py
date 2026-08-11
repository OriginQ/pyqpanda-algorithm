"""Restored from ``test/legacy_disabled/QAlgBase/comparator_qft_comparator.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_statevector`` makes the P(result qubit = 1) = 0.25
assertion (the docstring-pinned value for the uniform-superposition input)
exact.
"""

import pytest

from pyqpanda3.core import H, QProg

from pyqpanda_alg.QCmp import qft_comparator
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _bit_one_probability(prog, bit):
    """Exact probability that the given qubit is |1>."""
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    return sum(
        abs(amp) ** 2 for index, amp in enumerate(state) if (index >> bit) & 1
    )


class Test_comparator_qft_comparator:

    def test_qft_comparator_example_from_doc(self):
        value = 2
        prog = QProg()
        prog << H(0) << H(1)

        # compare the uniform superposition with 2
        cir = qft_comparator(value, [0, 1], [2], function='g')
        prog << cir

        prob_high = _bit_one_probability(prog, 2)
        assert prob_high == pytest.approx(0.25, abs=1e-9)
        assert 0 <= prob_high <= 1, f"probability should be in [0,1], got: {prob_high}"
