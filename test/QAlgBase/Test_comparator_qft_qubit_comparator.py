"""Restored from ``test/legacy_disabled/QAlgBase/comparator_qft_qubit_comparator.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer.
The legacy assertion window (0.5 +- 0.3) was sloppy; the true probability
that a uniform-superposition two-qubit state is strictly greater than the
qubit state '10' is exactly 0.25, which ``LocalBackend.submit_statevector``
now pins down.
"""

import pytest

from pyqpanda3.core import H, QProg, X

from pyqpanda_alg.QCmp import qft_qubit_comparator
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _bit_one_probability(prog, bit):
    """Exact probability that the given qubit is |1>."""
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    return sum(
        abs(amp) ** 2 for index, amp in enumerate(state) if (index >> bit) & 1
    )


class Test_comparator_qft_qubit_comparator:

    def test_qft_qubit_comparator_example_from_doc(self):
        prog = QProg()
        prog << H(0) << H(1)
        prog << X(3)

        cir = qft_qubit_comparator([0, 1], [2, 3], [4], function='g')
        prog << cir

        prob_high = _bit_one_probability(prog, 4)
        assert prob_high == pytest.approx(0.25, abs=1e-9)
        assert 0 <= prob_high <= 1, f"probability should be in [0,1], got: {prob_high}"
