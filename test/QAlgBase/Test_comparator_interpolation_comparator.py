"""Restored from ``test/legacy_disabled/QAlgBase/comparator_interpolation_comparator.py``.

Re-migrated from ``CPUQVM.run`` + ``get_prob_dict`` to the execution layer:
``LocalBackend.submit_statevector`` makes the P(result qubit = 1) = 0.2
assertion (the docstring-pinned value for state '110' vs 3.3) exact.
"""

import pytest

from pyqpanda3.core import I, QProg, X

from pyqpanda_alg.QCmp import interpolation_comparator
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _bit_one_probability(prog, bit):
    """Exact probability that the given qubit is |1>."""
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    return sum(
        abs(amp) ** 2 for index, amp in enumerate(state) if (index >> bit) & 1
    )


class Test_comparator_interpolation_comparator:

    def test_interpolation_comparator_example_from_doc(self):
        value = 3.3
        prog = QProg()
        prog << X(0) << X(1) << I(2)

        cir = interpolation_comparator(
            value, [0, 1, 2], [3, 4, 5], function='g', reuse=True
        )
        prog << cir

        prob_high = _bit_one_probability(prog, 5)
        assert prob_high == pytest.approx(0.2, abs=1e-9)
        assert 0 <= prob_high <= 1, f"probability should be in [0,1], got: {prob_high}"
