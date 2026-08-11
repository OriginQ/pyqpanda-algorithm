"""Purity and runtime-contract tests for the circuit-only components.

Plan 3 Task 1: the circuit builders in ``plugin.py``, ``QCmp/QCmp.py``,
``QAOA/default_circuits.py``, and ``QAOA/dstate.py`` stay backend-free —
they return circuits and never execute on a QVM.  The runtime contract
tests below verify that a QProg assembled from each component reaches
``QPandaRuntimeBackend.submit_sample()`` once measurements are attached.
"""

import pytest

from pyqpanda3.core import QCircuit, QProg

from pyqpanda_alg.QCmp import int_comparator
from pyqpanda_alg.QAOA.default_circuits import xy_mixer
from pyqpanda_alg.QAOA.dstate import prepare_dicke_state
from pyqpanda_alg.plugin import QFT, measure_all

from pyqpanda_alg.execution import ExecutionOptions


def test_circuit_components_return_circuits_without_backend():
    assert isinstance(int_comparator(2, 1, [0, 1, 2]), QCircuit)
    assert isinstance(xy_mixer([0, 1], 0.25), QCircuit)


def _measured_prog(circuit, measure_qubits):
    """Attach measurements to a builder circuit and return the QProg."""
    prog = QProg()
    prog << circuit
    prog << measure_all(measure_qubits, list(range(len(measure_qubits))))
    return prog


@pytest.mark.runtime_contract
@pytest.mark.parametrize(
    "circuit,measure_qubits",
    [
        pytest.param(QFT([0, 1, 2]), [0, 1, 2], id="plugin-qft"),
        pytest.param(
            int_comparator(2, 1, [0, 1, 2]), [0, 1, 2, 3],
            id="qcmp-int_comparator",
        ),
        pytest.param(xy_mixer([0, 1], 0.25), [0, 1], id="qaoa-xy_mixer"),
        pytest.param(
            prepare_dicke_state([0, 1, 2, 3], 2), [0, 1, 2, 3],
            id="qaoa-dstate",
        ),
    ],
)
def test_circuit_component_prog_reaches_runtime_submit_sample(
    runtime_backend, circuit, measure_qubits
):
    # The fake device's default capability surface is smaller than what
    # the representative circuits need (QFT uses CP, dstate uses BARRIER
    # and CRY), so mirror the capability configuration the execution
    # contract tests use: pin the fake device data before submitting.
    runtime_backend.device.available_qubits.return_value = list(range(6))
    runtime_backend.device.basic_gates.return_value = [
        "H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP",
        "CP", "CRY", "TOFFOLI", "BARRIER", "U1",
    ]
    runtime_backend.device.chip_topo_edges.return_value = [
        [first, second]
        for first in range(6)
        for second in range(6)
        if first != second
    ]
    task = runtime_backend.submit_sample(
        _measured_prog(circuit, measure_qubits), options=ExecutionOptions()
    )
    assert runtime_backend.service.sample_calls  # submit_sample reached the service
    assert task.result().single_counts() == {"00": 160, "11": 161}
