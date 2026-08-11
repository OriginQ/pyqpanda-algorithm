"""Plan 3 Task 6 inventory: every executable public algorithm has one
LocalBackend test and one credential-free runtime-contract test; every
transpiled circuit component has a runtime-contract test.

The ``MIGRATED_ALGORITHMS`` dict below names the full public algorithm
inventory (QAOA, QARM, QKmeans, QPCA, QSVM, QUBO_QAOA, QUBO_GAS, QAE,
QSVD, QSVR, Grover, GroverAdaptiveSearch, QmRMR, QSEncode) and adapts
each to a zero-argument call that runs end-to-end on ``LocalBackend``
through the migrated public surface (``backend=`` / ``options=``
keyword arguments only).

The ledger tests scan the sibling runtime-contract files so the two
coverage kinds can never silently degrade: an algorithm whose contract
test file stops referencing ``RecordingBackend`` (i.e. runs on a real
QVM or CPUQVM) fails the inventory.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from pyqpanda3.core import QCircuit, QProg, RY, X

from pyqpanda_alg import QARM
from pyqpanda_alg.Grover import Grover, mark_data_reflection
from pyqpanda_alg.Grover.Grover_core import GroverAdaptiveSearch
from pyqpanda_alg.QAE import QAE
from pyqpanda_alg.QARM.qarm import QuantumAssociationRulesMining
from pyqpanda_alg.QKmeans import QuantumKmeans
from pyqpanda_alg.QAOA.qaoa import QAOA
from pyqpanda_alg.QPCA import qpca
from pyqpanda_alg.QSEncode import QSpare_Code
from pyqpanda_alg.QSVD import SVD
from pyqpanda_alg.QSVM import QuantumKernel_vqnet
from pyqpanda_alg.QSVR import Quantum_SVR
from pyqpanda_alg.QUBO import QUBO_GAS_origin, QUBO_QAOA
from pyqpanda_alg.QmRMR.QmRMR_core import Feature_Selection
from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def _points():
    return np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])


def _xs():
    return np.array([[0.0, 0.0], [1.0, 1.0]])


def _ys():
    return np.array([0.0, 1.0])


def _qaoa_problem():
    return sp.Symbol("x0")


def _qubo_problem():
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    return (
        -0.5 * x0 * x1
        - 0.7 * x0 * x1
        + 0.9 * x1 * x2
        + 1.3 * x0
        - x1
        - 0.5 * x2
    )


def _qae_operator(qlist):
    """Operator whose target amplitude is sin(pi/3) on |11>."""
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    return cir


def _grover_search_prog():
    """Grover search circuit for the two marked states '101' and '001'."""
    q_state = list(range(3))

    def mark(qubits):
        return mark_data_reflection(qubits=qubits, mark_data=["101", "001"])

    prog = QProg()
    prog << Grover(flip_operator=mark).cir(q_input=q_state)
    return prog


def _qarm_transactions():
    data_file = os.path.join(QARM.__path__[0], "dataset/data2.txt")
    with open(data_file, "r", encoding="utf8") as handle:
        return [
            [data.strip() for data in line.strip().split(",")]
            for line in handle
            if line
        ]


def _gas_value_function(key):
    return -1 if key == "10" else 0


MIGRATED_ALGORITHMS = {
    "QAOA": lambda: QAOA(_qaoa_problem()).run(backend=LocalBackend()),
    "QARM": lambda: QuantumAssociationRulesMining(
        _qarm_transactions(), 0.2, 0.5
    ).run(backend=LocalBackend()),
    "QKmeans": lambda: QuantumKmeans(k=2).fit(_points(), backend=LocalBackend()),
    "QPCA": lambda: qpca(_points(), 1, backend=LocalBackend()),
    "QSVM": lambda: QuantumKernel_vqnet(n_qbits=2).evaluate(
        _xs(), backend=LocalBackend()
    ),
    "QUBO_QAOA": lambda: QUBO_QAOA(_qubo_problem()).run(
        layer=1,
        optimizer_option={"options": {"maxiter": 1}},
        backend=LocalBackend(),
    ),
    "QUBO_GAS": lambda: QUBO_GAS_origin(_qubo_problem()).run(
        init_value=0, continue_times=2, backend=LocalBackend()
    ),
    "QAE": lambda: QAE(
        operator_in=_qae_operator,
        qnumber=2,
        epsilon=0.01,
        res_index=[0, 1],
        target_state="11",
    ).run(backend=LocalBackend()),
    "QSVD": lambda: SVD([[1.0, 0.0], [0.0, 0.5]]).QSVD_min(
        backend=LocalBackend(), maxiter=1
    ),
    "QSVR": lambda: Quantum_SVR(_xs(), _ys()).get_res(backend=LocalBackend()),
    "Grover": lambda: LocalBackend().submit_sample(
        _grover_search_prog(), options=ExecutionOptions(shots=200)
    ).result().single_counts(),
    "GroverAdaptiveSearch": lambda: GroverAdaptiveSearch(
        init_value=0, n_index=2
    ).run(
        continue_times=2,
        n_value_function=lambda current_min: 1,
        value_function=_gas_value_function,
        backend=LocalBackend(),
    ),
    "QmRMR": lambda: Feature_Selection(
        [[0.3, 0.1], [0.1, 0.2]], [0.6, 0.4], 1
    ).get_his_res([0.5, 0.5], backend=LocalBackend()),
    "QSEncode": lambda: QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(
        backend=LocalBackend()
    ),
}

#: Sibling file whose recording-backend contract covers each algorithm.
ALGORITHM_CONTRACT_FILES = {
    "QAOA": "test_qaoa_backend.py",
    "QARM": "test_qarm_backend.py",
    "QKmeans": "test_sampling_ml_algorithms.py",
    "QPCA": "test_sampling_ml_algorithms.py",
    "QSVM": "test_sampling_ml_algorithms.py",
    "QUBO_QAOA": "test_qubo_backend.py",
    "QUBO_GAS": "test_grover_backend.py",
    "QAE": "test_qae_backend.py",
    "QSVD": "test_qsvd_backend.py",
    "QSVR": "test_sampling_ml_algorithms.py",
    "Grover": "test_grover_backend.py",
    "GroverAdaptiveSearch": "test_grover_backend.py",
    "QmRMR": "test_qmrmr_backend.py",
    "QSEncode": "test_sampling_ml_algorithms.py",
}

#: Transpilation inventory: circuit components stay backend-free and reach
#: ``submit_sample`` through the runtime-contract file.
TRANSPILED_COMPONENTS = {
    "QCmp": "test_circuit_components.py",
    "plugin": "test_circuit_components.py",
    "default_circuits": "test_circuit_components.py",
    "dstate": "test_circuit_components.py",
}


@pytest.mark.parametrize("name", sorted(MIGRATED_ALGORITHMS))
def test_executable_algorithm_runs_on_local_backend(name):
    result = MIGRATED_ALGORITHMS[name]()
    assert result is not None


def test_every_algorithm_has_a_credential_free_runtime_contract_test():
    suite_dir = Path(__file__).parent
    assert set(MIGRATED_ALGORITHMS) == set(ALGORITHM_CONTRACT_FILES), \
        "inventory and contract-file ledger must stay in sync"
    for name, file_name in sorted(ALGORITHM_CONTRACT_FILES.items()):
        text = (suite_dir / file_name).read_text(encoding="utf-8")
        assert name in text, f"{name} missing from {file_name}"
        assert "RecordingBackend" in text or "recording_backend" in text, \
            f"{file_name} no longer uses a recording backend for {name}"


def test_every_transpiled_component_has_a_runtime_contract_test():
    suite_dir = Path(__file__).parent
    for component, file_name in TRANSPILED_COMPONENTS.items():
        text = (suite_dir / file_name).read_text(encoding="utf-8")
        assert component in text, f"{component} missing from {file_name}"
        assert "submit_sample" in text, \
            f"{file_name} no longer reaches submit_sample for {component}"
