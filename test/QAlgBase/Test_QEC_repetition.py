import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE = (
    Path(__file__).resolve().parents[2]
    / "pyqpanda-algorithm"
    / "pyqpanda_alg"
    / "QEC"
    / "repetition_code.py"
)
spec = importlib.util.spec_from_file_location("qec_under_test", MODULE)
qec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qec
spec.loader.exec_module(qec)


SQRT2 = np.sqrt(2.0)
LOGICAL_STATES = [
    np.array([1.0, 0.0], dtype=complex),
    np.array([0.0, 1.0], dtype=complex),
    np.array([1.0 / SQRT2, 1.0 / SQRT2], dtype=complex),
    np.array([1.0 / SQRT2, -1.0 / SQRT2], dtype=complex),
    np.array([1.0 / SQRT2, 1.0j / SQRT2], dtype=complex),
]


def syndrome_integer(error_qubit):
    q1, q2 = qec.expected_syndrome(error_qubit)
    return q1 + 2 * q2


@pytest.mark.parametrize("code", ["bit_flip", "phase_flip"])
@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
@pytest.mark.parametrize("logical_state", LOGICAL_STATES)
def test_all_single_correctable_errors_restore_arbitrary_logical_state(
    code, error_qubit, logical_state
):
    result = qec.reference_repetition_recovery(
        code,
        logical_state=logical_state,
        error_qubit=error_qubit,
        return_state=True,
    )
    expected_density = np.outer(logical_state, logical_state.conj())
    np.testing.assert_allclose(
        result["logical_density"], expected_density, atol=1e-12, rtol=1e-12
    )
    assert result["logical_fidelity"] == pytest.approx(1.0, abs=1e-12)
    assert np.vdot(result["state"], result["state"]).real == pytest.approx(1.0)

    syndrome = np.zeros(4)
    syndrome[syndrome_integer(error_qubit)] = 1.0
    np.testing.assert_allclose(
        result["syndrome_probabilities"], syndrome, atol=1e-12, rtol=0.0
    )


def test_expected_syndrome_table():
    assert qec.expected_syndrome(None) == (0, 0)
    assert qec.expected_syndrome(0) == (1, 1)
    assert qec.expected_syndrome(1) == (1, 0)
    assert qec.expected_syndrome(2) == (0, 1)


def test_scope_boundary_bit_flip_code_does_not_correct_phase_error():
    logical = np.array([1.0 / SQRT2, 1.0 / SQRT2], dtype=complex)
    result = qec.reference_repetition_recovery(
        "bit_flip", logical, error_qubit=0, error_pauli="Z"
    )
    assert result["logical_fidelity"] == pytest.approx(0.0, abs=1e-12)


def test_scope_boundary_phase_flip_code_does_not_correct_bit_error():
    logical = np.array([1.0 / SQRT2, 1.0j / SQRT2], dtype=complex)
    result = qec.reference_repetition_recovery(
        "phase_flip", logical, error_qubit=0, error_pauli="X"
    )
    assert result["logical_fidelity"] < 1.0 - 1e-12


def test_convenience_wrapper_matches_functional_reference():
    logical = np.array([1.0 / SQRT2, -1.0j / SQRT2], dtype=complex)
    code = qec.ThreeQubitRepetitionCode("phase_flip")
    assert code.correctable_pauli == "Z"
    wrapped = code.reference(logical, error_qubit=2)
    direct = qec.reference_repetition_recovery(
        "phase_flip", logical, error_qubit=2
    )
    np.testing.assert_allclose(wrapped["logical_density"], direct["logical_density"])
    np.testing.assert_allclose(
        wrapped["syndrome_probabilities"], direct["syndrome_probabilities"]
    )


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda: qec.reference_repetition_recovery("bad"), ValueError),
        (lambda: qec.reference_repetition_recovery("bit_flip", [1, 1]), ValueError),
        (lambda: qec.reference_repetition_recovery("bit_flip", [1, 0], 3), ValueError),
        (
            lambda: qec.reference_repetition_recovery(
                "bit_flip", [1, 0], 0, error_pauli="Y"
            ),
            ValueError,
        ),
        (lambda: qec.expected_syndrome(4), ValueError),
    ],
)
def test_reference_validation(call, error):
    with pytest.raises(error):
        call()


class FakeGate:
    def __init__(self, name, *qubits):
        self.name = name
        self.qubits = tuple(qubits)


class FakeCircuit:
    def __init__(self):
        self.ops = []

    def __lshift__(self, operation):
        if isinstance(operation, FakeCircuit):
            self.ops.extend(operation.ops)
        else:
            self.ops.append(operation)
        return self


def fake_loader():
    return (
        FakeCircuit,
        lambda c, t: FakeGate("CNOT", c, t),
        lambda q: FakeGate("H", q),
        lambda a, b, t: FakeGate("TOFFOLI", a, b, t),
        lambda q: FakeGate("X", q),
        lambda q: FakeGate("Z", q),
    )


def op_tuples(circuit):
    return [(op.name, *op.qubits) for op in circuit.ops]


def test_circuit_plan_matches_repetition_code_algebra(monkeypatch):
    monkeypatch.setattr(qec, "_load_qpanda_gates", fake_loader)
    assert op_tuples(qec.bit_flip_encode([0, 1, 2])) == [
        ("CNOT", 0, 1),
        ("CNOT", 0, 2),
    ]
    assert op_tuples(qec.bit_flip_recover([0, 1, 2])) == [
        ("CNOT", 0, 1),
        ("CNOT", 0, 2),
        ("TOFFOLI", 1, 2, 0),
    ]
    assert op_tuples(qec.phase_flip_encode([0, 1, 2])) == [
        ("CNOT", 0, 1),
        ("CNOT", 0, 2),
        ("H", 0),
        ("H", 1),
        ("H", 2),
    ]
    assert op_tuples(qec.phase_flip_recover([0, 1, 2])) == [
        ("H", 0),
        ("H", 1),
        ("H", 2),
        ("CNOT", 0, 1),
        ("CNOT", 0, 2),
        ("TOFFOLI", 1, 2, 0),
    ]
    assert op_tuples(qec.inject_single_pauli_error([0, 1, 2], 2, "z")) == [
        ("Z", 2)
    ]


def test_circuit_input_validation_without_pyqpanda(monkeypatch):
    monkeypatch.setattr(qec, "_load_qpanda_gates", fake_loader)
    with pytest.raises(ValueError, match="exactly three"):
        qec.bit_flip_encode([0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        qec.bit_flip_encode([0, 0, 2])
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        qec.inject_single_pauli_error([0, 1, 2], 3, "X")


def reduced_density_q0_native(state):
    vector = np.asarray(state, dtype=complex).reshape(8)
    density = np.zeros((2, 2), dtype=complex)
    for syndrome in range(4):
        base = syndrome << 1
        logical = vector[[base, base | 1]]
        density += np.outer(logical, logical.conj())
    return density


def test_native_statevector_crosscheck_when_pyqpanda3_is_available():
    pytest.importorskip("pyqpanda3")
    from pyqpanda3.core import QCircuit, H, X, Z
    from pyqpanda3.quantum_info import StateVector

    target = np.array([1.0 / SQRT2, 1.0 / SQRT2], dtype=complex)
    target_density = np.outer(target, target.conj())
    for code in ("bit_flip", "phase_flip"):
        for error_qubit in (None, 0, 1, 2):
            circuit = QCircuit()
            circuit << H(0)
            if code == "bit_flip":
                circuit << qec.bit_flip_encode([0, 1, 2])
                if error_qubit is not None:
                    circuit << X(error_qubit)
                circuit << qec.bit_flip_recover([0, 1, 2])
            else:
                circuit << qec.phase_flip_encode([0, 1, 2])
                if error_qubit is not None:
                    circuit << Z(error_qubit)
                circuit << qec.phase_flip_recover([0, 1, 2])
            state = StateVector(3).evolve(circuit).ndarray()
            np.testing.assert_allclose(
                reduced_density_q0_native(state),
                target_density,
                atol=1e-10,
                rtol=1e-10,
            )
