import pytest

from pyqpanda3.core import CPUQVM, QProg

from pyqpanda_alg.QDenseCoding import SuperdenseCoding, normalize_message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("00", "00"),
        ("01", "01"),
        ("10", "10"),
        ("11", "11"),
        (0, "00"),
        (1, "01"),
        (2, "10"),
        (3, "11"),
    ],
)
def test_normalize_message(raw, expected):
    assert normalize_message(raw) == expected


@pytest.mark.parametrize("message", ["00", "01", "10", "11"])
def test_superdense_coding_recovers_message(message):
    protocol = SuperdenseCoding(message)
    program = QProg(2)
    qubits = program.qubits()
    program << protocol.cir(qubits)

    machine = CPUQVM()
    machine.run(program, 128)
    probabilities = machine.result().get_prob_dict(qubits)

    assert max(probabilities, key=probabilities.get) == message
    assert probabilities[message] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("message", [-1, 4, "", "0", "012", "2a", True, None])
def test_invalid_messages_are_rejected(message):
    with pytest.raises((TypeError, ValueError)):
        SuperdenseCoding(message)
