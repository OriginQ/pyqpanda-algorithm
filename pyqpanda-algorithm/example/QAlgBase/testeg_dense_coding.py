"""Minimal superdense-coding example for PyQPanda3."""

from pyqpanda3.core import CPUQVM, QProg

from pyqpanda_alg.QDenseCoding import SuperdenseCoding


def main():
    protocol = SuperdenseCoding("10")

    program = QProg(2)
    qubits = program.qubits()
    program << protocol.cir(qubits)

    machine = CPUQVM()
    machine.run(program, 256)
    probabilities = machine.result().get_prob_dict(qubits)
    recovered = max(probabilities, key=probabilities.get)

    print("sent:", protocol.expected_bits)
    print("recovered:", recovered)
    print("probabilities:", probabilities)


if __name__ == "__main__":
    main()
