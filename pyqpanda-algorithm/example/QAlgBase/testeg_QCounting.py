from math import pi

from pyqpanda3.core import QCircuit, TOFFOLI, Z
from pyqpanda_alg.QCounting import QCounting

"""
QCounting estimates how many items of a 2 ** qnumber search space are marked by an
oracle. QCounting(...).run() returns the estimate, and rounding it to the nearest
integer gives the count. The oracle is given either as mark_data, the list of marked
bit strings, or as a phase flip circuit through flip_operator. counting_qubits sets
the resolution of the phase estimation readout and defaults to qnumber + 2.
"""


def and_oracle(qubits):
    cir = QCircuit()
    cir << TOFFOLI(qubits[0], qubits[1], qubits[3])
    cir << Z(qubits[3])
    cir << TOFFOLI(qubits[0], qubits[1], qubits[3])
    return cir


def main():
    marked = ['001', '011', '111']
    counter = QCounting(qnumber=3, mark_data=marked)
    count = counter.run()

    print('marked states     :', marked)
    print('estimated count   :', round(count, 4))
    print('rounded count     :', int(round(count)))
    print('estimated eigenphase:', round(min(counter.theta, 2 * pi - counter.theta), 4))

    for counting_qubits in (5, 6, 7):
        count = QCounting(qnumber=3, mark_data=marked, counting_qubits=counting_qubits).run()
        print('counting_qubits', counting_qubits, '-> estimate', round(count, 4))

    counter = QCounting(qnumber=3, flip_operator=and_oracle, ancilla_qubits=1)
    count = counter.run()
    print('oracle marking q_0 and q_1:', int(round(count)), 'of', 2 ** 3, 'items')


if __name__ == "__main__":
    main()
