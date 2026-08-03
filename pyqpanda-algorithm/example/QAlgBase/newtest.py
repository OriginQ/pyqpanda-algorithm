import numpy as np
from pyqpanda3.core import CPUQVM, QCircuit, QProg, RY, X


def create_cir(qlist):
    cir = QCircuit()
    cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    return cir


if __name__ == '__main__':
    m = CPUQVM()
    q_state = QProg(2).qubits()

    prog = QProg()
    prog << create_cir(q_state)

    print(prog)
