from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda_alg.QAOA.qaoa import QAOA


def test_qaoa_accepts_hamiltonian():
    model = QAOA(Hamiltonian({"Z0": 1.0}))
    assert model.problem_dimension == 1
