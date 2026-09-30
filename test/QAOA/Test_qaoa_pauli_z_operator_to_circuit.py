# import pytest
# import sys
# from pathlib import Path
# import sympy as sp
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import qaoa
# from pyqpanda3.core import CPUQVM, QProg, H
#
#
# class TestPauliZOperatorToCircuit:
#
#     @classmethod
#     def setup_class(cls):
#         cls.machine = CPUQVM()
#
#     def setup_method(self):
#         self.prog = QProg(3)
#         self.qubits = self.prog.qubits()
#
#     def test_basic_functionality(self):
#         # f = 2*x0*x1 + 3*x2 - 1
#         vars = sp.symbols('x0:3')
#         f = 2*vars[0]*vars[1] + 3*vars[2] - 1
#
#         operator = qaoa.problem_to_z_operator(f)
#
#         gamma = 1.0
#         circuit, _ = qaoa.pauli_z_operator_to_circuit(operator, self.qubits, gamma)
#
#         assert circuit is not None
#         originir_str = circuit.originir()
#         assert isinstance(originir_str, str)
#         assert len(originir_str) > 0
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v", "-s"])