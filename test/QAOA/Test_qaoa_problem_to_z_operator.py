# import sys
# from pathlib import Path
# import pytest
# import sympy as sp
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import qaoa
# from pyqpanda3.hamiltonian import PauliOperator
#
#
# class TestProblemToZOperator:
#
#     def test_basic_functionality(self):
#         vars = sp.symbols('x0:3')
#         f = 2*vars[0]*vars[1] + 3*vars[2] - 1
#
#         hamiltonian = qaoa.problem_to_z_operator(f)
#
#         assert isinstance(hamiltonian, PauliOperator)
#
#         assert str(hamiltonian) is not None
#         assert len(str(hamiltonian)) > 0
#         print(f"{f}")
#         print(f"{hamiltonian}")
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v", "-s"])