# import pytest
# import sys
# from pathlib import Path
# import sympy as sp
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA.qaoa import QAOA, p_1
# from pyqpanda3.hamiltonian import PauliOperator
#
#
# class TestCalculateEnergy:
#
#     def test_basic_functionality_with_symbolic_problem(self):
#         # f = 2*x0*x1 + 3*x2 - 1
#         vars = sp.symbols('x0:3')
#         f = 2*vars[0]*vars[1] + 3*vars[2] - 1
#
#         qaoa_f = QAOA(f)
#
#         test_cases = [
#             ([1, 0, 0], 2*1*0 + 3*0 - 1),  # f(1,0,0) = -1
#             ([0, 1, 1], 2*0*1 + 3*1 - 1),  # f(0,1,1) = 2
#             ([1, 1, 0], 2*1*1 + 3*0 - 1),  # f(1,1,0) = 1
#             ([0, 0, 0], 2*0*0 + 3*0 - 1),  # f(0,0,0) = -1
#             ([1, 1, 1], 2*1*1 + 3*1 - 1),  # f(1,1,1) = 4
#         ]
#
#         for solution, expected_energy in test_cases:
#             calculated_energy = qaoa_f.calculate_energy(solution)
#             assert abs(calculated_energy - expected_energy) < 1e-10, \
#                 f"solve {solution} error : expect {expected_energy}, get {calculated_energy}"
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v"])