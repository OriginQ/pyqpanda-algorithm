# import sys
# from pathlib import Path
# import pytest
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import qaoa
# from pyqpanda3.hamiltonian import PauliOperator
#
#
# class TestP1Interface:
#
#     def test_p1_basic_functionality(self):
#         operator_0 = qaoa.p_1(0)
#         operator_1 = qaoa.p_1(1)
#         operator_2 = qaoa.p_1(2)
#
#         assert isinstance(operator_0, PauliOperator)
#         assert isinstance(operator_1, PauliOperator)
#         assert isinstance(operator_2, PauliOperator)
#         assert str(operator_0) != str(operator_1)
#         assert str(operator_1) != str(operator_2)
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])