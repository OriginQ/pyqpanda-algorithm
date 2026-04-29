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
# class TestP0Interface:
#
#     def test_p0_basic_functionality(self):
#         operator_0 = qaoa.p_0(0)
#
#         assert isinstance(operator_0, PauliOperator)
#
#         assert str(operator_0) is not None
#         assert len(str(operator_0)) > 0
#
#         print(f"p_0(0) = {operator_0}")
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])