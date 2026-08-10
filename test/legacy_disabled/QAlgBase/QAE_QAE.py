# import math
#
# import pytest
# import numpy as np
# from pyqpanda_alg.QAE import QAE
# from pyqpanda3.core import QCircuit, RY, X, H
#
#
# def create_cir_basic(qlist):
#     """基础测试量子线路"""
#     cir = QCircuit()
#     cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
#     return cir
#
#
# class Test_QAE_QAE:
#
#     def test_qae_basic_functionality(self):
#         W = QAE(
#             operator_in=create_cir_basic,
#             qnumber=2,
#             epsilon=0.01,
#             res_index=[0, 1],
#             target_state='11'
#         ).run()
#
#         assert W is not None, "振幅估计结果不应为None"
#         assert isinstance(W, (float, np.floating)), f"振幅应为数值类型，实际为{type(W)}"
#         assert 0.2 <= W <= 0.3, f"振幅应在[0,1]范围内，实际为{W}"
#
#
#
# if __name__ == "__main__":
#     # 运行所有测试
#     pytest.main([__file__, "-v", "-s"])
