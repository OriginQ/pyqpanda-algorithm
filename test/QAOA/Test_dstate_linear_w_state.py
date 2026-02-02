# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import dstate
# from pyqpanda3.core import CPUQVM, QProg
# from pyqpanda3.intermediate_compiler import convert_qprog_to_originir
#
#
# class TestLinearWState:
#     """linear_w_state接口测试类"""
#
#     def calculate_hamming_weight(self, number):
#         """计算数字的汉明权重"""
#         return bin(number).count('1')
#
#     def test_linear_w_state_basic_functionality(self):
#         """测试基础功能：n=3, compress=True"""
#         n = 3
#         machine = CPUQVM()
#
#         prog = QProg(n)
#         qubits = prog.qubits()
#         prog << dstate.linear_w_state(qubits, compress=True)
#
#         # 运行量子程序
#         shots = 1000
#         machine.run(prog, shots)
#         results = machine.result().get_prob_list()
#
#         # 验证结果：只有汉明权重为1的状态有概率
#         valid_states = 0
#         total_probability = 0
#
#         for key, prob in enumerate(results):
#             key_hmw = self.calculate_hamming_weight(key)
#             if key_hmw == 1:
#                 valid_states += 1
#                 total_probability += prob
#                 # 验证概率应该在合理范围内
#                 assert prob > 0, f"状态 {bin(key)[2:].zfill(n)} 的概率应为正"
#             else:
#                 # 汉明权重不为1的状态概率应该为0（允许小的浮点误差）
#                 assert abs(prob) < 1e-5, f"状态 {bin(key)[2:].zfill(n)} 的概率应为0，实际为 {prob}"
#
#         # 验证有效状态数量（应该有n个状态，每个状态只有一个1）
#         expected_states = n
#         assert valid_states == expected_states, f"应有 {expected_states} 个有效状态，实际有 {valid_states} 个"
#
#         # 验证总概率（考虑统计误差）
#         assert abs(total_probability - 1.0) < 0.1, f"有效状态总概率应为1，实际为 {total_probability}"
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v"])