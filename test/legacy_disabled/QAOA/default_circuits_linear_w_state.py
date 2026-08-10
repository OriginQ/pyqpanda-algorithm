# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# # 添加项目路径到系统路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import default_circuits
# from pyqpanda3.core import QProg, CPUQVM
#
#
# class TestLinearWState:
#     """linear_w_state接口测试类"""
#
#     def setup_method(self):
#         """测试方法前置设置"""
#         self.machine = CPUQVM()
#
#     def test_linear_w_state_basic_compressed(self):
#         n = 3
#         prog = QProg(n)
#         qubits = prog.qubits()
#
#         # 准备W态（压缩模式）
#         prog << default_circuits.linear_w_state(qubits, compress=True)
#
#         # 运行量子程序
#         self.machine.run(prog, 1000)
#         results = self.machine.result().get_prob_list(qubits)
#
#         # 验证结果：只有汉明权重为1的基态有概率
#         total_prob = 0.0
#         valid_states_count = 0
#
#         for key in range(2**n):
#             prob = results[key]
#             key_hmw = bin(key).count('1')
#
#             if key_hmw == 1:
#                 # 汉明权重为1的态应该有概率，且概率应该相等
#                 assert prob > 0, f"状态 {bin(key)[2:].zfill(n)} 应该有概率，但实际为0"
#                 total_prob += prob
#                 valid_states_count += 1
#             else:
#                 # 汉明权重不为1的态概率应该为0（允许小的浮点误差）
#                 assert abs(prob) < 1e-10, f"状态 {bin(key)[2:].zfill(n)} 概率应该为0，但实际为{prob}"
#
#         # 验证总概率约为1（允许小的浮点误差）
#         assert abs(total_prob - 1.0) < 1e-10, f"总概率应该为1，但实际为{total_prob}"
#
#         # 验证有效状态数量正确（应该有n个状态）
#         assert valid_states_count == n, f"有效状态数量应该为{n}，但实际为{valid_states_count}"
#
#         # 验证所有有效状态的概率大致相等
#         expected_prob = 1.0 / n
#         for key in range(2**n):
#             prob = results[key]
#             key_hmw = bin(key).count('1')
#             if key_hmw == 1:
#                 assert abs(prob - expected_prob) < 1e-10, \
#                     f"状态 {bin(key)[2:].zfill(n)} 概率应该为{expected_prob}，但实际为{prob}"
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v"])