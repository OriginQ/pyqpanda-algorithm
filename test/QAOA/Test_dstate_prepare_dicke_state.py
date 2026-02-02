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
# class TestPrepareDickeState:
#
#     @classmethod
#     def setup_class(cls):
#         cls.machine = CPUQVM()
#
#     def calculate_hamming_weight(self, number, n):
#         return bin(number).count('1')
#
#     def test_prepare_dicke_state_basic(self):
#         n = 4
#         k = 2
#
#         prog = QProg(n)
#         qubits = prog.qubits()
#         prog << dstate.prepare_dicke_state(qubits, k)
#
#         self.machine.run(prog, 1000)
#         results = self.machine.result().get_prob_list()
#
#         valid_states = 0
#         total_probability = 0
#
#         for key, prob in enumerate(results):
#             key_hmw = self.calculate_hamming_weight(key, n)
#             if key_hmw == k:
#                 valid_states += 1
#                 total_probability += prob
#                 assert prob > 0, f"状态 {bin(key)[2:].zfill(n)} 的概率应为正"
#
#         assert abs(total_probability - 1.0) < 0.1, f"有效状态总概率应为1，实际为 {total_probability}"
#
#
# if __name__ == "__main__":
#     # 直接run test
#     pytest.main([__file__, "-v"])