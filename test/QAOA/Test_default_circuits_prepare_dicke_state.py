# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import default_circuits
# from pyqpanda3.core import QProg, CPUQVM
#
#
# class TestPrepareDickeState:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_prepare_dicke_state_basic(self):
#         n = 4
#         k = 2
#         prog = QProg(n)
#         qubits = prog.qubits()
#
#         prog << default_circuits.prepare_dicke_state(qubits, k)
#
#         self.machine.run(prog, 1000)
#         results = self.machine.result().get_prob_list(qubits)
#
#         total_prob = 0.0
#         valid_states_count = 0
#
#         for key in range(2**n):
#             prob = results[key]
#             key_hmw = bin(key).count('1')
#
#             if key_hmw == k:
#                 assert prob > 0, f"状态 {bin(key)[2:].zfill(n)} 应该有概率，但实际为0"
#                 total_prob += prob
#                 valid_states_count += 1
#             else:
#                 assert abs(prob) < 1e-10, f"状态 {bin(key)[2:].zfill(n)} 概率应该为0，但实际为{prob}"
#
#         assert abs(total_prob - 1.0) < 1e-10, f"总概率应该为1，但实际为{total_prob}"
#
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v"])