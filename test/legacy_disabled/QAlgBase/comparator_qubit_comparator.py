# import pytest
# from pyqpanda_alg.QCmp import qubit_comparator
# from pyqpanda3.core import *
#
# class Test_comparator_qubit_comparator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_qubit_comparator_example_from_doc(self):
#
#         prog = QProg()
#         prog << H(0) << H(1)
#         prog << X(3)
#
#         cir = qubit_comparator(
#             q_state_1=[0, 1],
#             q_state_2=[2, 3],
#             q_anc_cmp=[4, 5],
#             function='g'
#         )
#         prog << cir
#
#         self.machine.run(prog, shots=1000)
#         result = self.machine.result().get_prob_dict([5])
#         prob_high = result.get('1', 0.0)
#         expected_prob = 0.25
#         tolerance = 0.05
#         assert abs(prob_high - expected_prob) < tolerance, (
#             f"期望概率 {expected_prob:.4f}, 实际概率 {prob_high:.4f}, 超出容忍范围"
#         )
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])