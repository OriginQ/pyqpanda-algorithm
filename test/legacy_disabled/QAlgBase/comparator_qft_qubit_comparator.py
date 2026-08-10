# import pytest
# from pyqpanda_alg.QCmp import qft_qubit_comparator
# from pyqpanda3.core import *
#
# class Test_comparator_qft_qubit_comparator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_qft_qubit_comparator_example_from_doc(self):
#         prog = QProg()
#         prog << H(0) << H(1)
#         prog << X(3)
#
#         cir = qft_qubit_comparator([0, 1], [2, 3], [4], function='g')
#         prog << cir
#
#         self.machine.run(prog, 1000)
#         prob_dict_result = self.machine.result().get_prob_dict([4])
#         prob_high = prob_dict_result.get('1', 0.0)
#         expected_prob = 0.5
#         tolerance = 0.3
#
#         assert abs(prob_high - expected_prob) < tolerance, (
#             f"期望概率 {expected_prob:.4f}, 实际概率 {prob_high:.4f}, 超出容忍范围"
#         )
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])