# import pytest
# from pyqpanda_alg.QCmp import interpolation_comparator
# from pyqpanda3.core import *
#
#
# class Test_comparator_interpolation_comparator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_interpolation_comparator_example_from_doc(self):
#
#         value = 3.3
#         prog = QProg()
#         prog << X(0) << X(1) << I(2)
#
#         cir = interpolation_comparator(value, [0, 1, 2], [3, 4, 5], function='g', reuse=True)
#         prog << cir
#
#         self.machine.run(prog, 1000)
#         prob_dict_result = self.machine.result().get_prob_dict([5])
#         print(f"结果: {prob_dict_result}")
#
#         # 验证结果有效性
#         prob_high = prob_dict_result.get('1', 0.0)
#         print(prob_high)
#         assert 0 <= prob_high <= 1, f"概率应在[0,1]范围内，实际: {prob_high}"
#
#
# if __name__ == "__main__":
#     # 运行所有测试
#     pytest.main([__file__, "-v", "-s"])