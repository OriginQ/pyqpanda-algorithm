# import pytest
# from pyqpanda_alg.QCmp import qft_comparator
# from pyqpanda3.core import *
#
#
# class Test_comparator_qft_comparator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_qft_comparator_example_from_doc(self):
#
#         value = 2
#         m = CPUQVM()
#
#         prog = QProg()
#         prog << H(0) << H(1)
#
#         # 测试该态和2相比的大小
#         cir = qft_comparator(value, [0, 1], [2], function='g')
#         prog << cir
#
#         m.run(prog, 1000)
#         prob_dict_result = m.result().get_prob_dict([2])
#         print(f"结果: {prob_dict_result}")
#
#         # 验证结果有效性
#         prob_high = prob_dict_result.get('1', 0.0)
#         assert 0 <= prob_high <= 1, f"概率应在[0,1]范围内，实际: {prob_high}"
#         assert len(prob_dict_result) > 0, "应该返回概率分布"
#
# if __name__ == "__main__":
#     # 运行所有测试
#     pytest.main([__file__, "-v", "-s"])
