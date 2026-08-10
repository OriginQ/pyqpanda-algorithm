# import math
#
# import pytest
# from pyqpanda_alg.QCmp import int_comparator
# from pyqpanda3.core import *
#
#
# class Test_comparator_int_comparator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_comparator_example_from_doc(self):
#         prog = QProg()
#         prog << H(0) << H(1)
#
#         # 测试该态和2相比的大小
#         cir = int_comparator(2, [0, 1], [2, 3], function='g', reuse=True)
#         prog << cir
#
#         self.machine.run(prog, 1000)
#         prob_dict_result = self.machine.result().get_prob_dict([3])
#
#         prob_high = prob_dict_result.get('1', 0.0)
#         assert math.isclose(prob_high, 0.25)
#         assert 0 <= prob_high <= 1, f"概率应在[0,1]范围内，实际: {prob_high}"
#         assert len(prob_dict_result) > 0, "应该返回概率分布"
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])