# import pytest
# from pyqpanda_alg.Grover import mark_data_reflection
# from pyqpanda_alg.Grover import Grover
# from pyqpanda3.core import CPUQVM, QProg
#
#
# class Test_grover_mark_data_reflection:
#
#     def test_mark_data_reflection_basic(self):
#         m = CPUQVM()
#         q_state = QProg(3).qubits()
#
#         def mark(qubits):
#             return mark_data_reflection(qubits=qubits, mark_data=['101', '001'])
#
#         demo_search = Grover(flip_operator=mark)
#         prog = QProg()
#         prog << demo_search.cir(q_input=q_state)
#
#         m.run(prog, shots=1000)
#         res = m.result().get_prob_dict()
#         assert '101' in res, "目标态 '101' 应该在结果中"
#         assert '001' in res, "目标态 '001' 应该在结果中"
#
#         target_prob = res.get('101', 0) + res.get('001', 0)
#         assert target_prob > 0.1, f"两个目标态的总概率应该显著高于随机，当前为: {target_prob}"
#
#         total_prob = sum(res.values())
#         assert abs(total_prob - 1.0) < 0.01, f"概率总和应该为1，当前为: {total_prob}"
#
# if __name__ == "__main__":
#     # 可以直接运行测试
#     pytest.main([__file__, "-v", "-s"])