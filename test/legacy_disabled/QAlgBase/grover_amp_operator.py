# import pytest
# from pyqpanda_alg.Grover import amp_operator
# from pyqpanda3.core import CPUQVM, QCircuit, QProg, TOFFOLI, Z, H, X, measure
#
#
# class Test_grover_amp_operator:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#         self.qubits = QProg(3).qubits()
#
#     def create_mark_operator(self):
#         def mark(qubits):
#             cir = QCircuit()
#             cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
#             cir << Z(qubits[2])
#             cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
#             return cir
#
#         return mark
#
#     def test_amp_operator_basic(self):
#         mark_operator = self.create_mark_operator()
#
#         circuit = amp_operator(
#             q_input=self.qubits[:2],
#             q_flip=self.qubits,
#             q_zero=self.qubits[:2],
#             flip_operator=mark_operator
#         )
#
#         assert isinstance(circuit, QCircuit), "amp_operator应返回QCircuit对象"
#         assert circuit is not None, "返回的量子线路不应为None"
#
#         circuit_str = str(circuit)
#         print(circuit_str)
#
#         assert len(circuit_str) > 0, "量子线路应有内容"
#
#         prog = QProg()
#         prog << circuit
#
#         try:
#             # 尝试执行线路
#             self.machine.run(prog, shots=1000)
#             result = self.machine.result().get_prob_dict(self.qubits[:2])
#             print(result)
#             assert result is not None, "线路应能正常执行"
#         except Exception as e:
#             pytest.fail(f"线路执行失败: {e}")
#
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])
