# import pytest
#
# from pyqpanda_alg.Grover import Grover
# from pyqpanda3.core import CPUQVM, QCircuit, QProg, TOFFOLI, Z
#
#
# def mark(qubits):
#     """标记函数：标记 |11⟩ 状态"""
#     cir = QCircuit()
#     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
#     cir << Z(qubits[2])
#     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
#     return cir
#
#
# class Test_grover_cir:
#
#     def setup_method(self):
#         self.qvm = CPUQVM()
#
#     def test_cir_basic_functionality(self):
#         q_state = QProg(3).qubits()
#         demo_search = Grover(flip_operator=mark)
#         circuit = demo_search.cir(
#             q_input=q_state[:2],
#             q_flip=q_state,
#             q_zero=q_state[:2],
#             iternum=1
#         )
#         print(f"电路内容: {circuit}")
#         print(circuit.originir())
#         assert circuit.originir() == """QINIT 3
# CREG 1
# H q[0]
# H q[1]
# CONTROL q[0],q[1]
# X q[2]
# ENDCONTROL
# Z q[2]
# CONTROL q[0],q[1]
# X q[2]
# ENDCONTROL
# DAGGER
# H q[1]
# H q[0]
# ENDDAGGER
#
# X q[0]
# X q[1]
# Z q[1]
# ENDCONTROL
#
# X q[0]
# X q[1]
# H q[0]
# H q[1]
# """
#         assert isinstance(circuit, QCircuit), "cir方法应返回QCircuit对象"
#         assert circuit is not None, "返回的电路不应为None"
#
#         prog = QProg()
#         prog << circuit
#         assert isinstance(prog, QProg), "电路应能正常添加到量子程序中"
#         assert len(str(prog)) > 0, "量子程序应有内容"
#
#
#
# if __name__ == "__main__":
#     # 直接运行所有测试
#     pytest.main([__file__, "-v"])
