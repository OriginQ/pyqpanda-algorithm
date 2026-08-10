# import pytest
# import sympy as sp
# from pyqpanda_alg.QUBO import QUBO
# from pyqpanda3.core import CPUQVM, QProg, QCircuit
#
#
# class Test_QUBO_cir:
#
#     def setup_method(self):
#         self.m = CPUQVM()
#
#     def test_cir_basic_functionality(self):
#
#         x0, x1, x2 = sp.symbols('x0 x1 x2')
#         function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
#         test0 = QUBO.QuadraticBinary(function)
#         n_key, n_res = test0.query_qnumber()
#         q_key = QProg(n_key + n_res).qubits()
#         circuit = test0.cir(q_key[:n_key], q_key[n_key:])
#         print(circuit.originir())
#         print(circuit.qubits())
#         assert len(circuit.qubits()) == n_key + n_res, \
#             f"线路应使用{n_key + n_res}个比特，实际使用{len(circuit.qubits())}个"
#         assert circuit is not None, "cir接口应该返回非None的量子线路"
#         assert isinstance(circuit, QCircuit), f"返回的应该是QCircuit类型，实际是{type(circuit)}"
#
#         print("✓ 基本功能测试通过")
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])