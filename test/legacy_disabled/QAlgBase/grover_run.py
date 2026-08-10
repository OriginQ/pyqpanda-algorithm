# import pytest
# import numpy as np
# from pyqpanda_alg.Grover import GroverAdaptiveSearch
# from pyqpanda3.core import QCircuit, U1
# from pyqpanda_alg.plugin import hadamard_circuit, QFT
#
#
# class Test_grover_run:
#
#     def flip_oracle_function(self, q_index_value, current_min):
#         q_index = q_index_value[:2]
#         q_value = q_index_value[2:]
#         n_value = len(q_value)
#         factor = np.pi * 2 ** (1 - n_value)
#         cal_cir = QCircuit()
#         cal_cir << hadamard_circuit(q_value)
#         for i, q_i in enumerate(q_value):
#             cal_cir << U1(q_i, factor * 2 ** i).control(q_index)
#             cal_cir << U1(q_i, factor * 2 ** i).control(q_index[0])
#             cal_cir << U1(q_i, -factor * 2 ** i).control(q_index[1])
#             cal_cir << U1(q_i, factor * 2 ** i * (-current_min))
#         cal_cir << QFT(q_value).dagger()
#         return cal_cir
#
#     def n_value_function_basic(self, current_min):
#         n_value = 2 if current_min == 0 else 3
#         return n_value
#
#     def value_function_basic(self, var_array):
#         var_array = list(map(int, var_array))[::-1]  # 转换为二进制并反转顺序
#         x0, x1 = var_array[0], var_array[1]
#         value = x0 * x1 + x0 - x1
#         return value
#
#     def calculate_optimal_solution_brute_force(self, value_function, n_bits=2):
#         best_value = float('inf')
#         best_solution = None
#
#         # 遍历所有可能的二进制组合
#         for i in range(2 ** n_bits):
#             binary_str = format(i, f'0{n_bits}b')
#             current_value = value_function(binary_str)
#
#             if current_value < best_value:
#                 best_value = current_value
#                 best_solution = binary_str
#
#         return best_solution, best_value
#
#     def test_run_basic_parameters(self):
#         """测试run接口基本参数"""
#         demo_search = GroverAdaptiveSearch(
#             init_value=0,
#             n_index=2,
#             oracle_circuit=self.flip_oracle_function
#         )
#
#         theoretical_opt_solution, theoretical_opt_value = self.calculate_optimal_solution_brute_force(
#             self.value_function_basic
#         )
#
#         res = demo_search.run(
#             continue_times=5,
#             n_value_function=self.n_value_function_basic,
#             value_function=self.value_function_basic,
#             process_show=True
#         )
#
#         optimal_solution, optimal_value = res[0], res[1]
#         print(f"✓ run接口返回有效结果: 最优解='{optimal_solution}', 函数值={optimal_value}")
#
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])