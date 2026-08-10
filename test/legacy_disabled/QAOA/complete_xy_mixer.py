# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, RX, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestCompleteXYMixer:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def calculate_hamming_weight_distribution(self, prob_dict, max_weight):
#         weight_probs = {}
#         for weight in range(max_weight + 1):
#             prob = sum(value for key, value in prob_dict.items()
#                       if key.count('1') == weight)
#             weight_probs[weight] = prob
#         return weight_probs
#
#     def test_complete_xy_mixer_hamming_weight_preservation(self):
#         n_qubits = 4
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         for q in qubits:
#             prog << RX(q, np.random.random() * 2 * np.pi)
#
#         self.machine.run(prog, shots=1)
#         original_result = self.machine.result().get_prob_dict()
#
#         original_weight_probs = self.calculate_hamming_weight_distribution(original_result, n_qubits)
#
#         # 应用 complete_xy_mixer
#         beta = np.pi / 5
#         circuit = default_circuits.complete_xy_mixer(qubits, beta)
#         prog << circuit
#
#         self.machine.run(prog, shots=1)
#         final_result = self.machine.result().get_prob_dict()
#
#         final_weight_probs = self.calculate_hamming_weight_distribution(final_result, n_qubits)
#         print("\nHamming weight distribution comparison:")
#         for weight in range(n_qubits + 1):
#             original_prob = original_weight_probs.get(weight, 0)
#             final_prob = final_weight_probs.get(weight, 0)
#             print(f"Hamming weight {weight}: {original_prob:.4f} -> {final_prob:.4f}")
#
#             tolerance = 0.05
#             assert abs(original_prob - final_prob) < tolerance, \
#                 f"Hamming weight {weight} not preserved within tolerance: {original_prob:.4f} -> {final_prob:.4f}"
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])