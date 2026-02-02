# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, QCircuit, RX, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestXYMixer:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def calculate_domain_hamming_weights(self, prob_dict, domains):
#         domain_probs = {}
#         for result_key, prob_value in prob_dict.items():
#             domain_weights = []
#             for domain in domains:
#                 weight = sum(1 for idx in domain if result_key[idx] == '1')
#                 domain_weights.append(weight)
#
#             weight_key = tuple(domain_weights)
#             domain_probs[weight_key] = domain_probs.get(weight_key, 0) + prob_value
#
#         return domain_probs
#
#     def test_xy_mixer_integer_domains_parity(self):
#         n_qubits = 4
#         k_domains = 2
#
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         for q in qubits:
#             prog << RX(q, np.random.random() * np.pi)
#
#         self.machine.run(prog, shots=1)
#         origin_result = self.machine.result().get_prob_dict()
#
#         circuit = default_circuits.xy_mixer(k_domains, 'PXY')(qubits, np.pi/2)
#         prog << circuit
#
#         self.machine.run(prog, shots=1)
#         final_result = self.machine.result().get_prob_dict()
#
#         domain_size = n_qubits // k_domains
#         domains = [list(range(i * domain_size, (i + 1) * domain_size)) for i in range(k_domains)]
#
#         origin_domain_probs = self.calculate_domain_hamming_weights(origin_result, domains)
#         final_domain_probs = self.calculate_domain_hamming_weights(final_result, domains)
#
#         print(f"\nInteger domains ({k_domains}) Parity XY Mixer - Domain Hamming Weight comparison:")
#         for weight_combo in origin_domain_probs:
#             origin_prob = origin_domain_probs[weight_combo]
#             final_prob = final_domain_probs.get(weight_combo, 0)
#             print(f"Domain weights {weight_combo}: {origin_prob:.4f} -> {final_prob:.4f}")
#
#             tolerance = 0.06
#             assert abs(origin_prob - final_prob) < tolerance, \
#                 f"Domain weights {weight_combo} not preserved: {origin_prob:.4f} -> {final_prob:.4f}"
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v", "-s"])