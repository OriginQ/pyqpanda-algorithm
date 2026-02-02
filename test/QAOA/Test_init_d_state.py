# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestInitDState:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def calculate_domain_hamming_weights(self, result_key, domains):
#         domain_weights = []
#         for domain in domains:
#             weight = sum(1 for idx in domain if result_key[idx] == '1')
#             domain_weights.append(weight)
#         return domain_weights
#
#     def test_init_d_state_integer_domains(self):
#         n_qubits = 6
#         k = 2 
#         domains = 2  
#
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         init_circuit_func = default_circuits.init_d_state(domains, k)
#         init_circuit = init_circuit_func(qubits)
#         prog << init_circuit
#
#         self.machine.run(prog, shots=1000)
#         results = self.machine.result().get_prob_dict(qubits)
#
#         domain_size = n_qubits // domains
#         domain_list = [list(range(i * domain_size, (i + 1) * domain_size)) for i in range(domains)]
#
#         print(f"\nInteger domains ({domains}) Dicke state k={k} - Measurement results:")
#         valid_states = 0
#
#         for key, prob in results.items():
#             if prob > 0.001: 
#                 key_reversed = key[::-1] 
#                 domain_weights = self.calculate_domain_hamming_weights(key_reversed, domain_list)
#
#                 print(f"State {key_reversed}: domain weights = {domain_weights}, prob = {prob:.4f}")
#
#                 for weight in domain_weights:
#                     assert weight == k, f"Domain weight should be {k}, but got {weight} for state {key_reversed}"
#
#                 valid_states += 1
#
#         assert valid_states > 0, "No valid Dicke states found"
#
#
# if __name__ == "__main__":
#     # run test
#     pytest.main([__file__, "-v", "-s"])