# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import default_circuits
# from pyqpanda3.core import QProg, RX, RY, RZ, CPUQVM
#
#
# class TestParityPartitionXYMixer:
#
#     def setup_method(self):
#         self.machine = CPUQVM()
#
#     def test_parity_partition_xy_mixer_spin_conservation(self):
#         for n_qubits in [2, 3, 4, 5]:
#             prog = QProg(n_qubits)
#             qubits = prog.qubits()
#             beta = np.pi / 6
#
#             for q in qubits:
#                 prog << RX(q, np.random.random() * 2 * np.pi)
#
#             self.machine.run(prog, shots=1000)
#             original_result = self.machine.result().get_prob_dict()
#             print(original_result)
#             circuit = default_circuits.parity_partition_xy_mixer(qubits, beta)
#             prog << circuit
#
#             self.machine.run(prog, shots=1000)
#             final_result = self.machine.result().get_prob_dict()
#             print(final_result)
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])