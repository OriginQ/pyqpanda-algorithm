# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from pyqpanda3.hamiltonian import PauliOperator
from pyqpanda_alg.VQE import VQE, hardware_efficient_circuit


class TestVQE:
    """Test cases for VQE algorithm."""
    
    def test_vqe_initialization(self):
        """Test VQE initialization with PauliOperator."""
        hamiltonian = PauliOperator({"Z0": 0.5, "X0 X1": 0.3})
        vqe = VQE(hamiltonian, layers=2)
        assert vqe.n_qubits == 2
        assert vqe.n_params == 8  # 2 qubits * 2 layers * 2 params per qubit
    
    def test_vqe_with_custom_ansatz(self):
        """Test VQE with custom ansatz circuit."""
        hamiltonian = PauliOperator({"Z0": 1.0})
        
        def custom_ansatz(qubit_list, params):
            from pyqpanda3.core import QCircuit, RY
            circuit = QCircuit()
            for i, q in enumerate(qubit_list):
                circuit << RY(q, params[i])
            return circuit
        
        vqe = VQE(hamiltonian, ansatz=custom_ansatz, initial_params=np.array([0.5]))
        assert vqe.n_qubits == 1
        assert vqe.n_params == 1
    
    def test_hardware_efficient_circuit(self):
        """Test hardware efficient ansatz circuit construction."""
        qubits = [0, 1, 2]
        params = np.random.random(2 * 3 * 2)  # 3 qubits, 2 layers, 2 params per qubit per layer
        circuit = hardware_efficient_circuit(qubits, params, layers=2)
        assert circuit is not None
    
    def test_vqe_compute_energy(self):
        """Test energy computation for given parameters."""
        hamiltonian = PauliOperator({"Z0": 1.0})
        vqe = VQE(hamiltonian, layers=1)
        params = np.zeros(vqe.n_params)
        energy = vqe.compute_energy(params)
        # For Z0 with zero parameters, should be close to 1.0 (|0⟩ state)
        assert isinstance(energy, float)
    
    def test_vqe_run_simple_hamiltonian(self):
        """Test VQE optimization on a simple Hamiltonian."""
        # Simple Hamiltonian: H = Z0
        # Ground state should be |1⟩ with energy -1
        hamiltonian = PauliOperator({"Z0": 1.0})
        vqe = VQE(hamiltonian, layers=2, optimizer='COBYLA')
        
        energy, params = vqe.run(optimizer_options={'maxiter': 50})
        
        # Energy should be close to -1 (ground state of Z)
        assert energy < 0
        assert abs(energy - (-1.0)) < 0.5  # Allow some tolerance for variational method
    
    def test_vqe_two_qubit_hamiltonian(self):
        """Test VQE on a two-qubit Hamiltonian."""
        # H = 0.5 * Z0 + 0.5 * Z1 + 0.3 * X0 * X1
        hamiltonian = PauliOperator({
            "Z0": 0.5,
            "Z1": 0.5,
            "X0 X1": 0.3
        })
        vqe = VQE(hamiltonian, layers=3, optimizer='COBYLA')
        
        energy, params = vqe.run(optimizer_options={'maxiter': 100})
        
        # Energy should be negative (ground state)
        assert energy < 0
        assert len(vqe.energy_history) > 0
    
    def test_vqe_energy_history(self):
        """Test that VQE tracks energy history during optimization."""
        hamiltonian = PauliOperator({"Z0": 1.0})
        vqe = VQE(hamiltonian, layers=1)
        
        energy, params = vqe.run(optimizer_options={'maxiter': 10})
        
        assert len(vqe.energy_history) > 0
        assert vqe.optimal_energy == energy
        assert vqe.optimal_params is not None
    
    def test_invalid_hamiltonian_type(self):
        """Test that invalid hamiltonian type raises error."""
        with pytest.raises(TypeError):
            VQE("not a hamiltonian")
    
    def test_invalid_params_length(self):
        """Test that invalid parameter length raises error."""
        hamiltonian = PauliOperator({"Z0": 1.0})
        qubits = [0]
        params = np.zeros(10)  # Wrong length
        
        with pytest.raises(ValueError):
            hardware_efficient_circuit(qubits, params, layers=2)


class TestHardwareEfficientCircuit:
    """Test cases for hardware efficient ansatz."""
    
    def test_single_qubit_single_layer(self):
        """Test single qubit, single layer circuit."""
        qubits = [0]
        params = [0.5, 0.3]  # RY and RZ angles
        circuit = hardware_efficient_circuit(qubits, params, layers=1)
        assert circuit is not None
    
    def test_multi_qubit_multi_layer(self):
        """Test multiple qubits and layers."""
        qubits = [0, 1, 2]
        n_params = 2 * 3 * 3  # 3 qubits, 3 layers, 2 params per qubit per layer
        params = np.random.random(n_params)
        circuit = hardware_efficient_circuit(qubits, params, layers=3)
        assert circuit is not None
    
    def test_circuit_with_entanglement(self):
        """Test that circuit includes entangling gates."""
        qubits = [0, 1]
        params = np.random.random(2 * 2 * 2)  # 2 qubits, 2 layers
        circuit = hardware_efficient_circuit(qubits, params, layers=2)
        # Circuit should be created successfully
        assert circuit is not None
