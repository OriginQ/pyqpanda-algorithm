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

"""
VQE (Variational Quantum Eigensolver) Example

This example demonstrates how to use the VQE algorithm to find the ground state
energy of a simple Hamiltonian.
"""

from pyqpanda_alg import VQE
from pyqpanda3.hamiltonian import PauliOperator
import numpy as np


def example_simple_hamiltonian():
    """
    Example 1: Simple single-qubit Hamiltonian
    
    Hamiltonian: H = Z
    Ground state: |1⟩ with energy -1
    """
    print("=" * 60)
    print("Example 1: Simple Single-Qubit Hamiltonian")
    print("=" * 60)
    
    # Define Hamiltonian: H = Z
    hamiltonian = PauliOperator({"Z0": 1.0})
    
    # Create VQE solver
    vqe = VQE.VQE(hamiltonian, layers=2, optimizer='COBYLA')
    
    # Run VQE
    energy, params = vqe.run(optimizer_options={'maxiter': 100})
    
    print(f"Hamiltonian: H = Z")
    print(f"Theoretical ground state energy: -1.0")
    print(f"VQE found energy: {energy:.6f}")
    print(f"Error: {abs(energy - (-1.0)):.6f}")
    print()


def example_two_qubit_hamiltonian():
    """
    Example 2: Two-qubit Hamiltonian with interaction
    
    Hamiltonian: H = 0.5*Z0 + 0.5*Z1 + 0.3*X0*X1
    """
    print("=" * 60)
    print("Example 2: Two-Qubit Hamiltonian with Interaction")
    print("=" * 60)
    
    # Define Hamiltonian
    hamiltonian = PauliOperator({
        "Z0": 0.5,
        "Z1": 0.5,
        "X0 X1": 0.3
    })
    
    # Create VQE solver with more layers for better accuracy
    vqe = VQE.VQE(hamiltonian, layers=3, optimizer='COBYLA')
    
    # Run VQE
    energy, params = vqe.run(optimizer_options={'maxiter': 200})
    
    print(f"Hamiltonian: H = 0.5*Z0 + 0.5*Z1 + 0.3*X0*X1")
    print(f"VQE found energy: {energy:.6f}")
    print(f"Number of optimization steps: {len(vqe.energy_history)}")
    print()


def example_custom_ansatz():
    """
    Example 3: Using custom ansatz circuit
    """
    print("=" * 60)
    print("Example 3: Custom Ansatz Circuit")
    print("=" * 60)
    
    from pyqpanda3.core import QCircuit, RY, RZ, CNOT
    
    # Define custom ansatz
    def custom_ansatz(qubit_list, params):
        circuit = QCircuit()
        n_qubits = len(qubit_list)
        
        # First layer: RY rotations
        for i, q in enumerate(qubit_list):
            circuit << RY(q, params[i])
        
        # Entangling layer
        for i in range(n_qubits - 1):
            circuit << CNOT(qubit_list[i], qubit_list[i + 1])
        
        # Second layer: RZ rotations
        for i, q in enumerate(qubit_list):
            circuit << RZ(q, params[n_qubits + i])
        
        return circuit
    
    # Define Hamiltonian
    hamiltonian = PauliOperator({
        "Z0": 1.0,
        "Z1": 1.0,
        "X0 X1": 0.5
    })
    
    # Create VQE with custom ansatz
    initial_params = np.random.uniform(-np.pi, np.pi, 4)  # 2 qubits * 2 params
    vqe = VQE.VQE(
        hamiltonian,
        ansatz=custom_ansatz,
        initial_params=initial_params,
        optimizer='COBYLA'
    )
    
    # Run VQE
    energy, params = vqe.run(optimizer_options={'maxiter': 150})
    
    print(f"Custom ansatz with RY-CNOT-RZ structure")
    print(f"VQE found energy: {energy:.6f}")
    print()


def example_energy_convergence():
    """
    Example 4: Track energy convergence during optimization
    """
    print("=" * 60)
    print("Example 4: Energy Convergence Tracking")
    print("=" * 60)
    
    # Define Hamiltonian
    hamiltonian = PauliOperator({
        "Z0": 1.0,
        "X0": 0.5
    })
    
    # Create VQE solver
    vqe = VQE.VQE(hamiltonian, layers=2, optimizer='COBYLA')
    
    # Run VQE
    energy, params = vqe.run(optimizer_options={'maxiter': 100})
    
    print(f"Hamiltonian: H = Z0 + 0.5*X0")
    print(f"Final energy: {energy:.6f}")
    print(f"Optimization steps: {len(vqe.energy_history)}")
    print(f"Initial energy: {vqe.energy_history[0]:.6f}")
    print(f"Final energy: {vqe.energy_history[-1]:.6f}")
    print(f"Energy reduction: {vqe.energy_history[0] - vqe.energy_history[-1]:.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VQE (Variational Quantum Eigensolver) Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_simple_hamiltonian()
    example_two_qubit_hamiltonian()
    example_custom_ansatz()
    example_energy_convergence()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
