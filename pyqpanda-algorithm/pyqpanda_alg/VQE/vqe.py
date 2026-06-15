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

from pyqpanda3.core import CPUQVM, QCircuit, QProg
from pyqpanda3.hamiltonian import PauliOperator, Hamiltonian
import numpy as np
from scipy.optimize import minimize
from typing import Union, Callable, Optional
from .ansatz import hardware_efficient_circuit


class VQE:
    """
    Variational Quantum Eigensolver (VQE) algorithm for finding the ground state energy
    of a Hamiltonian.
    
    VQE is a hybrid quantum-classical algorithm that uses a parameterized quantum circuit
    (ansatz) to prepare trial states, and a classical optimizer to minimize the energy
    expectation value. It is particularly suitable for near-term quantum devices.
    
    Parameters
        hamiltonian : ``PauliOperator`` or ``Hamiltonian``\n
            The Hamiltonian whose ground state energy we want to find.
        ansatz : ``callable``, ``optional``\n
            A function that constructs the parameterized quantum circuit (ansatz).
            Should accept (qubit_list, params) and return a QCircuit.
            Default is hardware_efficient_circuit.
        layers : ``int``, ``optional``\n
            Number of layers in the default hardware-efficient ansatz. Default is 1.
            Ignored if custom ansatz is provided.
        optimizer : ``str``, ``optional``\n
            Classical optimizer to use. Should be one of scipy.optimize.minimize methods.
            Default is 'COBYLA'.
        initial_params : ``array-like``, ``optional``\n
            Initial parameters for the ansatz circuit. If None, random initialization is used.
    
    Attributes
        n_qubits : ``int``\n
            Number of qubits in the system.
        energy_history : ``list``\n
            List of energy values during optimization.
        optimal_params : ``array-like``\n
            Optimized parameters after running VQE.
        optimal_energy : ``float``\n
            Ground state energy found by VQE.
    
    Methods
        run : Run the VQE algorithm to find the ground state energy.
        compute_energy : Compute the energy expectation value for given parameters.
    
    References
        [1] Peruzzo A, McClean J, Fowler P, et al. A variational eigenvalue solver on a 
        photonic quantum processor[J]. Nature Communications, 2014, 5: 4213.
        https://doi.org/10.1038/ncomms5213
        
        [2] Cerezo M, Arrasmith A, Babbush R, et al. Variational quantum algorithms[J]. 
        Nature Reviews Physics, 2021, 3(9): 625-644.
        https://doi.org/10.1038/s42254-021-00348-9
    
    Examples
        >>> from pyqpanda_alg import VQE
        >>> from pyqpanda3.hamiltonian import PauliOperator
        >>> # Define a simple Hamiltonian: H = 0.5 * Z0 + 0.3 * X0 * X1
        >>> hamiltonian = PauliOperator({"Z0": 0.5, "X0 X1": 0.3})
        >>> # Create VQE solver
        >>> vqe = VQE.VQE(hamiltonian, layers=2)
        >>> # Run VQE
        >>> energy, params = vqe.run()
        >>> print(f"Ground state energy: {energy}")
    """
    
    def __init__(self, 
                 hamiltonian: Union[PauliOperator, Hamiltonian],
                 ansatz: Optional[Callable] = None,
                 layers: int = 1,
                 optimizer: str = 'COBYLA',
                 initial_params: Optional[np.ndarray] = None):
        
        # Process hamiltonian
        if isinstance(hamiltonian, Hamiltonian):
            self.hamiltonian = hamiltonian.pauli_operator()
        elif isinstance(hamiltonian, PauliOperator):
            self.hamiltonian = hamiltonian
        else:
            raise TypeError("hamiltonian must be a PauliOperator or Hamiltonian")
        
        # Determine number of qubits
        self.n_qubits = self._get_n_qubits(self.hamiltonian)
        
        # Set up ansatz
        if ansatz is None:
            self.ansatz = hardware_efficient_circuit
            self.layers = layers
            self.n_params = 2 * self.n_qubits * layers
        else:
            self.ansatz = ansatz
            self.layers = None
            # Try to infer number of parameters
            test_params = np.zeros(100)  # Large enough for most cases
            try:
                test_circuit = self.ansatz(list(range(self.n_qubits)), test_params[:self.n_qubits*2])
                self.n_params = len(test_params[:self.n_qubits*2])
            except:
                # If inference fails, user must provide initial_params
                self.n_params = None
        
        # Set up optimizer
        self.optimizer = optimizer
        
        # Set initial parameters
        if initial_params is not None:
            self.initial_params = np.array(initial_params)
            if self.n_params is None:
                self.n_params = len(self.initial_params)
        else:
            if self.n_params is None:
                raise ValueError("Cannot determine number of parameters. Please provide initial_params.")
            self.initial_params = np.random.uniform(-np.pi, np.pi, self.n_params)
        
        # Initialize tracking variables
        self.energy_history = []
        self.optimal_params = None
        self.optimal_energy = None
        
        # Initialize quantum machine
        self.machine = CPUQVM()
    
    def _get_n_qubits(self, hamiltonian: PauliOperator) -> int:
        """Extract the number of qubits from the hamiltonian."""
        max_qubit = -1
        for term in hamiltonian.terms():
            for pauli in term.paulis():
                qubit_idx = pauli.qbit()
                if qubit_idx > max_qubit:
                    max_qubit = qubit_idx
        return max_qubit + 1
    
    def _build_circuit(self, params: np.ndarray) -> QCircuit:
        """Build the parameterized quantum circuit."""
        qubit_list = list(range(self.n_qubits))
        return self.ansatz(qubit_list, params)
    
    def compute_energy(self, params: np.ndarray) -> float:
        """
        Compute the energy expectation value <ψ(θ)|H|ψ(θ)> for given parameters.
        
        Parameters
            params : ``array-like``\n
                Parameters for the ansatz circuit.
        
        Returns
            energy : ``float``\n
                Energy expectation value.
        
        Examples
            >>> from pyqpanda_alg import VQE
            >>> from pyqpanda3.hamiltonian import PauliOperator
            >>> import numpy as np
            >>> hamiltonian = PauliOperator({"Z0": 0.5, "X0 X1": 0.3})
            >>> vqe = VQE.VQE(hamiltonian, layers=2)
            >>> params = np.random.random(vqe.n_params)
            >>> energy = vqe.compute_energy(params)
            >>> print(f"Energy: {energy}")
        """
        # Build the ansatz circuit
        ansatz_circuit = self._build_circuit(params)
        
        # Calculate energy expectation value
        energy = 0.0
        
        for term in self.hamiltonian.terms():
            coef = term.coef().real
            paulis = term.paulis()
            
            if len(paulis) == 0:
                # Identity term
                energy += coef
                continue
            
            # Build measurement circuit for this term
            prog = QProg(self.n_qubits)
            qubit_list = prog.qubits()
            
            # Apply ansatz
            prog << ansatz_circuit
            
            # Apply basis rotation for measurement
            measurement_circuit = self._get_measurement_circuit(paulis, qubit_list)
            prog << measurement_circuit
            
            # Run and get probabilities
            self.machine.run(prog, shots=1)
            prob_dict = self.machine.result().get_prob_dict(qubit_list)
            
            # Calculate expectation value
            expectation = self._calculate_expectation(prob_dict, paulis)
            energy += coef * expectation
        
        self.energy_history.append(energy)
        return energy
    
    def _get_measurement_circuit(self, paulis, qubit_list) -> QCircuit:
        """Get the circuit to rotate to the measurement basis."""
        circuit = QCircuit()
        
        for pauli in paulis:
            qubit_idx = pauli.qbit()
            pauli_type = pauli.pauli_char()
            
            if pauli_type == 'X':
                # Rotate from X basis to Z basis
                from pyqpanda3.core import H
                circuit << H(qubit_list[qubit_idx])
            elif pauli_type == 'Y':
                # Rotate from Y basis to Z basis
                from pyqpanda3.core import RX
                circuit << RX(qubit_list[qubit_idx], -np.pi / 2)
            # Z basis doesn't need rotation
        
        return circuit
    
    def _calculate_expectation(self, prob_dict: dict, paulis) -> float:
        """Calculate the expectation value from measurement probabilities."""
        expectation = 0.0
        
        for state, prob in prob_dict.items():
            # Calculate parity for the measured qubits
            parity = 1
            for pauli in paulis:
                qubit_idx = pauli.qbit()
                # State string is in reverse order (qubit 0 is rightmost)
                bit = int(state[-(qubit_idx + 1)])
                parity *= (-1) ** bit
            
            expectation += prob * parity
        
        return expectation
    
    def run(self, optimizer_options: Optional[dict] = None) -> tuple:
        """
        Run the VQE algorithm to find the ground state energy.
        
        Parameters
            optimizer_options : ``dict``, ``optional``\n
                Options to pass to the scipy optimizer.
        
        Returns
            optimal_energy : ``float``\n
                The ground state energy found by VQE.
            optimal_params : ``array-like``\n
                The optimized parameters for the ansatz circuit.
        
        Examples
            >>> from pyqpanda_alg import VQE
            >>> from pyqpanda3.hamiltonian import PauliOperator
            >>> hamiltonian = PauliOperator({"Z0": 0.5, "X0 X1": 0.3})
            >>> vqe = VQE.VQE(hamiltonian, layers=2)
            >>> energy, params = vqe.run()
            >>> print(f"Ground state energy: {energy}")
            >>> print(f"Optimal parameters: {params}")
        """
        if optimizer_options is None:
            optimizer_options = {}
        
        # Clear history
        self.energy_history = []
        
        # Define objective function
        def objective(params):
            return self.compute_energy(params)
        
        # Run optimization
        result = minimize(
            objective,
            self.initial_params,
            method=self.optimizer,
            options=optimizer_options
        )
        
        self.optimal_params = result.x
        self.optimal_energy = result.fun
        
        return self.optimal_energy, self.optimal_params
