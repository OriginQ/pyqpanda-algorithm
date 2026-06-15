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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, RY, RZ, CNOT, H
import numpy as np


def hardware_efficient_circuit(qubit_list, params, layers=1):
    """
    Construct a hardware-efficient ansatz circuit for VQE.
    
    This ansatz uses single-qubit rotation gates (RY, RZ) and entangling gates (CNOT)
    arranged in a pattern that is efficient for near-term quantum hardware.
    
    Parameters
        qubit_list : ``list[int]``\n
            List of qubit indices to apply the ansatz on.
        params : ``array-like``\n
            Parameters for the rotation gates. Should have length 2 * len(qubit_list) * layers.
            For each layer, we need 2 parameters per qubit (one for RY, one for RZ).
        layers : ``int``\n
            Number of layers in the ansatz circuit. Default is 1.
    
    Returns
        circuit : ``QCircuit``\n
            The parameterized quantum circuit.
    
    Examples
        >>> from pyqpanda_alg.VQE import hardware_efficient_circuit
        >>> import numpy as np
        >>> qubits = [0, 1, 2]
        >>> params = np.random.random(2 * 3 * 2)  # 2 qubits * 3 layers * 2 params per qubit
        >>> circuit = hardware_efficient_circuit(qubits, params, layers=2)
        >>> print(circuit)
    
    References
        [1] Kandala A, Mezzacapo A, Temme K, et al. Hardware-efficient variational quantum 
        eigensolver for small molecules and quantum magnets[J]. Nature, 2017, 549(7671): 242-246.
        https://doi.org/10.1038/nature23879
    """
    n_qubits = len(qubit_list)
    circuit = QCircuit()
    
    # Validate params length
    expected_length = 2 * n_qubits * layers
    if len(params) != expected_length:
        raise ValueError(f"Expected {expected_length} parameters, got {len(params)}")
    
    param_idx = 0
    
    for layer in range(layers):
        # Single-qubit rotation layer
        for i, qubit in enumerate(qubit_list):
            # RY rotation
            circuit << RY(qubit, params[param_idx])
            param_idx += 1
            # RZ rotation
            circuit << RZ(qubit, params[param_idx])
            param_idx += 1
        
        # Entangling layer (linear connectivity)
        if layer < layers - 1:  # No entangling layer after the last rotation layer
            for i in range(len(qubit_list) - 1):
                circuit << CNOT(qubit_list[i], qubit_list[i + 1])
    
    return circuit
