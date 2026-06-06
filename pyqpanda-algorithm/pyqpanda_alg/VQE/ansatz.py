"""
Parameterized quantum circuits (Ansatz) for VQE.
Author: Bai
"""

import numpy as np
from pyqpanda3.core import QCircuit, RY, RZ, CNOT, X


def hardware_efficient_ansatz(qubits, params, num_layers=2):
    """
    Hardware Efficient Ansatz (HEA).

    Structure per layer:
        1. RY rotation on each qubit (parameterized)
        2. RZ rotation on each qubit (parameterized)
        3. Linear CNOT entanglement (qubit[i] -> qubit[i+1])

    Total parameter count: num_qubits * 2 * num_layers

    Parameters
        qubits : list[int]
            Qubit indices, e.g. [0, 1, 2, 3].
        params : array_like
            Parameter vector. Length = len(qubits) * 2 * num_layers.
        num_layers : int, default=2
            Number of repetition layers.

    Returns
        QCircuit : the constructed parameterized circuit.

    Examples:
        .. code-block:: python

            import numpy as np
            from pyqpanda_alg.VQE import hardware_efficient_ansatz

            qubits = [0, 1, 2]
            params = np.random.uniform(0, 2*np.pi, size=3*2*2)
            circuit = hardware_efficient_ansatz(qubits, params, num_layers=2)
    """
    n = len(qubits)
    expected_params = n * 2 * num_layers
    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters, got {len(params)}."
        )

    cir = QCircuit()
    idx = 0

    for _ in range(num_layers):
        # RY rotation layer
        for q in qubits:
            cir << RY(q, params[idx])
            idx += 1
        # RZ rotation layer
        for q in qubits:
            cir << RZ(q, params[idx])
            idx += 1
        # Entanglement layer: linear CNOT chain
        for i in range(n - 1):
            cir << CNOT(qubits[i], qubits[i + 1])

    return cir


def ry_linear_ansatz(qubits, params, num_layers=2):
    """
    Simplified RY-only ansatz with linear entanglement.

    A lighter alternative to HEA — uses only RY gates (no RZ).
    Good for beginners and quick prototyping.

    Total parameter count: num_qubits * num_layers

    Parameters
        qubits : list[int]
            Qubit indices.
        params : array_like
            Parameter vector. Length = len(qubits) * num_layers.
        num_layers : int, default=2
            Number of repetition layers.

    Returns
        QCircuit : the constructed parameterized circuit.

    Examples:
        .. code-block:: python

            import numpy as np
            from pyqpanda_alg.VQE import ry_linear_ansatz

            qubits = [0, 1]
            params = np.random.uniform(0, 2*np.pi, size=2*2)
            circuit = ry_linear_ansatz(qubits, params, num_layers=2)
    """
    n = len(qubits)
    expected_params = n * num_layers
    if len(params) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters, got {len(params)}."
        )

    cir = QCircuit()
    idx = 0

    for _ in range(num_layers):
        for q in qubits:
            cir << RY(q, params[idx])
            idx += 1
        for i in range(n - 1):
            cir << CNOT(qubits[i], qubits[i + 1])

    return cir


def param_count(num_qubits, num_layers, ansatz_type='hea'):
    """
    Calculate total number of parameters for a given ansatz configuration.

    Parameters
        num_qubits : int
        num_layers : int
        ansatz_type : str, 'hea' or 'ry_linear'

    Returns
        int : total parameter count
    """
    if ansatz_type == 'hea':
        return num_qubits * 2 * num_layers
    elif ansatz_type == 'ry_linear':
        return num_qubits * num_layers
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
