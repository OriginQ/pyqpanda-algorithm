"""Hardware-efficient ansatz builder for the VQE package.

:func:`hardware_efficient_ansatz` assembles the built-in variational
circuit: alternating single-qubit rotation layers and linear CNOT
entanglement.  The returned :class:`~pyqpanda3.vqcircuit.VQCircuit`
declares its parameter group up front (``set_Param``), so the ansatz
is callable with a concrete parameter vector; the bound result exposes
the materialized circuit through ``circuits()``, which the execution
layer's variational sessions consume.

Parameter ordering is a stable API contract: layer-major, qubit-major,
and RY before RZ per qubit.  For parameter ``i`` the gate it feeds is

``index = layer * (2 * num_qubits) + qubit * 2 + gate``

with ``gate`` 0 for RY and 1 for RZ: all parameters of layer 0 come
first, then layer 1, and so on; within a layer every qubit contributes
RY then RZ, and each layer ends with the linear CNOT chain (0,1),
(1,2), ..., (n-2, n-1).
"""

from pyqpanda3.core import CNOT, RY, RZ
from pyqpanda3.vqcircuit import VQCircuit


def hardware_efficient_ansatz(num_qubits: int, layers: int) -> VQCircuit:
    """Build the hardware-efficient ansatz on ``num_qubits`` qubits.

    Each of the ``layers`` repetitions applies RY and RZ to every
    qubit, followed by CNOTs ``(0,1)``, ``(1,2)``, through
    ``(n-2,n-1)``.  The parameter count is ``2 * num_qubits * layers``
    and the ordering is layer-major, qubit-major, RY before RZ per
    qubit (see the module docstring).

    Args:
        num_qubits: Number of qubits, at least 2 (the entanglement
            chain needs a qubit pair).
        layers: Number of rotation-and-entanglement repetitions, at
            least 1.

    Returns:
        The parameterized ansatz circuit.

    Raises:
        ValueError: If ``num_qubits`` is smaller than 2 or ``layers``
            is smaller than 1.
    """
    if num_qubits < 2:
        raise ValueError(
            "num_qubits must be at least 2 for the CNOT entanglement, "
            f"got {num_qubits}"
        )
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")

    ansatz = VQCircuit(num_qubits)
    ansatz.set_Param([2 * num_qubits * layers])

    index = 0
    for _ in range(layers):
        for qubit in range(num_qubits):
            ansatz << RY(qubit, ansatz.Param([index]))
            index += 1
            ansatz << RZ(qubit, ansatz.Param([index]))
            index += 1
        for qubit in range(num_qubits - 1):
            ansatz << CNOT(qubit, qubit + 1)
    return ansatz
