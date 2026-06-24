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

r"""
Variational ansatz (trial-wavefunction) library for VQE.

An *ansatz* is a parameterized quantum circuit

.. math::
    \ket{\psi(\vec\theta)} = U(\vec\theta)\ket{0}

whose parameters :math:`\vec\theta` are tuned by the classical optimizer to
minimize the energy :math:`E(\vec\theta)=\bra{\psi(\vec\theta)}H\ket{\psi(\vec\theta)}`.

Three families are provided:

- :func:`hardware_efficient_ansatz` -- a generic, hardware-friendly circuit
  with layers of single-qubit rotations interleaved with entangling gates. It
  is expressive but can suffer from barren plateaus for deep circuits.
- :func:`ucc_ansatz` -- a chemistry-inspired ansatz that approximates the
  Unitary Coupled Cluster (UCC) operator; it preserves particle number and is
  the standard choice for molecular ground states.
- :func:`symmetry_preserving_ansatz` -- a particle-number-conserving ansatz
  built from :math:`\mathrm{XX}+\mathrm{YY}` entangling blocks.

Every ansatz builder is a *function* ``(qubit_list, params) -> QCircuit`` and
returns the number of variational parameters it consumes via the companion
``*_n_params`` helpers, so that they can be plugged directly into
:class:`~pyqpanda_alg.VQE.vqe.VQE`.
"""

import math
from pyqpanda3.core import (
    QCircuit, RX, RY, RZ, CNOT, RXX, RYY, X, H,
)


def hardware_efficient_ansatz(n_qubits, params, layers=1,
                              entangler="CNOT", rotations=("RY", "RZ")):
    r"""Build a hardware-efficient variational circuit.

    Each layer applies, on every qubit, the sequence of single-qubit rotations
    listed in ``rotations`` followed by a ladder of two-qubit entangling gates.
    The parameters are flattened layer by layer.

    Parameters
        n_qubits : ``int``\n
            Number of qubits.
        params : ``array-like``\n
            Variational parameters. Its length must equal
            ``n_qubits * len(rotations) * layers`` (see
            :func:`hardware_efficient_n_params`).
        layers : ``int``, optional\n
            Number of repeated layers. Default is 1.
        entangler : ``{"CNOT", "RXX", "RYY"}``, optional\n
            Entangling gate between adjacent qubits. ``"RXX"`` and ``"RYY"``
            additionally consume one parameter per pair. Default is ``"CNOT"``.
        rotations : ``tuple`` of ``str``, optional\n
            Single-qubit rotation gates applied in order. Each entry must be one
            of ``"RX"``, ``"RY"``, ``"RZ"``. Default is ``("RY", "RZ")``.

    Return
        ``QCircuit``\n
            The parameterized ansatz circuit.

    Examples
        >>> from pyqpanda_alg.VQE.ansatz import hardware_efficient_ansatz
        >>> circ = hardware_efficient_ansatz(2, [0.1, 0.2, 0.3, 0.4], layers=1)
        >>> print(type(circ).__name__)
            QCircuit
    """
    gate_map = {"RX": RX, "RY": RY, "RZ": RZ}
    for r in rotations:
        if r not in gate_map:
            raise ValueError("rotation must be one of RX/RY/RZ, got %r" % (r,))
    if entangler not in ("CNOT", "RXX", "RYY"):
        raise ValueError("entangler must be CNOT/RXX/RYY, got %r" % (entangler,))
    n_rot = len(rotations)
    expected = n_qubits * n_rot * layers
    if len(params) < expected:
        raise ValueError(
            "hardware_efficient_ansatz needs %d params, got %d" % (expected, len(params)))

    circ = QCircuit()
    k = 0
    for _ in range(layers):
        # single-qubit rotation wall
        for q in range(n_qubits):
            for r in rotations:
                circ << gate_map[r](q, params[k])
                k += 1
        # entanglement ladder
        for q in range(n_qubits - 1):
            if entangler == "CNOT":
                circ << CNOT(q, q + 1)
            elif entangler == "RXX":
                circ << RXX(q, q + 1, params[k]); k += 1
            elif entangler == "RYY":
                circ << RYY(q, q + 1, params[k]); k += 1
            else:
                raise ValueError("entangler must be CNOT/RXX/RYY, got %r" % (entangler,))
    return circ


def hardware_efficient_n_params(n_qubits, layers=1,
                                entangler="CNOT", rotations=("RY", "RZ")):
    """Return the number of variational parameters used by
    :func:`hardware_efficient_ansatz` for the given configuration."""
    n_ent = (n_qubits - 1) * layers if n_qubits > 1 else 0
    extra = n_ent if entangler in ("RXX", "RYY") else 0
    return n_qubits * len(rotations) * layers + extra


def ucc_ansatz(n_qubits, params, excitations=None):
    r"""Build a Unitary Coupled Cluster (UCC)-style chemistry ansatz.

    Each excitation :math:`(i,j)` implements the Trotterized singles operator

    .. math::
        U_{ij}(\theta) = \exp\!\big(\theta\,(a^\dagger_i a_j - a^\dagger_j a_i)\big)

    which, on the qubit register, reduces to a Givens-style rotation on the two
    involved qubits:

    .. math::
        U_{ij}(\theta) = R_y(-\pi/2)_j\, R_x(-\pi/2)_i\,
            \mathrm{XX}(\theta)\, R_x(\pi/2)_i\, R_y(\pi/2)_j,

    i.e. a basis-rotated :math:`\mathrm{XX}` entangler (``RXX``) sandwiched by
    single-qubit rotations. This preserves the fermionic excitation structure
    and is the workhorse ansatz of molecular VQE.

    Parameters
        n_qubits : ``int``\n
            Number of qubits (spin-orbitals).
        params : ``array-like``\n
            One variational parameter per excitation.
        excitations : ``list`` of ``(int, int)``, optional\n
            Pairs ``(i, j)`` of spin-orbitals to excite from ``i`` to ``j``
            (``i < j``). If ``None``, all unique adjacent pairs
            ``[(0,1), (1,2), ...]`` are used.

    Return
        ``QCircuit``\n

    Examples
        >>> from pyqpanda_alg.VQE.ansatz import ucc_ansatz
        >>> circ = ucc_ansatz(2, [0.5], excitations=[(0, 1)])
        >>> print(type(circ).__name__)
            QCircuit
    """
    if excitations is None:
        excitations = [(i, i + 1) for i in range(n_qubits - 1)]
    if len(params) < len(excitations):
        raise ValueError("ucc_ansatz needs %d params, got %d"
                         % (len(excitations), len(params)))

    circ = QCircuit()
    for idx, (i, j) in enumerate(excitations):
        theta = params[idx]
        # map a†_i a_j - h.c. to a Givens rotation via basis change
        circ << RX(i, math.pi / 2) << RY(j, -math.pi / 2)
        circ << RXX(i, j, theta)
        circ << RX(i, -math.pi / 2) << RY(j, math.pi / 2)
    return circ


def ucc_n_params(excitations):
    """Return the number of parameters used by :func:`ucc_ansatz`
    (one per excitation)."""
    return len(excitations)


def symmetry_preserving_ansatz(n_qubits, params, layers=1):
    r"""Build a particle-number-conserving (symmetry-preserving) ansatz.

    Each layer is a sequence of entangling blocks

    .. math::
        B_{i}(\theta) = \exp\!\big(-i\theta (X_i X_{i+1} + Y_i Y_{i+1})/2\big)

    implemented as ``RXX(-θ) RYY(-θ)``, followed by single-qubit ``RZ``
    rotations. Because :math:`X_i X_{i+1}+Y_i Y_{i+1}` commutes with the total
    particle-number operator, the circuit preserves the initial population,
    which drastically shrinks the search space and improves convergence.

    Parameters
        n_qubits : ``int``\n
            Number of qubits.
        params : ``array-like``\n
            Length ``layers * (n_qubits - 1 + n_qubits)``.
        layers : ``int``, optional\n
            Number of repeated layers. Default is 1.

    Return
        ``QCircuit``\n

    Examples
        >>> from pyqpanda_alg.VQE.ansatz import symmetry_preserving_ansatz
        >>> circ = symmetry_preserving_ansatz(2, [0.1, 0.2, 0.3])
        >>> print(type(circ).__name__)
            QCircuit
    """
    if n_qubits < 2:
        raise ValueError("symmetry_preserving_ansatz needs at least 2 qubits")
    expected = layers * ((n_qubits - 1) + n_qubits)
    if len(params) < expected:
        raise ValueError("symmetry_preserving_ansatz needs %d params, got %d"
                         % (expected, len(params)))
    circ = QCircuit()
    k = 0
    for _ in range(layers):
        for q in range(n_qubits - 1):
            circ << RXX(q, q + 1, -params[k])
            circ << RYY(q, q + 1, -params[k])
            k += 1
        for q in range(n_qubits):
            circ << RZ(q, params[k]); k += 1
    return circ


def symmetry_preserving_n_params(n_qubits, layers=1):
    """Return the number of parameters used by :func:`symmetry_preserving_ansatz`."""
    return layers * ((n_qubits - 1) + n_qubits)
