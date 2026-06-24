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
Hamiltonian construction utilities for the Variational Quantum Eigensolver (VQE).

This module provides tools to build the qubit Hamiltonians that VQE optimizes. Two
broad families are supported:

1. **Quantum-chemistry (molecular) Hamiltonians.** The electronic Hamiltonian in
   second quantization,

   .. math::
       H = \sum_{p,q} h_{pq}\, a^\dagger_p a_q
         + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs}\, a^\dagger_p a^\dagger_q a_r a_s,

   is mapped to a sum of Pauli operators through the **Jordan-Wigner
   transformation** :func:`jordan_wigner`. The convenience builder
   :func:`molecular_hamiltonian` turns one- and two-electron integrals directly
   into a :class:`~pyqpanda3.hamiltonian.PauliOperator`.

2. **Lattice / physics Hamiltonians.** :func:`transverse_field_ising` and
   :func:`heisenberg_model` build the textbook spin Hamiltonians that are the
   workhorses of condensed-matter VQE studies.

The well-known ``H2`` molecule is shipped as a pre-built Hamiltonian so that
users can run an example without a quantum-chemistry integral driver; larger
molecules can be obtained from :func:`molecular_hamiltonian` once the one-/two-
electron integrals are provided.
"""

import numpy as np
from pyqpanda3.hamiltonian import PauliOperator


def _identity():
    """Return the identity PauliOperator ``I`` with coefficient 1."""
    return PauliOperator({"": 1.0})


def _z_string(n, p):
    r"""Return the Jordan-Wigner Z-string :math:`\prod_{k=0}^{p-1} Z_k`."""
    op = _identity()
    for k in range(p):
        op = op * PauliOperator({"Z" + str(k): 1.0})
    return op


def jw_create(n, p):
    r"""Jordan-Wigner image of a fermionic creation operator :math:`a^\dagger_p`.

    Under the Jordan-Wigner transformation a creation operator reads

    .. math::
        a^\dagger_p = \frac{1}{2}\Big(\prod_{k=0}^{p-1} Z_k\Big)(X_p - i Y_p).

    Parameters
        n : ``int``\n
            Total number of spin-orbitals (qubits).
        p : ``int``\n
            Index of the spin-orbital on which the operator acts.

    Return
        ``PauliOperator``\n
            The qubit (Pauli) representation of :math:`a^\dagger_p`.

    Examples
        >>> from pyqpanda_alg.VQE.hamiltonian import jw_create, jw_annihilate
        >>> # number operator n_0 = a^\dagger_0 a_0 = (I - Z_0) / 2
        >>> n0 = jw_create(2, 0) * jw_annihilate(2, 0)
        >>> print(n0)
            { qbit_total = 1, pauli_with_coef_s = { '':0.5 + 0j, 'Z0 ':-0.5 + 0j, } }
    """
    z = _z_string(n, p)
    return 0.5 * (z * PauliOperator({"X" + str(p): 1.0})) \
        + (-0.5j) * (z * PauliOperator({"Y" + str(p): 1.0}))


def jw_annihilate(n, p):
    r"""Jordan-Wigner image of a fermionic annihilation operator :math:`a_p`.

    Under the Jordan-Wigner transformation an annihilation operator reads

    .. math::
        a_p = \frac{1}{2}\Big(\prod_{k=0}^{p-1} Z_k\Big)(X_p + i Y_p).

    Parameters
        n : ``int``\n
            Total number of spin-orbitals (qubits).
        p : ``int``\n
            Index of the spin-orbital on which the operator acts.

    Return
        ``PauliOperator``\n
            The qubit (Pauli) representation of :math:`a_p`.
    """
    z = _z_string(n, p)
    return 0.5 * (z * PauliOperator({"X" + str(p): 1.0})) \
        + (0.5j) * (z * PauliOperator({"Y" + str(p): 1.0}))


def jordan_wigner(one_body, two_body):
    r"""Map a fermionic Hamiltonian to a qubit Hamiltonian via Jordan-Wigner.

    The electronic Hamiltonian

    .. math::
        H = \sum_{p,q} h_{pq}\, a^\dagger_p a_q
          + \frac{1}{2}\sum_{p,q,r,s} h_{pqrs}\, a^\dagger_p a^\dagger_q a_r a_s

    is transformed by substituting every :math:`a^\dagger_p`, :math:`a_p` with
    its Jordan-Wigner Pauli representation (:func:`jw_create`,
    :func:`jw_annihilate`) and collecting the resulting Pauli terms.

    Parameters
        one_body : ``numpy.ndarray`` of shape ``(n, n)``\n
            One-electron integral matrix :math:`h_{pq}`.
        two_body : ``numpy.ndarray`` of shape ``(n, n, n, n)``\n
            Two-electron integral tensor :math:`h_{pqrs}` (chemists' notation,
            i.e. already including the :math:`1/2` spin-summed factor).

    Return
        ``PauliOperator``\n
            The qubit Hamiltonian :math:`H` as a sum of Pauli operators.

    Examples
        >>> import numpy as np
        >>> from pyqpanda_alg.VQE.hamiltonian import jordan_wigner
        >>> # trivial 2-orbital case with only h_00 = 1
        >>> h1 = np.zeros((2, 2)); h1[0, 0] = 1.0
        >>> h2 = np.zeros((2, 2, 2, 2))
        >>> H = jordan_wigner(h1, h2)
        >>> print(H)
            { qbit_total = 1, pauli_with_coef_s = { '':0.5 + 0j, 'Z0 ':-0.5 + 0j, } }
    """
    n = one_body.shape[0]
    h = PauliOperator({"": 0.0})

    # one-body part: sum_{pq} h_{pq} a^\dagger_p a_q
    for p in range(n):
        for q in range(n):
            coef = one_body[p, q]
            if abs(coef) < 1e-12:
                continue
            h = h + coef * (jw_create(n, p) * jw_annihilate(n, q))

    # two-body part: 1/2 sum_{pqrs} h_{pqrs} a^\dagger_p a^\dagger_q a_r a_s
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    coef = two_body[p, q, r, s]
                    if abs(coef) < 1e-12:
                        continue
                    term = jw_create(n, p) * jw_create(n, q) \
                        * jw_annihilate(n, r) * jw_annihilate(n, s)
                    h = h + 0.5 * coef * term
    return h


def molecular_hamiltonian(nuclear_repulsion, one_body, two_body):
    r"""Build the full molecular Hamiltonian from electronic integrals.

    Combines the nuclear-nuclear repulsion energy with the Jordan-Wigner image
    of the electronic Hamiltonian:

    .. math::
        H_{\text{mol}} = E_{\text{nuc}}
            + \sum_{pq} h_{pq} a^\dagger_p a_q
            + \frac{1}{2}\sum_{pqrs} h_{pqrs}
              a^\dagger_p a^\dagger_q a_r a_s.

    Parameters
        nuclear_repulsion : ``float``\n
            Classical nuclear-nuclear repulsion energy
            :math:`E_{\text{nuc}}`.
        one_body : ``numpy.ndarray`` of shape ``(n, n)``\n
            One-electron integral matrix in the spin-orbital basis.
        two_body : ``numpy.ndarray`` of shape ``(n, n, n, n)``\n
            Two-electron integral tensor (chemists' notation).

    Return
        ``PauliOperator``\n
            Molecular Hamiltonian ready to be passed to :class:`~pyqpanda_alg.VQE.vqe.VQE`.
    """
    h = jordan_wigner(one_body, two_body)
    h = h + nuclear_repulsion * _identity()
    return h


def transverse_field_ising(n, jz, hx, periodic=False):
    r"""Build a transverse-field Ising (TFI) Hamiltonian.

    .. math::
        H = \sum_{\langle i,j\rangle} J_z\, Z_i Z_j
            + \sum_{i} h_x\, X_i.

    Parameters
        n : ``int``\n
            Number of spins (qubits).
        jz : ``float``\n
            Nearest-neighbour :math:`Z_i Z_j` coupling strength.
        hx : ``float``\n
            Transverse-field strength along :math:`X`.
        periodic : ``bool``, optional\n
            Whether to close the chain into a ring (periodic boundary
            conditions). Default is ``False`` (open chain).

    Return
        ``PauliOperator``\n

    Examples
        >>> from pyqpanda_alg.VQE.hamiltonian import transverse_field_ising
        >>> H = transverse_field_ising(3, 1.0, 0.5)
        >>> print(H.max_qbit_idx() + 1, 'qubits')
            3 qubits
    """
    h = PauliOperator({"": 0.0})
    for i in range(n - 1):
        h = h + jz * PauliOperator({"Z" + str(i) + " Z" + str(i + 1): 1.0})
    if periodic and n > 2:
        h = h + jz * PauliOperator({"Z0 Z" + str(n - 1): 1.0})
    for i in range(n):
        h = h + hx * PauliOperator({"X" + str(i): 1.0})
    return h


def heisenberg_model(n, jx=1.0, jy=1.0, jz=1.0, periodic=False):
    r"""Build an (XXX/Heisenberg) spin-1/2 Hamiltonian.

    .. math::
        H = \sum_{\langle i,j\rangle}
            \big( J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j \big).

    Parameters
        n : ``int``\n
            Number of spins (qubits).
        jx, jy, jz : ``float``, optional\n
            Coupling strengths along each axis. With the defaults
            ``J_x=J_y=J_z=1`` this is the isotropic XXX model.
        periodic : ``bool``, optional\n
            Use periodic boundary conditions. Default is ``False``.

    Return
        ``PauliOperator``\n
    """
    h = PauliOperator({"": 0.0})
    pairs = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        pairs.append((0, n - 1))
    for i, j in pairs:
        if abs(jx) > 1e-15:
            h = h + jx * PauliOperator({"X" + str(i) + " X" + str(j): 1.0})
        if abs(jy) > 1e-15:
            h = h + jy * PauliOperator({"Y" + str(i) + " Y" + str(j): 1.0})
        if abs(jz) > 1e-15:
            h = h + jz * PauliOperator({"Z" + str(i) + " Z" + str(j): 1.0})
    return h


# ---------------------------------------------------------------------------
# Pre-built benchmark Hamiltonians
# ---------------------------------------------------------------------------
def h2_hamiltonian():
    r"""Return the 2-qubit reduced Hamiltonian of the H\ :sub:`2` molecule.

    The data corresponds to ``H2`` in a STO-3G basis at equilibrium bond length
    (``R ≈ 0.74 Å``) after freezing the chemically-inactive orbitals and
    tapering two qubits with the :math:`\mathbb{Z}_2` parity symmetry. This is
    the canonical 2-qubit benchmark used in the early VQE experiments
    (O'Malley 2016, Kandala 2017).

    .. math::
        H = c_0\,I + c_1 Z_0 + c_2 Z_1 + c_3 Z_0 Z_1 + c_4 X_0 X_1.

    Return
        ``PauliOperator``\n
            The 2-qubit H\ :sub:`2` Hamiltonian. Its ground-state energy is
            ``-1.857275`` Hartree.

    Examples
        >>> import numpy as np
        >>> from pyqpanda_alg.VQE.hamiltonian import h2_hamiltonian
        >>> H = h2_hamiltonian()
        >>> w = np.linalg.eigvalsh(np.array(H.matrix()))
        >>> print('ground state energy:', round(float(w[0].real), 6))
            ground state energy: -1.857275
    """
    c0 = -1.0523732457748599
    c1 = 0.39793742484318045
    c2 = -0.39793742484318045
    c3 = -0.01128010425623538
    c4 = 0.18093119978423156
    h = (c0 * PauliOperator({"": 1.0})
         + c1 * PauliOperator({"Z0": 1.0})
         + c2 * PauliOperator({"Z1": 1.0})
         + c3 * PauliOperator({"Z0 Z1": 1.0})
         + c4 * PauliOperator({"X0 X1": 1.0}))
    return h


def exact_ground_energy(hamiltonian):
    r"""Compute the exact ground-state energy by dense diagonalization.

    Useful as a reference (benchmark) against which the VQE result is compared.

    Parameters
        hamiltonian : ``PauliOperator``\n

    Return
        ``float``\n
            Smallest eigenvalue :math:`\lambda_0` of the Hamiltonian matrix.
    """
    mat = np.array(hamiltonian.matrix(), dtype=complex)
    return float(np.linalg.eigvalsh(mat)[0].real)
