"""
Pauli Hamiltonian representation for VQE.

A Hamiltonian is expressed as a weighted sum of Pauli strings:
    H = c0*P0 + c1*P1 + ... + cn*Pn
where each Pi is a tensor product of {I, X, Y, Z}.

Author: Bai
"""

import numpy as np


class PauliHamiltonian:
    """
    Represents a Hamiltonian as a linear combination of Pauli operators.

    Parameters
        None (use add_term to build the Hamiltonian)

    Examples:
        .. code-block:: python

            from pyqpanda_alg.VQE import PauliHamiltonian

            # H2 molecule Hamiltonian (STO-3G, bond length 0.735 Angstrom)
            H = PauliHamiltonian()
            H.add_term(-1.0523, "II")
            H.add_term(0.3979, "IZ")
            H.add_term(-0.3979, "ZI")
            H.add_term(-0.0112, "ZZ")
            H.add_term(0.1809, "XX")

            print(H)
            print("Qubits:", H.num_qubits)
    """

    VALID_PAULIS = {'I', 'X', 'Y', 'Z'}

    def __init__(self):
        self.terms = []

    def add_term(self, coeff, pauli_str):
        """
        Add a weighted Pauli term.

        Parameters
            coeff : float
                Real coefficient for this term.
            pauli_str : str
                String of Pauli operators, e.g. "IXYZ".
                Length must be consistent across all terms.
        """
        pauli_str = pauli_str.upper()
        # validate characters
        for ch in pauli_str:
            if ch not in self.VALID_PAULIS:
                raise ValueError(
                    f"Invalid Pauli character '{ch}'. Must be I, X, Y, or Z."
                )
        # validate length consistency
        if self.terms and len(pauli_str) != len(self.terms[0][1]):
            raise ValueError(
                f"Pauli string length {len(pauli_str)} does not match "
                f"existing length {len(self.terms[0][1])}."
            )
        self.terms.append((float(coeff), pauli_str))

    @property
    def num_qubits(self):
        """Number of qubits in this Hamiltonian."""
        if not self.terms:
            return 0
        return len(self.terms[0][1])

    @property
    def num_terms(self):
        """Number of Pauli terms."""
        return len(self.terms)

    def identity_offset(self):
        """Sum of coefficients for all-identity terms (constant energy offset)."""
        offset = 0.0
        for coeff, pauli_str in self.terms:
            if all(ch == 'I' for ch in pauli_str):
                offset += coeff
        return offset

    def non_identity_terms(self):
        """Return terms that contain at least one non-I Pauli operator."""
        return [
            (coeff, ps) for coeff, ps in self.terms
            if not all(ch == 'I' for ch in ps)
        ]

    def to_matrix(self):
        """
        Convert Hamiltonian to its full matrix representation (for small systems).
        Useful for verifying VQE results against exact diagonalization.

        Returns
            ndarray of shape (2^n, 2^n)
        """
        n = self.num_qubits
        dim = 2 ** n
        H_mat = np.zeros((dim, dim), dtype=complex)

        # single-qubit Pauli matrices
        pauli_mats = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }

        for coeff, pauli_str in self.terms:
            term_mat = np.array([[1.0]], dtype=complex)
            for ch in pauli_str:
                term_mat = np.kron(term_mat, pauli_mats[ch])
            H_mat += coeff * term_mat

        return H_mat

    def exact_ground_energy(self):
        """
        Compute exact ground state energy via numpy diagonalization.
        Only practical for small qubit counts (<=12).
        """
        H_mat = self.to_matrix()
        eigenvalues = np.linalg.eigvalsh(H_mat)
        return eigenvalues[0]

    def __repr__(self):
        lines = []
        for i, (coeff, ps) in enumerate(self.terms):
            sign = '+' if coeff >= 0 and i > 0 else ''
            lines.append(f"{sign}{coeff:.4f} * {ps}")
        return ' '.join(lines) if lines else "Empty Hamiltonian"

    def __len__(self):
        return len(self.terms)
