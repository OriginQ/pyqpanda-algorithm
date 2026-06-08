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

import numpy as np
from scipy.linalg import expm

from pyqpanda3.core import CPUQVM, QProg, QCircuit, H, X, RY, Oracle, Encode

from .. plugin import QFT


class HHL:
    """
    This class provides a framework for the Harrow-Hassidim-Lloyd (HHL)
    algorithm for solving linear systems of equations Ax = b [1].

    For a Hermitian matrix A and a vector b it prepares a quantum state
    proportional to the solution x. Phase estimation reads the eigenvalues of A
    into a clock register and a controlled rotation inverts them on an ancilla;
    the inverse phase estimation then uncomputes the clock. Post-selecting the
    ancilla on |1> leaves the solution register proportional to A^{-1}|b>.

    Parameters
        matrix : ``numpy.ndarray``\n
            The Hermitian coefficient matrix A of shape (N, N), N = 2^n_b.
        vector : ``numpy.ndarray``, ``list``\n
            The right-hand-side vector b of length N.
        clock_qubits : ``int``\n
            The number of clock qubits used in phase estimation. Defaults to a
            value inferred from the spectrum of A.
        evolution_time : ``float``\n
            The evolution time t for U = exp(i A t). Defaults to mapping the
            smallest eigenvalue of A to clock value 1.
        constant : ``float``\n
            The rotation constant C used in the eigenvalue inversion, C <= lam_min.
            Defaults to the smallest representable eigenvalue.

    References
        [1] Harrow A W, Hassidim A, Lloyd S. Quantum algorithm for linear systems
        of equations[J]. Physical Review Letters, 2009, 103(15): 150502.

    >>> from pyqpanda_alg.HHL import HHL
    >>> import numpy as np
    >>> A = np.array([[1.0, -1.0 / 3], [-1.0 / 3, 1.0]])
    >>> b = np.array([0.0, 1.0])
    >>> x = HHL(A, b, clock_qubits=2).run()
    >>> print(np.round(x, 4))
    [0.3162 0.9487]

    """

    def __init__(self, matrix=None,
                 vector=None,
                 clock_qubits: int = None,
                 evolution_time: float = None,
                 constant: float = None,
                 machine_type: str = 'CPU'):

        A = np.array(matrix, dtype=complex)
        b = np.array(vector, dtype=complex).flatten()

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('matrix must be a square 2D array')
        if not np.allclose(A, A.conj().T):
            raise ValueError('matrix must be Hermitian')
        n = A.shape[0]
        self.n_b = int(np.log2(n))
        if 2 ** self.n_b != n:
            raise ValueError('matrix dimension must be a power of 2')
        if b.shape[0] != n:
            raise ValueError('vector length must match matrix dimension')

        self.A = A
        self.b = b
        self.machine_type = machine_type
        if machine_type == 'CPU':
            self.machine = CPUQVM()
        else:
            raise NameError("Support 'CPU' only")

        eigvals = np.linalg.eigvalsh(A)
        abs_eig = np.abs(eigvals)
        lam_min = np.min(abs_eig[abs_eig > 1e-9])
        lam_max = np.max(abs_eig)

        if clock_qubits is None:
            ratio = lam_max / lam_min
            clock_qubits = max(2, int(np.ceil(np.log2(ratio + 1))) + 1)
        self.n_c = int(clock_qubits)

        # smallest eigenvalue maps to clock value 1
        if evolution_time is None:
            evolution_time = 2 * np.pi / (2 ** self.n_c * lam_min)
        self.t = float(evolution_time)

        lam_est_min = 2 * np.pi / (self.t * 2 ** self.n_c)
        if constant is None:
            constant = lam_est_min
        self.C = float(constant)

        # registers: b = [0, n_b), clock = [n_b, n_b + n_c), ancilla = last
        self.q_b = list(range(self.n_b))
        self.q_c = list(range(self.n_b, self.n_b + self.n_c))
        self.q_anc = self.n_b + self.n_c
        self.total = self.n_b + self.n_c + 1

        self.success_prob = None
        self.state = None

    def __del__(self):
        pass

    def _phase_estimation(self, inverse=False):
        cir = QCircuit()
        if not inverse:
            for c in self.q_c:
                cir << H(c)
            for l, c in enumerate(self.q_c):
                u_pow = expm(1j * self.A * self.t * (2 ** l))
                cir << Oracle(self.q_b, u_pow).control([c])
            cir << QFT(self.q_c).dagger()
        else:
            cir << QFT(self.q_c)
            for l in reversed(range(self.n_c)):
                c = self.q_c[l]
                u_pow = expm(-1j * self.A * self.t * (2 ** l))
                cir << Oracle(self.q_b, u_pow).control([c])
            for c in self.q_c:
                cir << H(c)
        return cir

    def _eigenvalue_inversion(self):
        cir = QCircuit()
        for m in range(1, 2 ** self.n_c):
            lam_est = 2 * np.pi * m / (self.t * 2 ** self.n_c)
            arg = self.C / lam_est
            if arg > 1.0:
                continue
            theta = 2 * np.arcsin(arg)
            # flip the 0 bits of m so the controlled RY fires on |m>
            bits = [(m >> k) & 1 for k in range(self.n_c)]
            for k, bit in enumerate(bits):
                if bit == 0:
                    cir << X(self.q_c[k])
            cir << RY(self.q_anc, theta).control(self.q_c)
            for k, bit in enumerate(bits):
                if bit == 0:
                    cir << X(self.q_c[k])
        return cir

    def _build_prog(self):
        prog = QProg(self.total)

        # amplitude encoding needs a unit vector; rescaling b keeps the direction
        b_unit = self.b / np.linalg.norm(self.b)
        if np.allclose(b_unit.imag, 0.0):
            data = [float(v.real) for v in b_unit]
        else:
            data = [complex(v) for v in b_unit]
        enc = Encode()
        enc.amplitude_encode(self.q_b, data)
        prog << enc.get_circuit()

        prog << self._phase_estimation(inverse=False)
        prog << self._eigenvalue_inversion()
        prog << self._phase_estimation(inverse=True)
        return prog

    def run(self):
        """
        Run the HHL algorithm and return the normalized solution vector.

        Returns
            x : ``numpy.ndarray``\n
                The normalized solution vector x of Ax = b (up to a global phase).

        """
        prog = self._build_prog()
        self.machine.run(prog, 1)
        sv = np.array(self.machine.result().get_state_vector())

        # ancilla = 1, clock = 0 branch (little-endian)
        base = 1 << (self.n_b + self.n_c)
        amps = np.array([sv[base + xb] for xb in range(2 ** self.n_b)])

        self.success_prob = float(np.sum(np.abs(amps) ** 2))
        norm = np.linalg.norm(amps)
        if norm < 1e-12:
            raise RuntimeError('post-selection failed: try more clock qubits')
        x = amps / norm

        pivot = np.argmax(np.abs(x))
        x = x * np.exp(-1j * np.angle(x[pivot]))
        self.state = x
        return np.real_if_close(x, tol=1e6)
