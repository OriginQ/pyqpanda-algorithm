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

import warnings
import itertools

import numpy as np
from scipy.optimize import minimize

from pyqpanda3.core import CPUQVM, QProg, QCircuit, RY, CZ, X, Y, Z, Encode, expval_pauli_operator
from pyqpanda3.hamiltonian import PauliOperator


class VQLS:
    """
    This class provides a framework for the Variational Quantum Linear Solver
    (VQLS) algorithm for solving linear systems of equations Ax = b [1].

    For a Hermitian matrix A it decomposes A into a weighted sum of Pauli
    strings and trains a hardware-efficient RY ansatz to minimize a global cost
    built from the Pauli expectation <x|A^2|x> and the overlap amplitudes
    <b|P_l|x(theta)>. The optimized ansatz prepares a state proportional to the
    solution x. Unlike phase-estimation solvers it is clock-register-free and
    ancilla-free, making it suited to NISQ hardware; the real RY ansatz limits
    it to solutions expressible with real amplitudes.

    Parameters
        matrix : ``numpy.ndarray``\n
            The Hermitian coefficient matrix A of shape (N, N), N = 2^n.
        vector : ``numpy.ndarray``, ``list``\n
            The right-hand-side vector b of length N.
        ansatz_layers : ``int``\n
            The number of entangling layers in the hardware-efficient ansatz.
        optimizer : ``str``\n
            The classical optimizer, any scipy.optimize.minimize method name.
        maxiter : ``int``\n
            The maximum number of classical optimizer iterations.
        seed : ``int``\n
            The random seed for the initial ansatz parameters.

    References
        [1] Bravo-Prieto C, LaRose R, Cerezo M, et al. Variational quantum linear
        solver[J]. Quantum, 2023, 7: 1188.

    >>> from pyqpanda_alg.VQLS import VQLS
    >>> import numpy as np
    >>> A = np.array([[1.0, -1.0 / 3], [-1.0 / 3, 1.0]])
    >>> b = np.array([0.0, 1.0])
    >>> x = VQLS(A, b, seed=0).run()
    >>> print(np.round(x, 4))
    [0.3162 0.9487]

    """

    def __init__(self, matrix=None,
                 vector=None,
                 ansatz_layers: int = 3,
                 optimizer: str = 'SLSQP',
                 maxiter: int = 200,
                 seed: int = 0,
                 machine_type: str = 'CPU'):

        A = np.array(matrix, dtype=complex)
        b = np.array(vector, dtype=complex).flatten()

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('matrix must be a square 2D array')
        if not np.allclose(A, A.conj().T):
            raise ValueError('matrix must be Hermitian')
        N = A.shape[0]
        self.n_qubits = int(np.log2(N))
        if 2 ** self.n_qubits != N:
            raise ValueError('matrix dimension must be a power of 2')
        if b.shape[0] != N:
            raise ValueError('vector length must match matrix dimension')
        if np.allclose(A, 0):
            raise ValueError('matrix must be nonzero')
        if np.linalg.norm(b) < 1e-12:
            raise ValueError('vector must be nonzero')

        self.A = A
        self.b = b
        self.N = N
        self.ansatz_layers = ansatz_layers
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.seed = seed
        self.machine_type = machine_type
        if machine_type == 'CPU':
            self.machine = CPUQVM()
        else:
            raise NameError("Support 'CPU' only")

        n = self.n_qubits
        paulis = {'I': np.eye(2, dtype=complex),
                  'X': np.array([[0, 1], [1, 0]], dtype=complex),
                  'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                  'Z': np.array([[1, 0], [0, -1]], dtype=complex)}

        pauli_dict = {}
        for combo in itertools.product('IXYZ', repeat=n):
            # little-endian kron: qubit n-1 is outermost factor, qubit 0 innermost
            mat = np.array([[1.0 + 0j]])
            for q in range(n - 1, -1, -1):
                mat = np.kron(mat, paulis[combo[q]])
            c_l = np.trace(mat.conj().T @ A) / N
            if abs(c_l) > 1e-9:
                tokens = [combo[q] + str(q) for q in range(n) if combo[q] != 'I']
                label = ' '.join(tokens) if tokens else 'I'
                pauli_dict[label] = float(c_l.real)

        self.pauli_dict = pauli_dict
        self.A_op = PauliOperator(pauli_dict)
        # pyqpanda3 pauli Y is conjugated; harmless for the real ansatz,
        # <x|conj(A)^2|x> = <x|A^2|x> when x is real
        self.A2_op = self.A_op * self.A_op

        b_unit = b / np.linalg.norm(b)
        if np.allclose(b_unit.imag, 0.0):
            data = [float(v.real) for v in b_unit]
        else:
            data = [complex(v) for v in b_unit]
        enc = Encode()
        enc.amplitude_encode(list(range(n)), data)
        self.u_b = enc.get_circuit()

        self.n_params = n * (ansatz_layers + 1)

        self.final_cost = None
        self.opt_params = None
        self.n_iters = None
        self.cost_history = []
        self.converged = None
        self.state = None

    def __del__(self):
        pass

    def _ansatz(self, params):
        cir = QCircuit()
        n = self.n_qubits
        k = 0
        for q in range(n):
            cir << RY(q, params[k])
            k += 1
        for _ in range(self.ansatz_layers):
            for q in range(n - 1):
                cir << CZ(q, q + 1)
            # ring entangler only for n > 2, CZ(0,1)+CZ(1,0) cancels at n=2
            if n > 2:
                cir << CZ(n - 1, 0)
            for q in range(n):
                cir << RY(q, params[k])
                k += 1
        return cir

    def _pauli_gates(self, label):
        cir = QCircuit()
        if label == 'I':
            return cir
        for tok in label.split():
            p = tok[0]
            q = int(tok[1:])
            if p == 'X':
                cir << X(q)
            elif p == 'Y':
                cir << Y(q)
            elif p == 'Z':
                cir << Z(q)
        return cir

    def _overlap(self, params, label):
        n = self.n_qubits
        prog = QProg(n) << self._ansatz(params) << self._pauli_gates(label) << self.u_b.dagger()
        self.machine.run(prog, 1)
        # <b|P_l|x(theta)>
        return complex(np.array(self.machine.result().get_state_vector())[0])

    def _cost(self, params):
        n = self.n_qubits
        D = expval_pauli_operator(QProg(n) << self._ansatz(params), self.A2_op, 1)
        if D < 1e-12:
            cost = 1.0
        else:
            amp = 0.0 + 0j
            for label, c_l in self.pauli_dict.items():
                amp += c_l * self._overlap(params, label)
            cost = 1.0 - abs(amp) ** 2 / D
        self.cost_history.append(cost)
        return cost

    def run(self):
        """
        Run the variational quantum linear solver.

        Returns
            x : ``numpy.ndarray``\n
                The normalized solution vector x of Ax = b (up to a global phase).

        """
        n = self.n_qubits
        x0 = np.random.default_rng(self.seed).uniform(-np.pi, np.pi, self.n_params)
        options = {'maxiter': self.maxiter}
        if self.optimizer == 'SLSQP':
            options['ftol'] = 1e-10
        res = minimize(self._cost, x0, method=self.optimizer, options=options)

        self.final_cost = float(res.fun)
        self.opt_params = res.x
        self.n_iters = int(res.get('nit', res.nfev))
        self.converged = self.final_cost < 1e-2
        if not self.converged:
            warnings.warn('VQLS did not converge: final cost %.3e' % self.final_cost)

        self.machine.run(QProg(n) << self._ansatz(res.x), 1)
        x = np.array(self.machine.result().get_state_vector())

        pivot = np.argmax(np.abs(x))
        x = x * np.exp(-1j * np.angle(x[pivot]))
        x = np.real_if_close(x, tol=1e6)
        self.state = x
        return x
