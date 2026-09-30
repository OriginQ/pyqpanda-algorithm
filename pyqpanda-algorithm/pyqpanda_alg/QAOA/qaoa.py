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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, I, H, RZ, RX, CNOT, measure
from pyqpanda3.hamiltonian import PauliOperator, Hamiltonian
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import barycentric_interpolate as b_interp
import sympy as sp
from . import spsa
from .default_circuits import *

from .. plugin import *



def p_1(n):
    """
    Transfer binary variable :math:`x_n` to pauli operator :math:`\\frac{I-Z_n}{2}`

    Parameters
        n : ``int``\n
            index of the variable, start with 0.

    Return
        operator : ``PauliOperator``\n
            Pauli operator :math:`\\frac{I-Z_n}{2}`

    Examples
        Transfer :math:`x_0` into pauli operator :math:`\\frac{I-Z_0}{2}`
    
    >>> from pyqpanda_alg.QAOA import qaoa
    >>> operator_0 = qaoa.p_1(0)
    >>> print(operator_0)
        { qbit_total = 1, pauli_with_coef_s = { '':0.5 + 0j, 'Z0 ':-0.5 + 0j, } }
    """
    operator = PauliOperator({"I" : 0.5}) - PauliOperator({"Z" + str(n) : 0.5})
    return operator


def p_0(n):
    """
    Transfer binary variable :math:`x_n` to pauli operator :math:`\\frac{I+Z_n}{2}`

    Parameters
        n : ``integer``\n
            index of the variable, start with 0.

    Return
        operator : ``PauliOperator``\n
            Pauli operator :math:`\\frac{I+Z_n}{2}`

    Examples
        Transfer :math:`x_0` into pauli operator :math:`\\frac{I+Z_0}{2}`
    
    >>> from pyqpanda_alg.QAOA import qaoa
    >>> operator_0 = qaoa.p_0(0)
    >>> print(operator_0)
        { qbit_total = 1, pauli_with_coef_s = { '':0.5 + 0j, 'Z0 ':0.5 + 0j, } }
    """
    operator = PauliOperator({"I" : 0.5}) + PauliOperator({"Z" + str(n) : 0.5})
    return operator


def problem_to_z_operator(problem, norm=False):
    """
    Transfer polynomial function with binary variables :math:`f(x_0, \cdots, x_n)` to
    pauli operator :math:`f(\\frac{I-Z_0}{2}, \cdots, \\frac{I-Z_n}{2})`

    Parameters
        problem : ``expression`` in sympy\n

    Return
        hamiltonian : ``PauliOperator``\n
            Pauli operators :math:`f(\\frac{I-Z_n}{2})` in list form.

    Examples
        Transfer :math:`2 x_0 x_1 + 3 x_2 - 1` into pauli operators

    >>> import sympy as sp
    >>> from pyqpanda_alg.QAOA import qaoa
    >>> vars = sp.symbols('x0:3')
    >>> f = 2*vars[0]*vars[1] + 3*vars[2] - 1
    >>> print(f)
        2*x0*x1 + 3*x2 - 1
    >>> hamiltonian = qaoa.problem_to_z_operator(f)
    >>> print(hamiltonian)
        { qbit_total = 3, pauli_with_coef_s = { '':1 + 0j, 'Z2 ':-1.5 + 0j, 'Z1 ':-0.5 + 0j, 'Z0 ':-0.5 + 0j, 'Z0 Z1 ':0.5 + 0j, } }

    """
    problem_symbols = list(sorted(problem.free_symbols, key=lambda symbol: symbol.name))
    operator_symbols = np.array([sp.Symbol('z%d' % i) for i in range(len(problem_symbols))])

    hamiltonian = PauliOperator({"I": 0.0})
    operator_problem = problem.xreplace(dict(zip(problem_symbols, (1 - operator_symbols) / 2)))
    problem = sp.Poly(operator_problem).as_dict()

    for monomial, coefficient in problem.items():
        hami_one_term = PauliOperator({"I" : 1.0})
        for index, one_term in enumerate(monomial):
            if one_term % 2 == 0:
                hami_one_term *= 1
            else:
                hami_one_term *= PauliOperator({"Z%d" % index : 1.0})
        hamiltonian += coefficient * hami_one_term
    # hamiltonian = hamiltonian.toHamiltonian(True)

    if norm:
        # coef_list = [np.fabs(x) for term, x in hamiltonian.toHamiltonian(True) if term]
        coef_list = [np.fabs(term.coef().real) for term in hamiltonian.terms()]
        norm_factor = 1 / np.mean(coef_list)
        # hamiltonian = norm_hami(hamiltonian, norm_factor)
        hamiltonian = norm_factor * hamiltonian

    return hamiltonian


def parameter_interpolate(pm):
    """
    Use INTERP heuristic strategy to guess the initial parameter of :math:`p+1` layer QAOA
    from the optimal parameter found from :math:`p` layer QAOA circuit.

    Parameters
        pm : ``array-like``\n
            Optimal parameters of :math:`p` layer QAOA circuit, with length :math:`2\times p`

    Return
        operator : ``array-like``\n
            A guess for the initial parameter of :math:`p+1` layer QAOA, with length :math:`2\times (p+1)`

    References
        [1] ZHOU L, WANG S T, CHOI S, et. Quantum Approximate Optimization Algorithm: Performance, Mechanism, and
        Implementation on Near-Term Devices[J/OL]. Physical Review X, 2020, 10(2): 021067.
        :DOI: `10.1103/PhysRevX.10.021067.`


    Examples
        Transfer :math:`x_0` into pauli operator :math:`\\frac{I-Z_0}{2}`

        >>> import numpy as np
        >>> from pyqpanda_alg.QAOA import qaoa
        >>> initial_parameter = np.array([0.1, 0.2, 0.2, 0.1])
        >>> new_parameter = qaoa.parameter_interpolate(initial_parameter)
        >>> print(new_parameter)
            [0.1  0.15 0.25 0.2  0.15 0.05]

    """

    def interp_x(n):
        # Chebyshev zero modes
        return np.cos(np.pi * (np.arange(n)) / n)

    level = len(pm) // 2
    gamma = pm[:level]
    beta = pm[level:]
    if level == 1:
        print('warning of one-element interpolation')
        new_gamma = [gamma[0], gamma[0] / 2]
        new_beta = [beta[0] / 2, beta[0]]
        return np.append(new_gamma, new_beta)
    old_level = interp_x(level)
    new_level = interp_x(level + 1)
    new_gamma = b_interp(old_level, gamma, new_level)
    new_beta = b_interp(old_level, beta, new_level)
    return np.append(new_gamma, new_beta)


def pauli_z_operator_to_circuit(operator, qlist, gamma=np.pi):
    """
    Circuit of simulation diagonal Hamiltonian :math:`e^{-iH\theta}`.

    Parameters
        operator : ``list``\n
            Pauli Operator in list form. (By method `operator.toHamiltonian(1)`)
        qlist : ``qubit list``\n
        gamma : ``float``\n
            Value of theta in :math:`e^{-iH\theta}`.

    Return
        circuit : ``pq.QCircuit``\n
            Circuit of simulation diagonal Hamiltonian :math:`e^{-iH\theta}`.

        constant : ``float``\n
            Constant number in the hamiltonian.

    Example
        Print a circuit of hamiltonian for problem :math:`f(\\vec{x})=2x_0x_1 + 3x_2 - 1` with :math:`\\theta=0`

    .. code-block:: python
    
        import sympy as sp
        from pyqpanda_alg.QAOA import qaoa
        vars = sp.symbols('x0:3')
        f = 2*vars[0]*vars[1] + 3*vars[2] - 1

        hamiltonian = qaoa.problem_to_z_operator(f)
        circuit, constant = qaoa.pauli_z_operator_to_circuit(hamiltonian, list(range(3)), 0)
        print(circuit)

    .. parsed-literal::

                  ┌────────────┐                              ┌─┐
        q_0:  |0>─┤RZ(0.000000)├ ───■── ────────────── ───■── ┤I├
                  ├────────────┤ ┌──┴─┐ ┌────────────┐ ┌──┴─┐ ├─┤
        q_1:  |0>─┤RZ(0.000000)├ ┤CNOT├ ┤RZ(0.000000)├ ┤CNOT├ ┤I├
                  ├────────────┤ ├─┬──┘ └────────────┘ └────┘ └─┘
        q_2:  |0>─┤RZ(0.000000)├ ┤I├─── ────────────── ────── ───
                  └────────────┘ └─┘


    """
    circuit = QCircuit()
    constant = 0
    for term in operator.terms():
        coef = term.coef().real
        paulis = term.paulis()
        index_list = [q.qbit() for q in paulis if q.is_Z()]
        ang = coef * gamma
        n = len(index_list)
        if n == 0:
            constant += coef
        elif n == 1:
            circuit << RZ(qlist[index_list[0]], 2 * ang)
        else:
            q = [qlist[j] for j in index_list]
            for j in range(n - 1):
                circuit << CNOT(q[j], q[j + 1])
            circuit << RZ(q[-1], 2 * ang)

            for j in range(n - 1):
                circuit << CNOT(q[-j - 2], q[-j - 1])
    for i in range(len(qlist)):
        circuit << I(qlist[i])
    return circuit, constant


class QAOA:
    """
    This class provides the quantum alternative operator ansatz algorithm optimizer. It assumes a polynomial problem
    consisting only of binary variables. The problem is then translated into an Ising Hamiltonian whose minimal eigen
    vector and corresponding eigenstate correspond to the optimal solution of the original optimization problem.
    The provided solver is then used to approximate the ground state of the Hamiltonian to find a good solution for the
    optimization problem.

    Parameters
        problem : ``expression`` in sympy or ``pq.PauliOperator``\n
            A polynomial function with binary variables to be optimized. Support an expression in sympy. Next version will
            support an object from pypanda PauliOperator.

        init_circuit : ``function``,  ``optional``\n
            The quantum circuit to create the initial state of QAOA algorithm. Default is Hadamard circuit to create an
            equal superposition state :math:`\ket{\psi} = 2^{-n/2}\sum_{i=0}^{2^n-1}\ket{i}`.

        mixer_circuit : ``function``, ``optional``\n
            The function which returns a mixer quantum circuit :math:`U_M(\\beta)=\exp(-i\\beta H_M)`.
            The function should accept two parameters (qubit list, array-like angles) as input, and return a quantum
            circuit as output. Default is X mixer circuit :math:`\exp(-i\\beta \sum_i X_i)=RX(2\\beta)^{\otimes n}`

        norm : ``bool``\n


    Attributes
        energy_dict : ``dict``\n
            The dict which stores the function value for solutions being sampled during the optimization.
        problem_dimension : ``integer``\n
            The problem dimension, and also the qubit number.
        circuit iter : ``integer``\n
            The number of times the quantum circuit being called during optimization.

    Methods
        calculate_energy : Calculate the function value for one solution.

        run_qaoa_circuit : Given parameters, run the qaoa circuit and get the theoretical probability distribution.

        run : run the optimization process



    Reference
        [1] FARHI E, GOLDSTONE J, GUTMANN S. A Quantum Approximate Optimization Algorithm[J/OL]. 2014[2022-03-09].
        https://arxiv.org/abs/1411.4028v1. DOI:10.48550/arXiv.1411.4028.\n
        [2] ZHOU L, WANG S T, CHOI S, et. Quantum Approximate Optimization Algorithm: Performance, Mechanism,
        and Implementation on Near-Term Devices[J/OL]. Physical Review X, 2020, 10(2): 021067.
        DOI:10.1103/PhysRevX.10.021067.

    """

    def __init__(self, problem, init_circuit=None,
                 mixer_circuit=None, norm=False):

        self.measure_type = None
        self.optimizer = None
        self.optimize_option = None
        self.shots = None
        self.optimize_type = None
        self.temperature = None
        self.alpha = None
        self.loss_type = None

        problem_dimension = 0
        self.problem = problem
        self.operator = None
        if isinstance(problem, sp.Basic):
            self.problem = sp.simplify(problem)
            problem_dimension = len(problem.atoms(sp.Symbol))
            self.operator = problem_to_z_operator(self.problem, norm)

        elif isinstance(problem, Hamiltonian):
            self.problem = problem.pauli_operator()
            self.operator = problem.pauli_operator()
            qubit = set()
            for term in problem.terms():
                qubit = qubit |set([qubit.qbit() for qubit in term.paulis()])
            problem_dimension = len(qubit)

        elif isinstance(problem, PauliOperator):
            self.problem = problem
            self.operator = problem
            qubit = set()
            for term in problem.terms():
                qubit = qubit |set([qubit.qbit() for qubit in term.paulis()])
            problem_dimension = len(qubit)
        
        else:
            raise TypeError("problem must be a sympy expression or a PauliOperator")
        
        self.problem_dimension = problem_dimension
        self.init_circuit = init_circuit

        self.mixer_circuit = mixer_circuit
        self.layer = None

        self.iter = 0
        self.circuit_iter = 0

        self.energy_dict = {}

    def calculate_energy(self, x):
        """
        Calculate the function value for one solution.
        TODO: using new method to acccelrate the calculation.

        Parameter
            x : ``array-like``\n
                one binary variables solution in vector form.

        Return 
            ``float``\n
            function value of the solution :math:`f(x)`.

        Example
            Let :math:`f(\\vec{x})=2x_0x_1 + 3x_2 - 1`, calculate :math:`f(1,0,0)`

        >>> import sympy as sp
        >>> from pyqpanda_alg.QAOA.qaoa import *
        >>> vars = sp.symbols('x0:3')
        >>> f = 2*vars[0]*vars[1] + 3*vars[2] - 1
        >>> print(f)
            2*x0*x1 + 3*x2 - 1
        >>> qaoa_f = QAOA(f)
        >>> solution_1 = [1, 0, 0]
        >>> print(qaoa_f.calculate_energy(solution_1))
            -1
        >>> solution_2 = [0, 1, 1]
        >>> print(qaoa_f.calculate_energy(solution_2))
            2
        >>> ham_f = 2 * p_1(0) * p_1(1) + 3 * p_1(2)- PauliOperator('I')*1
        >>> qaoa_ham = QAOA(ham_f)
        >>> print(qaoa_ham.calculate_energy(solution_1))
            -1.0

        """
        bit_form = x
        result = 0
        if isinstance(self.problem, sp.Basic):
            symbols = sorted(self.problem.free_symbols, key=lambda symbol: symbol.name)
            # problem = sp.Poly(self.problem)
            value_dict = {}
            for i in range(len(x)):
                value_dict[symbols[i]] = bit_form[i]
            f = sp.lambdify(symbols, self.problem, 'numpy')
            raw_result = f(*bit_form)
            result = raw_result.real if isinstance(raw_result, complex) else raw_result

        if isinstance(self.problem, PauliOperator):
            result = 0
            for term in self.problem.terms():
                coef = term.coef()
                term_dic = {pauli.qbit(): pauli.pauli_char() for pauli in term.paulis() if pauli.is_Z()}
                real_coef = coef.real if isinstance(coef, complex) else coef
                if term_dic.keys():
                    exp = 1
                    for index, value in term_dic.items():
                        exp = exp * ((-1) ** bit_form[index])
                    result += exp * real_coef
                else:
                    result += real_coef
        return result

    def _init_circuit(self, qlist):
        if self.init_circuit is None:
            return hadamard_circuit(qlist)
        else:
            return self.init_circuit(qlist)

    def _phase_circuit(self, qlist):
        return lambda gamma: pauli_z_operator_to_circuit(self.operator, qlist, gamma)[0]

    def _mixer_circuit(self, qlist):
        if self.mixer_circuit is None:
            # return lambda beta: RX(qlist, - beta * 2)
            def _mixer(beta):
                circuit = QCircuit()
                for qubit in qlist:
                    circuit << RX(qubit, - beta * 2)
                return circuit
            return _mixer
        else:
            return lambda beta: self.mixer_circuit(qlist, - beta * 2)

    def _qaoa_circuit(self, qlist, gammas, betas):
        """
        Given qubit list and parameters, return the QAOA circuit.

        Parameters
            qlist : ``list``\n
                qubit list\n

            gammas : ``array-like``\n
                parameter gamma for QAOA phase circuit\n

            betas : ``array-like``\n
                parameter beta for QAOA mixer circuit\n

        Return
            pyqpanda Circuit\n

        """
        if self.layer is None:
            self.layer = len(gammas)
        circuit = QCircuit()
        circuit << self._init_circuit(qlist)
        for i in range(self.layer):
            circuit << self._phase_circuit(qlist)(gammas[i]) << self._mixer_circuit(qlist)(betas[i])
        self.circuit = circuit
        return circuit

    def run_qaoa_circuit(self, gammas, betas, shots=-1):
        """
        Given parameters, run the qaoa circuit and get the theoretical probability distribution.

        Parameters
            gammas : ``array-like``\n
                Parameter gamma for QAOA phase circuit\n

            betas : ``array-like``\n
                Parameter beta for QAOA mixer circuit\n

            shots : ``integer``, ``optional``\n
                Times of running the same circuit. Must be positive integer or -1.
                If it is -1, the results are given as amplitudes of all state vectors,
                which can be viewed as running the circuit infinite times. Default is -1.

        Return
            prob_result : ``dict``\n
                Probability of each computational basis state. The keys are binary form
                of qubits where the first qubit sits at the right-most position and the
                items are the corresponding probability (if shots = -1) or frequency (if shots > 0).

        Example
            Run a two-layer QAOA algorithm circuit of problem :math:`f(\\vec{x})=2x_0x_1 + 3x_2 - 1` with parameters
            :math:`[0, 0, 0, 1, 1, 1]`

        .. code-block:: python

            import sympy as sp
            from pyqpanda_alg.QAOA.qaoa import *

            vars = sp.symbols('x0:3')
            f = 2*vars[0]*vars[1] + 3*vars[2] - 1
            qaoa_f = QAOA(f)

            gammas = [0, 0]
            betas = [1, 1]

            qaoa_result = qaoa_f.run_qaoa_circuit(gammas, betas, -1)
            print(qaoa_result)
            qaoa_result = qaoa_f.run_qaoa_circuit(gammas, betas, 500)
            print(qaoa_result)

        The codes above would give results like (with errors due to randomness):

        .. parsed-literal::
            {'000': 0.12500000000000008, '001': 0.12500000000000008, '010': 0.12500000000000008,
            '011': 0.12500000000000008, '100': 0.12500000000000008, '101': 0.12500000000000008,
            '110': 0.12500000000000008, '111': 0.12500000000000008}

            {'000': 0.132, '001': 0.134, '010': 0.112, '011': 0.136, '100': 0.094, '101': 0.13,
            '110': 0.122, '111': 0.14}

        """
        qvm = CPUQVM()
        qaoa_prog = QProg(self.problem_dimension)
        qlist = qaoa_prog.qubits()
        qaoa_prog << self._qaoa_circuit(qlist, gammas, betas)

        if shots == -1:
            qvm.run(qaoa_prog, shots=1)
            prob_result = qvm.result().get_prob_dict()
            prob_result = parse_quantum_result_dict(prob_result, qlist, select_max=-1)
        elif shots > 0:
            qaoa_prog << measure_all(qlist, qlist)
            # prob_result = qvm.run_with_configuration(qaoa_prog, clist, shots)
            qvm.run(qaoa_prog, shots=shots)
            prob_result = qvm.result().get_prob_dict(qlist)
            # for key in prob_result.keys():
            #     prob_result[key] = prob_result[key] / shots
        else:
            raise ValueError(f'Invalid shots number: {shots}')

        self.circuit_iter += 1

        return prob_result

    def _loss_function_default(self, measure_result):
        """
        Given a result, calculate the energy expectation.
        If measure type is sample, return :math:`E=\frac{1}{N_{\rm{shots}}}\sum_{i=0}^{2^n-1} n_iE_i`.

        If measure type is theoretical, return :math:`E=\sum_{i=0}^{2^n-1} p_iE_i`.


        Parameter
            measure_result : ``dict``\n
                measured result if measure type is sample, or probability distribution if measure type is theoretical.

        Return
            lost : ``float``\n
                energy expectation
        """
        lost = 0.
        for solution, hits in measure_result.items():
            if solution not in self.energy_dict:
                solution_list = [int(i) for i in solution[::-1]]
                self.energy_dict[solution] = self.calculate_energy(solution_list)
            prob = hits
            lost += prob * self.energy_dict[solution]
        return lost

    def _loss_function_cvar(self, measure_result):
        """
        Given a result, calculate the CVaR energy expectation.

        Parameter
            measure_result : ``dict``\n
                measured result if measure type is sample, or probability distribution if measure type is theoretical.

        Return
            lost : ``float``\n
                CVaR energy expectation
        """
        if not any(isinstance(self.alpha, t) for t in [int, float]):
            raise ValueError('CVaR method needs parameter alpha to be a number between 0~1')
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError('CVaR method needs parameter alpha to be a number between 0~1')
        cdf = 0.
        loss = 0.

        measure_result = sorted(measure_result.items(), key=lambda k: k[1], reverse=True)
        for solution, hits in measure_result:
            if solution not in self.energy_dict:
                solution_list = [int(i) for i in solution[::-1]]
                self.energy_dict[solution] = self.calculate_energy(solution_list)
            prob = hits
            if cdf < self.alpha:
                if cdf + prob < self.alpha:
                    loss += self.energy_dict[solution] * prob
                else:
                    loss += self.energy_dict[solution] * (self.alpha - cdf)
                cdf += prob
        return loss

    def _loss_function_Gibbs(self, measure_result):
        """
        Given a result, calculate the Gibbs energy expectation.

        Parameter
            measure_result : ``dict``\n
                measured result if measure type is sample, or probability distribution if measure type is theoretical.

        Return
            lost : ``float``\n
                Gibbs energy expectation
        """
        if not any(isinstance(self.temperature, t) for t in [int, float]):
            raise ValueError('Gibbs free energy method needs parameter temperature to be a number between 0~1')
        if self.temperature > 1 or self.temperature < 0:
            raise ValueError('Gibbs free energy method needs parameter temperature to be a number between 0~1')
        lost = 0.
        for solution, hits in measure_result.items():
            if solution not in self.energy_dict:
                solution_list = [int(i) for i in solution[::-1]]
                self.energy_dict[solution] = self.calculate_energy(solution_list)
            lost += hits * np.exp(-self.energy_dict[solution] / self.temperature)
        return - np.log(lost)

    def _loss_function(self, paras):
        """
        Given parameters, run the QAOA circuit and calculate the loss function.

        Parameter
            paras : ``array-like``\n
                parameters of :math:`p` layer QAOA circuit, with length :math:`2\times p`

        Return
            loss_f : ``float``\n
                lost function value
        """
        gammas = paras[:self.layer]
        betas = paras[self.layer:]

        result = self.run_qaoa_circuit(gammas, betas, self.shots)

        loss_dict = {'default': self._loss_function_default,
                     'CVaR': self._loss_function_cvar,
                     'Gibbs': self._loss_function_Gibbs}
        if self.loss_type not in loss_dict.keys():
            support_type = ', '.join(loss_dict.keys())
            raise ValueError('wrong loss types, only support ' + support_type)
        loss_f = loss_dict[self.loss_type](result)
        self.iter += 1
        return loss_f

    def _check_bounds(self, initial_para, gamma_bounds, beta_bounds):
        """
        check bounds and make new bounds

        Parameters
            initial_para : parameter to be bounded\n
            gamma_bounds : gamma bounds\n
            beta_bounds : beta bounds\n

        Returns
            new gamma bounds and new beta bounds

        """
        layer = len(initial_para) // 2
        if gamma_bounds is None:
            gamma_bounds = [(-np.inf, np.inf) for i in range(layer)]
        elif len(gamma_bounds) == 1:
            gamma_bounds = [gamma_bounds[0] for i in range(layer)]
        elif len(gamma_bounds) != 1 and len(gamma_bounds) < layer:
            raise ValueError('Shape of parameter bounds should be same with the layer')
        elif len(gamma_bounds) > layer:
            gamma_bounds = gamma_bounds[:layer]
        if beta_bounds is None:
            beta_bounds = [(-np.inf, np.inf) for i in range(layer)]
        elif len(beta_bounds) == 1:
            beta_bounds = [beta_bounds[0] for i in range(layer)]
        elif len(beta_bounds) != 1 and len(beta_bounds) < layer:
            raise ValueError('Shape of parameter bounds should be same with the layer')
        elif len(beta_bounds) > layer:
            beta_bounds = beta_bounds[:layer]
        return gamma_bounds, beta_bounds

    def _optimize_qaoa_parameter_default(self, initial_para, gamma_bounds, beta_bounds, **optimize_option):
        """
        Optimize QAOA algorithm parameters in traditional way.

        Parameters
            initial_para : ``array-like``\n
                initial parameters of :math:`p` layer QAOA circuit, with length :math:`2\\times p`

        Return
            final_para : ``array-like``\n
                optimized parameters of :math:`p` layer QAOA circuit, with length :math:`2\\times p`

        """
        gamma_bounds, beta_bounds = self._check_bounds(initial_para, gamma_bounds, beta_bounds)
        bounds = gamma_bounds + beta_bounds

        if self.optimizer == 'SPSA':
            final = spsa.spsa_minimize(self._loss_function, initial_para, bounds=bounds,
                                       **optimize_option)
            final_para = final
        else:
            final = minimize(self._loss_function, initial_para, bounds=bounds,
                             method=self.optimizer, **optimize_option, )
            final_para = final.x
        return final_para

    def _optimize_qaoa_parameter_interp(self, initial_para, start_layer=None, gamma_bounds=None, beta_bounds=None,
                                        **optimize_options):
        """
        Optimize QAOA algorithm parameters by interp method.

        Parameters
            initial_para : ``array-like``\n
                initial parameters of :math:`p_0` layer QAOA circuit, with length :math:`2\\times p_0`. :math:`p_0` is the
                start layer number of the optimization progress.

            start_layer : ``integer``, ``optional``\n
                the start layer number of the optimization progress. Default is 1.

        Return
            final_para : ``array-like``\n
                optimized parameters of :math:`p` layer QAOA circuit, with length :math:`2\\times p`

        """
        if start_layer is None:
            start_layer = 1
        if not isinstance(start_layer, int):
            raise ValueError('start layer number should be a positive integer')
        if start_layer <= 0 or start_layer > self.layer:
            raise ValueError('start layer number should be a positive integer and less than final layer')

        final_layer = self.layer

        if initial_para is None:
            initial_para = np.random.random(start_layer * 2)
        for layer in range(start_layer, final_layer):
            initial_para = initial_para[:2*layer]
            new_gamma_bounds, new_beta_bounds = self._check_bounds(initial_para, gamma_bounds, beta_bounds)
            self.layer = layer
            opt_para = self._optimize_qaoa_parameter_default(initial_para, new_gamma_bounds, new_beta_bounds,
                                                             **optimize_options)
            initial_para = parameter_interpolate(opt_para)
        self.layer = final_layer
        new_gamma_bounds, new_beta_bounds = self._check_bounds(initial_para, gamma_bounds, beta_bounds)
        opt_para = self._optimize_qaoa_parameter_default(initial_para, new_gamma_bounds, new_beta_bounds,
                                                         **optimize_options)
        return opt_para

    def _check_layer_and_generate_initial_para(self, layer=1, initial_para=None):
        """
        generate the proper initial layer and initial parameters.
        """
        if not isinstance(layer, int) or layer <= 0:
            raise ValueError('layer number must be a positive integer')
        if self.optimize_type == 'default':
            if initial_para is None:
                self.layer = layer
                initial_para = np.random.random(self.layer * 2) * np.pi
                return initial_para, None
            else:
                para_layer = len(initial_para) // 2
                if para_layer == layer:
                    self.layer = layer
                    return initial_para[:2 * layer], None
                elif para_layer > layer:
                    self.layer = layer
                    para = np.concatenate((initial_para[:layer], initial_para[para_layer:para_layer+layer]))
                    return para, None
                elif para_layer < layer:
                    self.layer = layer
                    para = np.concatenate((initial_para[:para_layer], np.random.random(layer - para_layer) * np.pi,
                                          initial_para[para_layer:], np.random.random(layer - para_layer) * np.pi))
                    return para, None

        if self.optimize_type == 'interp':
            if initial_para is None:
                raise ValueError('Interpolation method requires initial parameters.')
            else:
                self.layer = layer
                para_layer = len(initial_para) // 2
                if para_layer >= self.layer:
                    raise ValueError('Interpolation method initial parameter size should less than final layer.')
                else:
                    start_layer = para_layer
                    return initial_para, start_layer

    def run(self, layer=1, initial_para=None, shots=-1, loss_type=None, optimize_type=None, optimizer=None,
            optimizer_option=None, **loss_option):
        """
        Optimize the function by QAOA algorithm.

        Parameters
            layer : ``integer``, ``optional``\n
                Layers number of QAOA circuit. Default is 1.
                If optimize type is interp, then it represents the final layer of the optimization progress.

            initial_para : ``array-like``, ``optional``\n
                initial parameters of :math:`p` layer QAOA circuit, with length :math:`2\\times p`. If not given, a random
                distribution from :math:`U(0, \pi)` of size :math:`2p` is generated.

            shots : ``integer``, ``optional``\n
                Circuit measured times. If shots takes -1, then use theoretical probability (by state vector) instead.
                Default is -1

            loss_type : ``string``, ``optional``\n
                The loss function used by the optimizer. Should be one of

                    - ``default`` : Given a result, calculate the energy expectation.\n
                        See Note ``Energy expectation``
                    - ``Gibbs`` : Given a result and argument temperature :math:`T`, calculate the Gibbs energy expectation.\n
                        See Note ``Gibbs energy``
                    - ``CVaR`` : Given a result and argument :math:`\\alpha`, calculate the CVaR loss function.\n
                        See Note ``CVaR loss functio``

                If not given, default by ``default``.

            optimize_type : ``string``, ``optional``\n
                The method to optimize the QAOA circuit. Should be one of

                    - ``default``: Directly optimize the :math:`p` layer QAOA circuit.\n
                    - ``interp``: Use interpolate method to train a big QAOA circuit.\n
                        See Note ``interp method``

                If not given, default by ``default``.

            optimizer : ``string``, ``optional``\n
                Type of solver. Should be one of

                    - ``SPSA`` : See ``spsa.spsa_minimize``\n

                    - one of ``Nelder-Mead``, ``Powell``, ``CG``, ``BFGS``, ``Newton-CG``, ``TNC``, ``COBYLA``, ``SLSQP``,
                    ``trust-constr``, ``dogleg``, ``trust-ncg``, ``trust-exact``, ``trust-krylov``.
                    See ``scipy.optimize.minimize``.

                If not given, default by ``SLSQP``.

            optimizer_option : ``dict``, ``optional``\n
                A dictionary of solver options. Accept the following generic options:\n
                    - bounds : ``List[tuple]``, ``optional``\n
                        Bounds for the variables. Sequence of ``(min, max)`` pairs for each element in `x`.
                        If specified, variables are clipped to fit inside the bounds after each iteration.
                        None is used to specify no bound.
                    - options : ``integer``\n
                        Maximum number of iterations to perform. Depending on the
                        method each iteration may use several function evaluations.

                        For `TNC` use `maxfun` instead of `maxiter`.

            loss_option :\n

                temperature : ``float``, ``optional``\n
                    parameter calculated in _loss_function_Gibbs. Default is 1. See Note ``Gibbs energy``.

                alpha : ``float``, ``optional``\n
                    parameter calculated in _loss_function_cvar. Default is 1. See Note ``Gibbs energy``.

        Return
            qaoa_result : ``dict``\n
                dict of all possible solutions with corresponding probabilities.
                The elements are arranged in descending order of probability.

            para_result : ``array-like``\n
                Array of the optimized QAOA parameters.

            loss_result : ``float``\n
                Loss function value of the optimized QAOA parameters.

        Example
            Run a two-layer QAOA algorithm circuit of problem :math:`f(\\vec{x})=2x_0 + x_1 + 3x_2 - 1` with parameters
            ``[0, 0, 0, 1, 1, 1]``

        .. code-block:: python

            import sympy as sp
            from pyqpanda_alg.QAOA import qaoa
            vars = sp.symbols('x0:3')
            f = 2*vars[0]*vars[1] + 3*vars[2] - 1
            qaoa_f = qaoa.QAOA(f)

            qaoa_result = qaoa_f.run(layer=3)
            sorted_result = sorted(qaoa_result[0].items(), key=lambda k:k[1], reverse=True)[:5]
            print(sorted_result[:5])

        .. parsed-literal::
            [('000', 0.6848946054168573),
             ('010', 0.1575526972909123), 
             ('001', 0.15755269729091226), 
             ('100', 7.957749518311524e-13), 
             ('110', 1.8305953815081342e-13)]


        Notes
            - Energy expectation:\n
                In traditional QAOA algorithm, the parameter is optimized by minimize the energy expectation

                :math:`\\bra{\psi(\gamma, \\beta}H\ket{\psi(\gamma, \\beta)}`

                If measure type is sample, it is calculated by

                :math:`E=\\frac{1}{N_{\\rm{shots}}}\sum_{i=0}^{2^n-1} n_iE_i`.


                If measure type is theoretical, it is calculated by

                :math:`E=\sum_{i=0}^{2^n-1} p_iE_i`.

            - Gibbs energy:\n

                Inspired by Ref[1]. Instead of the traditional energy expectation value, using the Gibbs function as the
                object function. The function is

                    :math:`f_G=-\ln \langle e^{-E/T}\\rangle`

                Here :math:`T` is the hyperparameter of temperature, as :math:`T` decreases, the weight of the lower
                energy states in the loss function then becomes larger. When :math:`T=1`, it turns back to energy
                expectation function.

                If measure type is sample, it is calculated by

                    :math:`G=-\log (\\frac{1}{N_{\\rm{shots}}}\sum_{i=0}^{2^n-1} n_i \exp(-E_i/T))`.

                If measure type is theoretical, it is calculated by

                    :math:`G=-\log (\sum_{i=0}^{2^n-1} p_i \exp(-E_i/T))`.


            - CVaR loss function:\n
                Inspired by Ref[3].Instead of the traditional energy expectation value, using the Conditional Value at
                Risk function as the object function. The function is

                    :math:`CVaR_\\alpha(X) = \mathbb{E}[X|X\leq F_X^{-1}(alpha)]`

                Here :math:`\\alpha` is the confidence level. CVaR is the expected value of the lower α-tail of the
                distribution of X. :math:`\\alpha=0` corresponds to the minimum, and :math:`\\alpha=1` corresponds to the
                expectation value.

                If measure type is sample, it is calculated by

                    :math:`E=\\frac{1}{\\alpha N}(\sum_{i=0}^{k} n_iE_i + (\\alpha N - n_{k+1})E_{k+1}),\sum_{i=0}^k n_i < \\alpha N`

                If measure type is theoretical, it is calculated by

                    :math:`E=\sum_{i=0}^{k} p_iE_i + (\\alpha - p_{k+1})E_{k+1}, \sum_{i=0}^k p_i < \\alpha`

            - Interpolate method:\n
                Inspired by Ref[2].

        Reference
            [1] LI L, FAN M, CORAM M, et. Quantum Optimization with a Novel Gibbs Objective Function and Ansatz
            Architecture Search[J/OL]. Physical Review Research, 2020, 2(2): 023074. DOI:10.1103/PhysRevResearch.2.023074.\n

            [2] ZHOU L, WANG S T, CHOI S, et. Quantum Approximate Optimization Algorithm:
            Performance, Mechanism, and Implementation on Near-Term Devices[J/OL].
            Physical Review X, 2020, 10(2): 021067. DOI:10.1103/PhysRevX.10.021067.\n

            [3] BARKOUTSOS P K, NANNICINI G, ROBERT A, et. Improving Variational Quantum Optimization using CVaR[J/OL].
            Quantum, 2020, 4: 256. DOI:10.22331/q-2020-04-20-256.\n



        """

        if loss_type is None:
            loss_type = 'default'
        self.loss_type = loss_type

        if optimize_type is None:
            optimize_type = 'default'
        self.optimize_type = optimize_type

        self.optimizer = optimizer

        self.alpha = loss_option.get('alpha', 1)
        self.temperature = loss_option.get('temperature', 1)
        self.shots = shots

        initial_para, start_layer = self._check_layer_and_generate_initial_para(layer, initial_para)

        if optimizer_option is None:
            optimizer_option = {}

        gamma_bounds = loss_option.get('gamma_bounds', None)
        beta_bounds = loss_option.get('beta_bounds', None)

        if self.optimize_type == 'default':
            para_result = self._optimize_qaoa_parameter_default(initial_para, gamma_bounds, beta_bounds,
                                                                **optimizer_option)
        elif self.optimize_type == 'interp':

            para_result = self._optimize_qaoa_parameter_interp(initial_para, start_layer, gamma_bounds, beta_bounds,
                                                               **optimizer_option)
        else:
            raise ValueError('wrong optimize type, only support default, interp')

        qaoa_result = self.run_qaoa_circuit(para_result[:self.layer], para_result[self.layer:], self.shots)
        loss_result = self._loss_function(para_result)
        qaoa_result_list = sorted(qaoa_result.items(), key=lambda k: k[1], reverse=True)
        keys = [i[0] for i in qaoa_result_list]
        items = [i[1] for i in qaoa_result_list]
        qaoa_result = dict(zip(keys, items))

        return qaoa_result, para_result, loss_result
