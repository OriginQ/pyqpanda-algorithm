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

from typing import Union, Optional, List

import numpy as np
from pyqpanda3.core import QCircuit, U1, Z
from pyqpanda3.hamiltonian import PauliOperator
import sympy as sp

from .. plugin import *

from pyqpanda_alg.Grover import Grover,GroverAdaptiveSearch
from pyqpanda_alg.QAOA import qaoa

def _quadratic_func_to_coeff(quadratic_func):
    free_syms = list(quadratic_func.free_symbols)
    if not free_syms:
        raise ValueError("Unable to create polynomial from given function: no variables found.")

    quadratic_func = sp.Poly(quadratic_func)
    uni = quadratic_func.monoms()
    coeffs = quadratic_func.coeffs()
    x = list(quadratic_func.gens)

    constant = np.zeros(1)
    linear = np.zeros(len(x))
    quadratic = np.zeros((len(x), len(x)))

    for i, unit in enumerate(uni):
        tmp = _get_index(unit, 1)
        tmp2 = _get_index(unit, 2)
        if len(tmp2) > 0:
            linear[tmp2[0]] += coeffs[i]
        elif len(tmp) == 0:
            constant[0] = coeffs[i]
        elif len(tmp) == 1:
            linear[tmp[0]] += coeffs[i]
        else:
            quadratic[tmp[0]][tmp[1]] = coeffs[i]
    return quadratic, linear, constant[0]


class QuadraticBinary:
    """
    Represent a quadratic form and compute the function value using a quantum circuit

    .. math::

        Q(x) = x^T A x + x^T b + c

    .. math::

        |x\\rangle_n |0\\rangle_m \mapsto |x\\rangle_n |(Q(x) + 2^m) \mod 2^m \\rangle_m

    According to the above formula, a negative value can also be represent by this
    method using two's complement.

    Parameters
        problem : ``sympy.Basic`` or ``dict``
            A quadratic form function with binary variables to be optimized. Support an expression in sympy.
            Keys followed should be included if expression in dict:

            ``quadratic`` : A, Optional ``[Union[np.ndarray, List[List[float]]]]`` , the quadratic coefficients matrix.
            ``linear`` : b, Optional ``[Union[np.ndarray, List[float]]]`` , the linear coefficients array.
            ``constant`` : c, ``float``, a constant.\n

    """

    def __init__(self, problem):
        if isinstance(problem, sp.Basic):
            quadratic, linear, constant = _quadratic_func_to_coeff(problem)
        elif isinstance(problem, dict):
            quadratic, linear, constant = problem['quadratic'], problem['linear'], problem['constant']
        else:
            raise ValueError('Input error.')

        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError('Quadratic size should be eqaul to linear size')
        if quadratic is None and linear is not None:
            quadratic = [[0] * len(linear)] * len(linear)
        if linear is None and quadratic is not None:
            linear = [0] * len(quadratic)
        if linear is None and quadratic is None:
            linear = [0]
            quadratic = [0]

        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

    def query_qnumber(self) -> List[int]:
        """
        Returns
            [n_key, n_res] : ``list[int]``
                Returns the size(number of qubits) of the variable and result registers for the given problem.

        Examples
            An example for function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        >>> from pyqpanda_alg import QUBO
        >>> import sympy as sp
        >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
        >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        >>> test0 = QUBO.QuadraticBinary(function)
        >>> n_key, n_res = test0.query_qnumber()
        >>> print(n_key, n_res)
        3 3

        """
        n_key = np.max([1, len(self.linear), len(self.quadratic)])

        bounds = []

        def pos(x): return x > 0
        def neg(x): return x < 0

        max_val = 0
        max_val += sum(sum(q_ij for q_ij in q_i if pos(q_ij)) for q_i in self.quadratic)
        max_val += sum(l_i for l_i in self.linear if pos(l_i))
        max_val += self.constant if pos(self.constant) else 0

        min_val = 0
        min_val += sum(sum(q_ij for q_ij in q_i if neg(q_ij)) for q_i in self.quadratic)
        min_val += sum(l_i for l_i in self.linear if neg(l_i))
        min_val += self.constant if neg(self.constant) else 0

        pos_bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 0
        neg_bits = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val < 0 else 0
        n_res = max(pos_bits, neg_bits)

        return [n_key, n_res]

    def cir(self, q_key, q_res):
        """
        Parameters
            q_key : ``QVec``
                Qubit(s) for the variable register.
            q_res : ``QVec``
                Qubit(s) for the result register.

        Returns
            main_cir : ``QCircuit``
                Returns the quantum circuit for computing the function.

        Examples
            An example for function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        >>> from pyqpanda_alg import QUBO
        >>> import sympy as sp
        >>> import numpy as np

        >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
        >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        >>> test0 = QUBO.QuadraticBinary(function)
        >>> n_key, n_res = test0.query_qnumber()
        >>> n = n_key+n_res
        >>> q = list(range(n))
        >>> q_key = q[:n_key]
        >>> q_res = q[n_key:]
        >>> print(test0.cir(q_key, q_res))
        
        .. parsed-literal::
            q_0:  \|0>──── ───────■────── ───────■─────────────── ─────────────────────── ───────■────────────────────── >
                                 │              │                                               │                       >
            q_1:  \|0>──── ───────┼────── ───────┼───────■─────── ───────■─────────────── ───────■────────────────────── >
                                 │              │       │               │                       │                       >
            q_2:  \|0>──── ───────┼────── ───────┼───────┼─────── ───────┼───────■─────── ───────┼──────────────■─────── >
                      ┌─┐ ┌──────┴─────┐        │┌──────┴──────┐        │┌──────┴──────┐ ┌──────┴──────┐       │        >
            q_3:  \|0>─┤H├ ┤U1(2.042035)├ ───────┼┤U1(-1.570796)├ ───────┼┤U1(-0.785398)├ ┤U1(-1.884956)├───────┼─────── >
                      ├─┤ └────────────┘ ┌──────┴┴────┬────────┘ ┌──────┴┴─────┬───────┘ └─────────────┘┌──────┴──────┐ >
            q_4:  \|0>─┤H├ ────────────── ┤U1(4.084070)├───────── ┤U1(-3.141593)├──────── ───────────────┤U1(-1.570796)├ >
                      └─┘                └────────────┘          └─────────────┘                        └─────────────┘ >

            
            q_0:  \|0>───────■─────── ────────────── ────────────── ─ ─── ────────────────── ───
                            │
            q_1:  \|0>───────■─────── ───────■────── ───────■────── ─ ─── ────────────────── ───
                            │               │              │
            q_2:  \|0>───────┼─────── ───────■────── ───────■────── ─ ─── ────────────────── ───
                            │        ┌──────┴─────┐        │         ┌─┐
            q_3:  \|0>───────┼─────── ┤U1(1.413717)├ ───────┼────── X ┤H├ ─────────■──────── ───
                     ┌──────┴──────┐ └────────────┘ ┌──────┴─────┐ │ └─┘ ┌────────┴───────┐ ┌─┐
            q_4:  \|0>┤U1(-3.769911)├ ────────────── ┤U1(2.827433)├ X ─── ┤CR(1.570796).dag├ ┤H├
                     └─────────────┘                └────────────┘       └────────────────┘ └─┘

        """
        n_key = len(q_key)
        n_res = len(q_res)

        factor = np.pi * 2 ** (1 - n_res)
        main_cir = QCircuit()
        main_cir << hadamard_circuit(q_res)

        if self.constant != 0:
            for i, q_i in enumerate(q_res):
                main_cir << U1(q_i, factor * 2 ** i * self.constant)

        for i in range(n_key):
            linear = self.linear[i] if self.linear is not None else 0
            linear += self.quadratic[i][i] if self.quadratic is not None else 0
            if linear != 0:
                for j, q_j in enumerate(q_res):
                    main_cir << U1(q_j, factor * 2 ** j * linear).control(q_key[i])

        if self.quadratic is not None:
            for j in range(n_key):
                for k in range(j + 1, n_key):
                    value = self.quadratic[j][k] + self.quadratic[k][j]
                    if value != 0:
                        for i, q_i in enumerate(q_res):
                            main_cir << U1(q_i, factor * 2 ** i * value).control([q_key[j], q_key[k]])

        main_cir << QFT(q_res).dagger()
        return main_cir

    def function_value(self, var_array):
        """
        Parameters
            var_array : ``array_like``
                An array of binary values.

        Returns
            res : ``float``
                The result of the function under given variables array.

        Examples
            An example for function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        >>> from pyqpanda_alg import QUBO
        >>> import sympy as sp
        >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
        >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        >>> test0 = QUBO.QuadraticBinary(function)
        >>> # calculate the quadratic function value above with x0, x1, x2= 0, 1, 0
        >>> print(test0.function_value([0, 1, 0]))
        -1.0

        """
        linear = sum(self.linear * np.array(var_array))
        quadratic = float(0)
        for i in range(len(var_array)):
            for j in range(len(var_array)):
                quadratic += float(self.quadratic[i][j] * var_array[i] * var_array[j])
        res = self.constant + linear + quadratic
        return float(res)

    def qubobytraversal(self):
        """
        Traversing the entire solution space to find the minimum value solution.

        Returns
            index_list, min_value : ``list``, ``float``
                The solution obtained by traversing the entire solution space.

        Examples
            An example for function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        >>> from pyqpanda_alg import QUBO
        >>> import sympy as sp
        >>> import numpy as np
        >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
        >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        >>> test0 = QUBO.QuadraticBinary(function)
        >>> # find the minimum function value by traversing
        >>> res0 = test0.qubobytraversal()
        >>> print('result of traversal: ', res0)
        result of traversal:  ([[0, 1, 0]], -1.0)

        """
        
        process_show = False
        min_value = float(np.inf)
        min_index = []
        value_dict = {}
        
        n = len(self.linear)
        
        for i in range(int(2 ** n)):
            index = _bit_to_list(i, n)
            value = float(self.function_value(index))
            value_dict[str(index)] = value
            if value < min_value:
                min_value = float(value)
                min_index = [index]
            elif value == min_value:
                min_index.append(index)
        index_list = min_index
        if process_show:
            print(value_dict)
        return index_list, float(min_value)


def _bit_to_list(t, n):
    res = [0 for i in range(n)]
    i = -1
    while t != 0:
        res[i] = t % 2
        t = t >> 1
        i -= 1
    return res


def _get_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def _pre_init_select(quadratic, linear, constant):
    res = constant
    m = 1 / len(quadratic)
    for i in linear:
        res += i * m

    for i in range(len(quadratic)):
        res += quadratic[i][i] * m
    for i in range(len(quadratic)):
        for j in range(i + 1, len(quadratic)):
            res += quadratic[i][j] * m ** 2
    return res


class QUBO_GAS_origin(QuadraticBinary):

    """

    Represent a quadratic unconstrained binary optimization problem 
    and solve it using the Grover Adaptive Search.

    Inheritance class of QuadraticBinary. Using GAS to find the minimum value solution
    of given quadratic binary optimization problem.

    .. math::
        Q(x) = x^T A x + x^T b + c

    .. math::
        |x\\rangle_n |0\\rangle_m \mapsto |x\\rangle_n |(Q(x) + 2^m) \mod 2^m \\rangle_m

    According to the above formula, a negative value can also be represent by this
    method using two's complement.

    Parameters
        problem : ``sympy.Basic`` or ``dict``
            A quadratic form function with binary variables to be optimized. Support an expression in sympy.
            Keys followed should be included if expression in dict:

            ``quadratic`` : A, Optional ``[Union[np.ndarray, List[List[float]]]]`` , the quadratic coefficients matrix.
            ``linear`` : b, Optional ``[Union[np.ndarray, List[float]]]`` , the linear coefficients array.
            ``constant`` : c, ``float`` , a constant.\n
    >>> from pyqpanda_alg import QUBO
    >>> import sympy as sp
    >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
    >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
    >>> # find the minimum function value using GAS
    >>> test1 = QUBO.QUBO_GAS_origin(function)
    >>> res1 = test1.run(init_value=0, continue_times=10, process_show=False)
    >>> print('result of Grover adaptive search: ', res1)
    result of Grover adaptive search:  ([[0, 1, 0]], -1.0)

    """

    def __init__(self, problem):
        super(QUBO_GAS_origin, self).__init__(problem)
        self.init_value = self.constant

    def _flip_oracle_function(self, q_index_value, current_min):
        n_index = len(self.quadratic)
        q_index = q_index_value[:n_index]
        q_value = q_index_value[n_index:]
        circuit = QCircuit()
        constant = self.constant - current_min
        quadratic_form = QuadraticBinary({'quadratic': self.quadratic, 'linear': self.linear, 'constant': constant})
        circuit << quadratic_form.cir(q_index, q_value) << Z(q_value[-1]) << quadratic_form.cir(q_index,
                                                                                                   q_value).dagger()
        return circuit

    def _n_value_function(self, current_min):
        quadratic_form = QuadraticBinary({'quadratic': self.quadratic, 'linear': self.linear,
                                          'constant': self.constant - current_min})
        n_value = quadratic_form.query_qnumber()[1]
        return n_value

    def _value_function(self, var_array):
        var_array = list(map(int, var_array))[::-1]
        var_linear = sum(np.array(self.linear) * np.array(var_array))
        var_quadratic = 0
        for i in range(len(var_array)):
            for j in range(len(var_array)):
                var_quadratic += self.quadratic[i][j] * var_array[i] * var_array[j]
        value = self.constant + var_linear + var_quadratic
        return value

    def run(self, continue_times: int = 5, init_value=None, process_show=False):
        """
        Run the solver to find the minimum.

        Parameters
            continue_times : ``int``
                The maximum number of repeated searches at the current optimal point in GAS algorithm.
            init_value : ``float``
                The given initial value of the optimization function. Default the constant item of the problem.
            process_show : ``bool``
                Set to True to print the detail during search.

        Returns
            minimum_indexes, minimum_res : ``list[list[int]]``, ``float``
                The optimization result including the solution array and the optimal value.

        Examples
            An example for minimization of quadratic binary function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2


        """
        n_index = len(self.quadratic)
        init = init_value if init_value is not None else self.constant
        gas_model = GroverAdaptiveSearch(init_value=init,
                                                n_index=n_index,
                                                oracle_circuit=self._flip_oracle_function)

        res = gas_model.run(continue_times=continue_times,
                            n_value_function=self._n_value_function,
                            value_function=self._value_function,
                            process_show=process_show)

        return res


class QUBO_QAOA(QuadraticBinary):
    """

    Represent a quadratic unconstrained binary optimization problem 
    and solve it using the Quantum Approximate Optimization Algorithm.

    Inheritance class of QuadraticBinary. Using QAOA to find the minimum value solution
    of given quadratic binary optimization problem.

    .. math::
        Q(x) = x^T A x + x^T b + c

    .. math::
        |x\\rangle_n |0\\rangle_m \mapsto |x\\rangle_n |(Q(x) + 2^m) \mod 2^m \\rangle_m

    According to the above formula, a negative value can also be represent by this
    method using two's complement.

    Parameters
        problem : ``sympy.Basic`` or ``dict``
            A quadratic form function with binary variables to be optimized. Support an expression in sympy.
            Keys followed should be included if expression in dict:

            ``quadratic`` : A, Optional ``[Union[np.ndarray, List[List[float]]]]``, the quadratic coefficients matrix.
            ``linear`` : b, Optional ``[Union[np.ndarray, List[float]]]``, the linear coefficients array.
            ``constant`` : c, ``float``, a constant.\n

    >>> from pyqpanda_alg import QUBO
    >>> import sympy as sp
    >>> x0, x1, x2 = sp.symbols('x0 x1 x2')
    >>> function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
    >>> # find the minimum function value using QAOA
    >>> test2 = QUBO.QUBO_QAOA(function)
    >>> res2 = test2.run(layer=5, optimizer='SLSQP',
    >>>                  optimizer_option={'options':{'eps':1e-3}})
    >>> print('result of QAOA: ', res2)
    result of QAOA:  {'000': 0.0004125955977882364, '001': 0.020540129989231624, '010': 0.9152063391500159, '011': 0.003439453872904533, '100': 6.389251180087758e-05, '101': 0.0013381332738120826, '110': 0.034300546853266084, '111': 0.024698908751180276}

    """

    def __init__(self, problem):
        super(QUBO_QAOA, self).__init__(problem)

    def submit(self, layer=None, optimizer='SLSQP', optimizer_option=None, *,
               backend=None, execution_options=None):
        """
        Run the solver to find the minimum and return a resumable algorithm task.

        The synchronous :meth:`run` drives this task to completion; the
        backend and execution_options parameters are keyword-only.

        Parameters
            layer : ``int``
                Layers number of QAOA circuit.
                If optimize type is interp, then it represents the final layer of the optimization progress.
            optimizer : ``str``, ``optional``
                Type of solver. Should be one of

                    - ``SPSA`` : See ``spsa_minimize`` in the module ``pyqpanda_alg.QAOA.spsa``
                    - one of  ``['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr','dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']``. See ``scipy.optimize.minimize``.

                If not given, default by ``SLSQP``.
            optimizer_option : ``dict``, ``optional``
                A dictionary of solver options. Accept the following generic options:
                    - bounds : ``List[tuple]``, ``optional``
                        Bounds for the variables. Sequence of ``(min, max)`` pairs for each element in ``x``.
                        If specified, variables are clipped to fit inside the bounds after each iteration.
                        None is used to specify no bound.
                    - options : ``int``
                        Maximum number of iterations to perform. Depending on the
                        method each iteration may use several function evaluations.

                        For ``TNC`` use ``maxfun`` instead of ``maxiter``.
            backend : ``ExecutionBackend``, ``optional``
                The execution backend to run on. Keyword-only. If not given, the CPU
                LocalBackend is used.
            execution_options : ``ExecutionOptions``, ``optional``
                Options controlling the backend submissions. Keyword-only. If not
                given, defaults are derived from the delegated QAOA run.

        Returns
            task : ``AlgorithmTask``
                Resumable algorithm task holding the optimization run.
                ``task.result()`` returns ``(qaoa_result, para_result, loss_result)``.

        Examples
            An example for minimization of quadratic binary function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        """
        H_linear = 0 * PauliOperator({"" : 1})
        n_key = len(self.quadratic)
        for i in range(n_key):
            linear_i = self.linear[i] if self.linear is not None else 0
            linear_i += self.quadratic[i][i] if self.quadratic is not None else 0
            H_linear += linear_i * qaoa.p_1(n_key - 1 - i)
        H_quadratic = 0 * PauliOperator({"" : 1})
        if self.quadratic is not None:
            for j in range(n_key):
                for k in range(j + 1, n_key):
                    quadratic_jk = self.quadratic[j][k] + self.quadratic[k][j]
                    H_quadratic += quadratic_jk * qaoa.p_1(n_key - 1 - j) * qaoa.p_1(n_key - 1 - k)
        H_constant = PauliOperator({"" : self.constant})
        H = H_linear + H_quadratic + H_constant

        qaoa_model = qaoa.QAOA(problem=H)
        return qaoa_model.submit(layer=layer, loss_type='default', optimize_type='default',
                                 optimizer=optimizer, optimizer_option=optimizer_option,
                                 backend=backend, execution_options=execution_options)

    def run(self, layer=None, optimizer='SLSQP', optimizer_option=None, *,
            backend=None, execution_options=None):
        """
        Run the solver to find the minimum.

        Parameters
            layer : ``int``
                Layers number of QAOA circuit.
                If optimize type is interp, then it represents the final layer of the optimization progress.
            optimizer : ``str``, ``optional``
                Type of solver. Should be one of

                    - ``SPSA`` : See ``spsa_minimize`` in the module ``pyqpanda_alg.QAOA.spsa``
                    - one of  ``['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr','dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']``. See ``scipy.optimize.minimize``.

                If not given, default by ``SLSQP``.
            optimizer_option : ``dict``, ``optional``
                A dictionary of solver options. Accept the following generic options:
                    - bounds : ``List[tuple]``, ``optional``
                        Bounds for the variables. Sequence of ``(min, max)`` pairs for each element in ``x``.
                        If specified, variables are clipped to fit inside the bounds after each iteration.
                        None is used to specify no bound.
                    - options : ``int``
                        Maximum number of iterations to perform. Depending on the
                        method each iteration may use several function evaluations.

                        For ``TNC`` use ``maxfun`` instead of ``maxiter``.

        Returns
            qaoa_result : ``dict``
                Mapping of each possible solution (bit string) to its
                corresponding probability.  The solution of the problem we are
                looking for should generally be the maximum probability.

        Examples
            An example for minimization of quadratic binary function = -0.5 * x0 * x1 - 0.7 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2

        """
        task = self.submit(layer=layer, optimizer=optimizer, optimizer_option=optimizer_option,
                           backend=backend, execution_options=execution_options)
        return task.result()[0]
