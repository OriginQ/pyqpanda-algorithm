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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, Z, X, H, BARRIER
import numpy as np

from .. plugin import *

class Grover:
    """ This class provides a framework for Grover Search algorithm [1].

    Parameters
        in_operator : callable ``f(qubits)``\n
            Operator/Circuit of the initial search state for the algorithm, default Hadamards.
        flip_operator : callable ``f(qubits)``\n
            Operator/Circuit of marking the good states by phase-flip. Default doing a pauli-Z
            gate at the last qubit.
        zero_flip : callable ``f(qubits)``\n
            Operator/Circuit of reflects 0s by phase-flip. Default doing a zero-controled pauli-Z
            gate on qubits.
        mark_data : ``str``, ``list[str]``\n
            Marked target state. Default None.
            Only used when simply marking a known query state, as the designed flip_operator part.
        amplify_operator : callable ``f(qubits)``\n
            Constructed complete Grover amplitude amplification operator circuit. Default None.
            For users' special designed amplitude amplification operator.

    References
        [1] L. K. Grover, A fast quantum mechanical algorithm for database search. Proceedings
        28th Annual Symposium on the Theory of Computing (STOC) 1996, pp. 212-219.
        https://arxiv.org/abs/quant-ph/9605043

    """
    def __init__(self,
                 in_operator=None,
                 flip_operator=None,
                 zero_flip=None,
                 mark_data=None,
                 amplify_operator=None):
        self.in_operator = hadamard_circuit if in_operator is None else in_operator

        if flip_operator is None and mark_data is not None:
            def s_f(q_num):
                return mark_data_reflection(q_num, mark_data)
            self.flip_operator = s_f
        else:
            self.flip_operator = flip_operator
        self.u_s = zero_flip
        self.amplify = amplify_operator

    def cir(self, q_input=None, q_flip=None, q_zero=None, iternum: int = 1):
        """
        Get full circuit of Grover search.

        Parameters
            q_input : ``QVec``\n
                Target qubit(s) for in_operator (initial preparation circuit).
                Using Hadamard gates to create the uniform superposition at the beginning most of time.
                Although in most simple cases it includes the full workspace qubits,
                auxiliary qubits can be excluded when dealing with some complex problems.
            q_flip : ``QVec``\n
                Target qubit(s) for flip_operator.
            q_zero : ``QVec``\n
                Target qubit(s) for zero_flip.
            iternum : ``int``\n
                The number of iterations. In another word number of repetition of applying the Grover operator.

        Returns
            circuit : ``QCircuit``\n
                Full quantum circuit for given Grover search.

        Examples
            An example for implementing an Grover search for state where q_0 `and` q_1 is 1.

        >>> from pyqpanda3.core import CPUQVM, QCircuit, Z, TOFFOLI
        >>> from pyqpanda_alg import Grover
        >>> m = CPUQVM()
        >>> q_state = list(range(3))

        >>> def mark(qubits):
        >>>     cir = QCircuit()
        >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
        >>>     cir << Z(qubits[2])
        >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
        >>>     return cir

        >>> demo_search = Grover.Grover(flip_operator=mark)
        >>> prog = QProg()
        >>> prog << demo_search.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=1)
        >>> m.run(prog,1000)
        >>> res = m.result().get_prob_dict(q_state[:2])
        >>> print(res)
        >>> print(prog)
        {'00': 0.0, '01': 0.0, '10': 0.0, '11': 1.0000000000000004}

        .. parsed-literal::
                      ┌─┐             ┌─┐ ┌─┐     ┌─┐ ┌─┐
            q_0:  |0>─┤H├ ─■─ ─── ─■─ ┤H├ ┤X├ ─■─ ┤X├ ┤H├
                      ├─┤  │       │  ├─┤ ├─┤ ┌┴┐ ├─┤ ├─┤
            q_1:  |0>─┤H├ ─■─ ─── ─■─ ┤H├ ┤X├ ┤Z├ ┤X├ ┤H├
                      └─┘ ┌┴┐ ┌─┐ ┌┴┐ └─┘ └─┘ └─┘ └─┘ └─┘
            q_2:  |0>──── ┤X├ ┤Z├ ┤X├ ─── ─── ─── ─── ───
                          └─┘ └─┘ └─┘

        """
        if not hasattr(q_input, '__len__'):
            q_input = [q_input]

        if q_flip is None:
            q_flip = q_input
        if not hasattr(q_flip, '__len__'):
            q_flip = [q_flip]

        if q_zero is None:
            q_zero = q_input
        if not hasattr(q_zero, '__len__'):
            q_zero = [q_zero]

        q_all = list(set(q_input + q_zero + q_flip))
        if self.amplify is not None:
            amp_cir = self.amplify(q_all)
        else:
            amp_cir = amp_operator(q_input, q_flip, q_zero, self.in_operator, self.flip_operator, self.u_s)

        circuit = QCircuit()
        circuit << self.in_operator(q_input)
        while iternum > 0:
            circuit << amp_cir
            iternum -= 1
        return circuit


def iter_num(q_num, sol_num):
    """
    Calculate the optimal number of iterations in Grover search.

    Parameters
        q_num : ``int``\n
            The number of qubits in the search space. Search space size:  :math:`N = 2 ^ {\\text {q_num}}`.
        sol_num : ``int``\n
            Number of target solution states.

    Returns
        num : The optimal number of iterations in Grover search.

    Examples
        An example for the case we show in the Grover search circuit. And we know there
        is only one solution to be found. And total 2 qubits for the search space.

    >>> from pyqpanda3.core import CPUQVM, QCircuit, Z, TOFFOLI
    >>> from pyqpanda_alg import Grover
    >>> m = CPUQVM()
    >>> q_state = list(range(3))

    >>> def mark(qubits):
    >>>     cir = QCircuit()
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     cir << Z(qubits[2])
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     return cir
    >>> demo_search = Grover.Grover(flip_operator=mark)
    >>> iter_num = Grover.iter_num(q_num=len(q_state), sol_num=2)
    >>> print('best iter num: ', iter_num)
    best iter num:  1

    """
    num = int(np.floor(np.pi * np.sqrt(2 ** q_num / sol_num) / 4))
    return num


def iter_analysis(q_num, sol_num, iternum=1):
    """
    Calculate the amplification probability and rotation angle for given amplitude
    amplification iteration number.

    Parameters
        q_num : ``int``\n
            The number of qubits in the search space. Search space size:  :math:`N = 2 ^ {\\text {q_num}}`.
        sol_num : ``int``\n
            Number of target solution states.
        iternum : ``int``\n
            Given number of iteration.

    Returns
        prob, theta : (``float``, ``float``)\n
            The amplification probability and rotation angle for given iteration.

    Examples
        An example for the case we show in the Grover search circuit. And we know there
        is only one solution to be found. And total 2 qubits for the search space.

    >>> from pyqpanda3.core import CPUQVM, QCircuit, Z, TOFFOLI
    >>> from pyqpanda_alg import Grover
    >>> m = CPUQVM()
    >>> q_state = list(range(3))

    >>> def mark(qubits):
    >>>     cir = QCircuit()
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     cir << Z(qubits[2])
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     return cir
    >>> demo_search = Grover.Grover(flip_operator=mark)
    >>> prob, angle = Grover.iter_analysis(q_num=len(q_state), sol_num=2, iternum=1)
    >>> print('prob for getting one of the solution with given iter num 1:', prob)
    >>> prob, angle = Grover.iter_analysis(q_num=len(q_state), sol_num=2, iternum=2)
    >>> print('prob for getting one of the solution with given iter num 2:', prob)
    prob for getting one of the solution with given iter num 1: 1.0
    prob for getting one of the solution with given iter num 2: 0.24999999999999956

    """
    sol_prob = sol_num / (2 ** q_num)
    theta = np.arcsin(np.sqrt(sol_prob)) * 2
    prob = np.sin(((2 * iternum + 1) / 2) * theta) ** 2
    return prob, theta



def amp_operator(q_input=None, q_flip=None, q_zero=None, in_operator=None, flip_operator=None, zero_flip=None):
    """
    Construct complete Grover amplitude amplification operator.
    Can be part of Grover/Quantum Count/QAE and other amplitude amplification related algorithm.

    Parameters
        q_input : ``QVec``\n
            Target qubit(s) for in_operator (initial preparation circuit).
            Using Hadamard gates to create the uniform superposition at the beginning most of time.
            Although in most simple cases it includes the full workspace qubits,
            auxiliary qubits can be excluded when dealing with some complex problems.
        q_flip : ``QVec``\n
            Target qubit(s) for flip_operator.
        q_zero : ``QVec``\n
            Target qubit(s) for zero_flip.
        in_operator : callable ``f(qubits)``\n
            Operator/Circuit of the initial search state for the algorithm, default Hadamards.
        flip_operator : callable ``f(qubits)``\n
            Operator/Circuit of marking the good states by phase-flip. Default doing a pauli-Z
            gate at the last qubit.
        zero_flip : callable ``f(qubits)``\n
            Operator/Circuit of reflects 0s by phase-flip. Default doing a zero-controled pauli-Z
            gate on qubits.

    Returns
        circuit : QCircuit\n
            Amplitude amplification operator.

    Examples
        An example for constucting a amplitude amplification operator used in the case we show
        in the Grover search circuit.

    >>> from pyqpanda3.core import CPUQVM, QCircuit, Z, TOFFOLI
    >>> from pyqpanda_alg import Grover
    >>> m = CPUQVM()
    >>> q_state = list(range(3))

    >>> def mark(qubits):
    >>>     cir = QCircuit()
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     cir << Z(qubits[2])
    >>>     cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    >>>     return cir
    >>> print(Grover.amp_operator(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], flip_operator=mark))

    .. parsed-literal::
                              ┌─┐ ┌─┐     ┌─┐ ┌─┐
        q_0:  |0>──■─ ─── ─■─ ┤H├ ┤X├ ─■─ ┤X├ ┤H├
                   │       │  ├─┤ ├─┤ ┌┴┐ ├─┤ ├─┤
        q_1:  |0>──■─ ─── ─■─ ┤H├ ┤X├ ┤Z├ ┤X├ ┤H├
                  ┌┴┐ ┌─┐ ┌┴┐ └─┘ └─┘ └─┘ └─┘ └─┘
        q_2:  |0>─┤X├ ┤Z├ ┤X├ ─── ─── ─── ─── ───
                  └─┘ └─┘ └─┘

    """
    if not hasattr(q_input, '__len__'):
        q_input = [q_input]

    if q_flip is None:
        q_flip = [q_input[-1]]
    if not hasattr(q_flip, '__len__'):
        q_flip = [q_flip]

    if q_zero is None:
        q_zero = q_input
    if not hasattr(q_zero, '__len__'):
        q_zero = [q_zero]

    in_operator_no_dagger = in_operator(q_input) if in_operator is not None else apply_QGate(q_input, H)
    in_operator = in_operator(q_input) if in_operator is not None else apply_QGate(q_input, H)

    flip_operator = flip_operator(q_flip) if flip_operator is not None else Z(q_flip[-1])

    if zero_flip is not None:
        zero_flip = zero_flip(q_zero)
    elif len(q_zero) == 1:
        zero_flip = QCircuit()
        zero_flip << X(q_zero[0]) << Z(q_zero[0]) << X(q_zero[0])
    else:
        zero_flip = QCircuit()
        zero_flip << apply_QGate(q_zero, X)
        zero_flip << apply_QGate([q_zero[-1]], Z).control(q_zero[:-1]) 
        zero_flip << apply_QGate(q_zero, X)

    circuit = QCircuit()
    circuit << flip_operator
    circuit << in_operator.dagger() << zero_flip << in_operator_no_dagger

    return circuit


def mark_data_reflection(qubits: list = None, mark_data=None):
    """
    Can be used to construct a phase flip operator for given target states.

    Parameters
        qubits : ``QVec``\n
            Target qubit(s) for flip_operator.
        mark_data : ``str``, ``list[str]``\n
            Marked target state(s).

    Returns
        flip_operator : ``QCircuit``\n
            A phase flip operator for given target states

    Examples
        An example for searching '101' and '001' using the flip operator given by this function.

    >>> from pyqpanda3.core import CPUQVM, QProg
    >>> from pyqpanda_alg import Grover
    >>> m = CPUQVM()

    >>> q_state = list(range(3))
    >>> def mark(qubits):
    >>>     return Grover.mark_data_reflection(qubits=qubits, mark_data=['101', '001'])
    >>> demo_search = Grover.Grover(flip_operator=mark)
    >>> prog = QProg()
    >>> prog << demo_search.cir(q_input=q_state)
    >>> m.run(prog,1000)
    >>> res = m.result().get_prob_dict(q_state)
    >>> print(res)
    {'000': 0.0, '001': 0.5000000000000002, '010': 0.0, '011': 0.0, '100': 0.0, '101': 0.5000000000000002, '110': 0.0, '111': 0.0}

    """
    if not hasattr(qubits, '__len__'):
        qubits = [qubits]
    flip_operator = QCircuit()

    if isinstance(mark_data, str):
        mark_data = [mark_data]
    n = len(qubits)

    for i in mark_data:
        for j in range(n):
            if i[-(j + 1)] == '0':
                flip_operator << X(qubits[j])
            else:
                flip_operator << BARRIER([qubits[j]])
        flip_operator << Z(qubits[-1]).control(qubits[:-1])

        for j in range(n):
            if i[-(j + 1)] == '0':
                flip_operator << X(qubits[j])
            else:
                flip_operator << BARRIER([qubits[j]])
    return flip_operator


class GroverAdaptiveSearch:
    """This class provides a framework for Grover Adaptive Search [2].

    Parameters
        init_value : ``float``\n
            The given initial value of the optimization function.
        n_index : ``int``\n
            The number of qubits in the search space. Search space size: N = 2 ** q_num.
        init_circuit : callable ``f(qubits)``\n
            Operator/Circuit of the initial search state for the algorithm, default Hadamards.
        oracle_circuit : callable ``f(qubits, value)``\n
            Operator/Circuit of marking the `better` states by phase-flip. Default doing a pauli-Z
            gate at the last qubit.

    References
        [2] A. Gilliam, S. Woerner, C. Gonciulea, Grover Adaptive Search for Constrained
        Polynomial Binary Optimization. https://arxiv.org/abs/1912.04088

    >>> from pyqpanda_alg import Grover
    >>> import numpy as np
    >>> from pyqpanda3.core import QCircuit, H, U1, SWAP

    >>> def qft(qbs):
    >>>     n = len(qbs)
    >>>     qbs = qbs[::-1]
    >>>     cir = QCircuit()
    >>>     for j in range(1, n):
    >>>         cir << H(qbs[j - 1])
    >>>         for i in range(j, n):
    >>>             cir << U1(qbs[j - 1], np.pi * 2 ** (j - 1 - i)).control(qbs[i])
    >>>     cir << H(qbs[-1])
    >>>     for x in range(int(len(qbs)/2)):
    >>>         cir << SWAP(qbs[x], qbs[-x-1])
    >>>     return cir

    >>> # flip if x0 * x1 + x0 - x1 - current_min < 0
    >>> def flip_oracle_function(q_index_value, current_min):
    >>>     q_index = q_index_value[:2]
    >>>     q_value = q_index_value[2:]
    >>>     n_value = len(q_value)
    >>>     factor = np.pi * 2 ** (1 - n_value)
    >>>     cal_cir = QCircuit()
    >>>     for i, q_i in enumerate(q_value):
    >>>         cal_cir << H(q_i)
    >>>         cal_cir << U1(q_i, factor * 2 ** i).control(q_index)
    >>>         cal_cir << U1(q_i, factor * 2 ** i).control(q_index[0])
    >>>         cal_cir << U1(q_i, -factor * 2 ** i).control(q_index[1])
    >>>         cal_cir << U1(q_i, factor * 2 ** i * (-current_min))
    >>>     cal_cir << qft(q_value).dagger()
    >>>     return cal_cir

    >>> demo_search = Grover.GroverAdaptiveSearch(init_value=0, n_index=2, oracle_circuit=flip_oracle_function)

    >>> def n_value_function(current_min):
    >>>     n_value = 2 if current_min == 0 else 3
    >>>     return n_value

    >>> def value_function(var_array):
    >>>     var_array = list(map(int, var_array))[::-1]
    >>>     value = var_array[0] * var_array[1] + var_array[0] - var_array[1]
    >>>     return value

    >>> res = demo_search.run(continue_times=3,
    >>>                       n_value_function=n_value_function,
    >>>                       value_function=value_function,
    >>>                       process_show=True)
    >>> print(res)
    ======searching 1 ,rotation = 1 ======
    minimum Key Again:  00
    minimum Value No Change:  0
    ======searching 2 ,rotation = 1 ======
    Current minimum Key:  10
    Current minimum Value:  -1
    ======searching 1 ,rotation = 1 ======
    minimum Key Again:  10
    minimum Value No Change:  -1
    ======searching 2 ,rotation = 1 ======
    ======searching 3 ,rotation = 1 ======
    rotations:  5
    ([[0, 1]], -1)

    """
    def __init__(self, init_value, n_index, init_circuit=None, oracle_circuit=None):
        self._current_min = init_value
        self.n_index = n_index
        self.n_value = None
        self.init_circuit = init_circuit
        self.oracle_circuit = oracle_circuit

    def _init_circuit(self, qlist):
        if self.init_circuit is None:
            return hadamard_circuit(qlist[:self.n_index])
        else:
            return self.init_circuit(qlist, self._current_min)

    def _oracle_circuit(self, qlist):
        if self.oracle_circuit is None:
            return Z(qlist[-1])
        else:
            return self.oracle_circuit(qlist, self._current_min)

    def run(self, continue_times: int = 3, n_value_function=None, value_function=None,
            rotation_change='random', process_show=False):
        """
        Run the Grover Adaptive Search algorithm to find the minimum.

        Parameters
            continue_times : ``int``\n
                The maximum number of repeated searches at the current optimal point.
            n_value_function : callable ``f(value)``\n
                Function for computing the number of qubits for marking the `better` states at current
                best value, variable qubits not included.
            value_function : callable ``f(var_array)``\n
                Function for computing the problem value of given varriables array(str given as qpanda state).
            rotation_change : ``str{'random', 'increase'}``, optional\n
                The method to get the number of Grover iterations for each search of a search cycle.

               - ``random`` : The number of Grover iterations for each search is randomly obtained from a
                increasing interval. (Default)
               - ``increase`` : The number of Grover iterations for each search is increasing.
            process_show : ``bool``\n
                Set to True to print the detail during search.

        Returns
            minimum_indexes, minimum_res : ( ``list[list[int]]``, ``float``)\n
                The optimization result including the solution array and the optimal value.

        Examples
            An example for minimization of quadratic binary function: x0 * x1 + x0 - x1.


        """
        binary = True
        num_all_solutions = 2 ** self.n_index
        machine = CPUQVM()

        minimum_found = False
        minimum_res = self._current_min
        minimum_indexes = []
        indexes_measured = []

        rotations = 0

        while not minimum_found:
            m = 1
            improvement_found = False
            loops_with_no_improvement = 0
            while not improvement_found:
                rotations += 1
                self.n_value = n_value_function(self._current_min)

                q_index_value = QProg(self.n_index + self.n_value).qubits()

                if rotation_change == 'random':
                    rotation_count = 1 if m < 2 else np.random.randint(low=1, high=m)
                elif rotation_change == 'increase':
                    rotation_count = int(m)
                else:
                    raise NameError("Method not recognized")

                if process_show:
                    print('======searching', loops_with_no_improvement + 1, ',rotation =', rotation_count, '======')
                Grover_instance = Grover(in_operator=self._init_circuit, flip_operator=self._oracle_circuit)

                zxy = Grover_instance.cir(q_input=q_index_value, q_flip=q_index_value, q_zero=q_index_value[:self.n_index],
                                          iternum=rotation_count)

                prog = QProg()
                prog << zxy << measure_all(q_index_value[:self.n_index], q_index_value[:self.n_index])
                machine.run(prog, shots=1)
                prog_res = machine.result().get_prob_dict()
                outcome = list(prog_res.keys())[0]

                current_value = value_function(outcome)
                v = current_value - self._current_min

                if v < 0:
                    indexes_measured.append(outcome)
                    minimum_res = current_value
                    minimum_indexes = [outcome]
                    if process_show:
                        print('Current minimum Key: ', outcome)
                        print('Current minimum Value: ', minimum_res)
                    improvement_found = True
                    self._current_min = minimum_res


                else:
                    loops_with_no_improvement += 1
                    if outcome not in indexes_measured:
                        indexes_measured.append(outcome)
                    if v == 0:
                        if outcome not in indexes_measured:
                            minimum_indexes.append(outcome)
                        if process_show:
                            print('minimum Key Again: ', outcome)
                            print('minimum Value No Change: ', minimum_res)
                    m = min(m * 1.34, 2 ** (self.n_index / 2)) if rotation_change == 'random' \
                        else min(m * 1.34, 2 ** (self.n_index / 2))
                    if loops_with_no_improvement >= continue_times or \
                            len(indexes_measured) == num_all_solutions:
                        improvement_found = True
                        minimum_found = True

        minimum_indexes = list(set(minimum_indexes))
        if binary:
            opt_xs = []
            for key in minimum_indexes:
                opt_x = np.array([1 if s == '1' else 0 for s in ('{0:%s}' % self.n_index).format(key)])
                opt_xs.append(opt_x.tolist()[::-1])
            minimum_indexes = opt_xs
        if process_show:
            print('rotations: ', rotations)
        return minimum_indexes, minimum_res

    @staticmethod
    def _bin_to_int(bin_value):    #def _bin_to_int(bin_value):
        n_value = len(bin_value)
        if bin_value.startswith("1"):
            int_v = int(bin_value, 2) - 2 ** n_value
        else:
            int_v = int(bin_value, 2)
        return int_v
    


