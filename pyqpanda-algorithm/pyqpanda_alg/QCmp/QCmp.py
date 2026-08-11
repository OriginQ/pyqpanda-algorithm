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

from pyqpanda3.core import QCircuit, TOFFOLI, CNOT, RY, U1, X
import numpy as np

from .. plugin import *

def int_comparator(value, q_state, q_anc_cmp, function='geq', reuse=False):
    """
    This function provides comparators to compare basis states(can be superposition states)
    against a given classical integer.

    Parameters
        value : ``int``
            The given classical integer.
        q_state : ``Qubit``, ``QVec``
            State qubits.
        q_anc_cmp : ``QVec``
            Ancilla and comparison result qubits. The comparison result qubit should be the last element.
            The qubit number of this register should be equal to q_state.
        function : ``str{'geq', 'g', 'seq', 's'}``, optional
            Evaluate conditions:
            - ``geq`` : evaluate a ``>=`` condition. (Default)
            - ``g``   : evaluate a ``>`` condition.
            - ``seq`` : evaluate a ``<=`` condition.
            - ``s``   : evaluate a ``<`` condition.

        reuse : ``bool``
            Set to True to add a reverse circuit part to reuse ancilla qubits.

    Returns
        circuit : ``QCircuit``
            The result this function return is a quangtum circuit.
            The comparison result qubit would be in state  :math:`|1\\rangle` when the quantum state
            satisfies the comparison condition, otherwise  :math:`|0\\rangle`. Therefore we can get a
            probability outcome of the comparison.

    Examples
        An example of implementing a two-qubit uniform superposition state compared with 2.

    >>> from pyqpanda_alg import QCmp
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> value = 2
    >>> m = CPUQVM()
    >>> prog = QProg()
    >>> prog << H(0) << H(1)
    >>> cir = QCmp.int_comparator(value, [0, 1], [2, 3], function='g', reuse=True)
    >>> prog << cir
    >>> m.run(prog, 1000)
    >>> prob_dict_result = m.result().get_prob_dict([3])
    >>> print(prob_dict_result)
    {'0': 0.7500000000000003, '1': 0.2500000000000001}

    """
    if not hasattr(q_state, '__len__'):
        q_state = [q_state]
    if not hasattr(q_anc_cmp, '__len__'):
        q_anc_cmp = [q_anc_cmp]


    q_cmp = q_anc_cmp[-1]
    circuit = QCircuit()

    if function == 's' or function == 'seq' or function == 'neq':
        circuit << X(q_cmp)
    elif function == 'g' or function == 'geq' or function == 'eq':
        pass
    else:
        raise NameError('method is not recognized')

    if value < 0:
        circuit << X(q_cmp)
        return circuit

    n_state = len(q_state)
    remainder = 2 ** n_state - 1 - value
    if remainder >= 0:
        value_twos = '{:b}'.format(value).rjust(n_state, '0')
        value_inverse = list(map(int, value_twos[::-1]))
    else:
        return circuit

    for i in range(n_state):
        if i == 0:
            if function == 'g' or function == 'seq':
                if value_inverse[0] == 0:
                    circuit << CNOT(q_state[0], q_anc_cmp[0])
            elif function == 'geq' or function == 's':
                if value_inverse[0] == 0:
                    circuit << X(q_anc_cmp[0])
                else:
                    circuit << CNOT(q_state[0], q_anc_cmp[0])
            else:
                if value_inverse[0] == 0:
                    circuit << _zero_control(q_state[0], X(q_anc_cmp[0]))
                else:
                    circuit << CNOT(q_state[0], q_anc_cmp[0])
        elif i == n_state - 1:
            if function == 'g' or function == 'seq' or function == 'geq' or function == 's':
                if value_inverse[i] == 0:
                    circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_cmp)
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
            else:
                if value_inverse[i] == 0:
                    circuit << _zero_control(q_state[i], CNOT(q_anc_cmp[i - 1], q_cmp))
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
        else:
            if function == 'g' or function == 'seq' or function == 'geq' or function == 's':
                if value_inverse[i] == 0:
                    circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
            else:
                if value_inverse[i] == 0:
                    circuit << _zero_control(q_state[i], CNOT(q_anc_cmp[i - 1], q_anc_cmp[i]))
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])


    if reuse and n_state > 1:
        for i in reversed(range(n_state - 1)):
            if i == 0:
                if function == 'g' or function == 'seq':
                    if value_inverse[0] == 0:
                        circuit << CNOT(q_state[0], q_anc_cmp[0])
                elif function == 'geq' or function == 's':
                    if value_inverse[0] == 0:
                        circuit << X(q_anc_cmp[0])
                    else:
                        circuit << CNOT(q_state[0], q_anc_cmp[0])
                else:
                    if value_inverse[0] == 0:
                        circuit << _zero_control(q_state[0], X(q_anc_cmp[0]))
                    else:
                        circuit << CNOT(q_state[0], q_anc_cmp[0])
            elif i == n_state - 1:
                if function == 'g' or function == 'seq' or function == 'geq' or function == 's':
                    if value_inverse[i] == 0:
                        circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_cmp)
                    else:
                        circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
                else:
                    if value_inverse[i] == 0:
                        circuit << _zero_control(q_state[i], CNOT(q_anc_cmp[i - 1], q_cmp))
                    else:
                        circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
            else:
                if function == 'g' or function == 'seq' or function == 'geq' or function == 's':
                    if value_inverse[i] == 0:
                        circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
                    else:
                        circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
                else:
                    if value_inverse[i] == 0:
                        circuit << _zero_control(q_state[i], CNOT(q_anc_cmp[i - 1], q_anc_cmp[i]))
                    else:
                        circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
    return circuit


def interpolation_comparator(value, q_state, q_anc_cmp, function='g', reuse=False):
    """
    This function provides comparators to compare basis states(generate from a smooth distribution)
    against a given classical number(can be float).
    This function introduces an interpolation method to make qubits "look" like a float number when compared,
    so that comparisons can be made in the real number domain. In detail, when sampling, we regard the
    quantum state ``a`` as the interval from ``inf:=a-0.5`` to ``sup:=a+1.5``. When comparing the quantum
    state with ``inf+delta``, the probability of smaller is delta.

    Parameters
        value : ``float``
            The given classical number.
        q_state : ``Qubit``, ``QVec``
            State qubits.
        q_anc_cmp : ``QVec``
            Ancilla and comparison result qubits. The comparison result qubit should be the last element.
            The qubit number of this register should be equal to q_state.
        function : ``str{'g', 's'}``, optional
            Evaluate conditions:
            - ``g`` : evaluate a ``>`` condition.(Default)
            - ``s`` : evaluate a ``<`` condition.

        reuse : ``bool``
            Set to True to add a reverse circuit part to reuse ancilla qubits.

    Returns
        circuit : ``QCircuit``
            The result this function return is a quangtum circuit.
            The comparison result qubit would be in state :math:`|1\\rangle` when the quantum state
            satisfies the comparison condition, otherwise :math:`|0\\rangle`. Therefore we can get a
            probability outcome of the comparison.

    Examples
        An example of implementing qubit state '110' compared with 3.3.

    >>> from pyqpanda_alg import QCmp
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> value = 3.3
    >>> m = CPUQVM()
    >>> prog = QProg()
    >>> prog << X(0) << X(1) << I(2)
    >>> cir = QCmp.interpolation_comparator(value, [0, 1, 2], [3, 4, 5], function='g', reuse=True)
    >>> prog << cir
    >>> m.run(prog, 1000)
    >>> prob_dict_result = m.result().get_prob_dict([5])
    >>> print(prob_dict_result)
    {'0': 0.7999999999999997, '1': 0.20000000000000023}

    """
    if not hasattr(q_state, '__len__'):
        q_state = [q_state]
    if not hasattr(q_anc_cmp, '__len__'):
        q_anc_cmp = [q_anc_cmp]

    q_cmp = q_anc_cmp[-1]
    circuit = QCircuit()

    if function == 's':
        pass

    elif function == 'g':
        circuit << X(q_cmp)

    else:
        raise NameError('method is not recognized')

    if value < 0:
        raise ValueError('Negative input')

    value_int = int(round(value + 1e-10))
    value_res = value - value_int + 0.5
    if value_res < 0:
        value_res = 0
    angle = np.arcsin(value_res ** 0.5) * 2

    n_state = len(q_state)
    remainder = 2 ** n_state - 1 - value_int
    if remainder >= 0:
        value_twos = '{:b}'.format(value_int).rjust(n_state, '0')
        value_inverse = list(map(int, value_twos[::-1]))
    else:
        return circuit

    for i in range(n_state):
        if i == 0:
            if value_inverse[0] == 0:
                circuit << RY(q_anc_cmp[0], angle)
            else:
                circuit << RY(q_anc_cmp[0], angle).control(q_state[0])

        elif i != n_state-1:
            if value_inverse[i] == 0:
                circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
            else:
                circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
        else:
            if value_inverse[i] == 0:
                circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_cmp)
            else:
                circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
    if reuse and n_state > 1:
        for i in reversed(range(n_state - 1)):
            if i == 0:
                pass
            elif i != n_state-1:
                if value_inverse[i] == 0:
                    circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i])
            else:
                if value_inverse[i] == 0:
                    circuit << _qor(q_state[i], q_anc_cmp[i - 1], q_cmp)
                else:
                    circuit << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_cmp)
    return circuit


def qubit_comparator(q_state_1, q_state_2, q_anc_cmp, function='geq'):
    """
    This function provides comparators to compare between two basis states(can be superposition states).

    Parameters
        q_state_1 : ``Qubit``, ``QVec``
            The first state qubits.
        q_state_2 : ``Qubit``, ``QVec``
            The second state qubits.
        q_anc_cmp : ``QVec``
            Ancilla and comparison result qubits. The comparison result qubit should be the last element.
            The qubit number of this register should be equal to q_state.
        function : ``str{'geq', 'g', 'seq', 's', 'eq', 'neq'}``, optional
            Evaluate conditions:
            - ``geq`` : evaluate a ``>=`` condition. (Default)
            - ``g``   : evaluate a ``>`` condition.
            - ``seq`` : evaluate a ``<=`` condition.
            - ``s``   : evaluate a ``<`` condition.
            - ``eq``  : evaluate a ``==`` condition.
            - ``neq`` : evaluate a ``!=`` condition.
        reuse : bool
            Set to True to add a reverse circuit part to reuse ancilla qubits.

    Returns
        circuit : ``QCircuit``
            The result this function return is a quangtum circuit.
            The comparison result qubit would be in state :math:`|1\\rangle` when the quantum state
            satisfies the comparison condition, otherwise :math:`|0\\rangle`. Therefore we can get a
            probability outcome of the comparison.

    Examples
        An example of implementing a two-qubit uniform superposition state compared with qubit state '01'.

    >>> from pyqpanda_alg import QCmp
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> m = CPUQVM()
    >>> prog = QProg()
    >>> prog << H(0) << H(1) << X(3)
    >>> cir = QCmp.qubit_comparator([0, 1], [2, 3],  [4, 5], function='g')
    >>> prog << cir
    >>> m.run(prog, 1000)
    >>> prob_dict_result = m.result().get_prob_dict([5])
    >>> print(prob_dict_result)
    {'0': 0.7500000000000003, '1': 0.2500000000000001}

    """

    if not hasattr(q_state_1, '__len__'):
        q_state_1 = [q_state_1]
    if not hasattr(q_state_2, '__len__'):
        q_state_2 = [q_state_2]
    if not hasattr(q_anc_cmp, '__len__'):
        q_anc_cmp = [q_anc_cmp]

    q_cmp = q_anc_cmp[-1]
    circuit = QCircuit()

    if function == 's' or function == 'seq' or function == 'neq':
        circuit << X(q_cmp)
    elif function == 'g' or function == 'geq' or function == 'eq':
        pass
    else:
        raise NameError('Distribution is not recognized')

    if len(q_state_1) > len(q_state_2):
        q_state = q_state_2
        q_ctr = q_state_1
        circuit << X(q_cmp)
        if function == 's' or function == 'geq':
            flag = True
        else:
            flag = False

    else:
        q_state = q_state_1
        q_ctr = q_state_2
        if function == 'g' or function == 'seq':
            flag = True
        else:
            flag = False

    n_state = len(q_state)

    for i in range(n_state):
        if i == 0:
            if function == 'eq' or function == 'neq':
                circuit << CNOT(q_state[0], q_anc_cmp[0]).control(q_ctr[0])
                circuit << _zero_control([q_state[0], q_ctr[0]], X(q_anc_cmp[0]))
            else:
                if flag:
                    circuit << _zero_control(q_ctr[0], CNOT(q_state[0], q_anc_cmp[0]))
                else:
                    circuit << X(q_anc_cmp[0])
                    circuit << _zero_control(q_state[0], CNOT(q_ctr[0], q_anc_cmp[0]))

        elif i < n_state - 1:
            if function == 'eq' or function == 'neq':
                circuit << CNOT(q_anc_cmp[i - 1], q_anc_cmp[i]).control([q_state[i], q_ctr[i]])
                circuit << _zero_control([q_state[i], q_ctr[i]], CNOT(q_anc_cmp[i - 1], q_anc_cmp[i]))
            else:
                circuit << CNOT(q_anc_cmp[i - 1], q_anc_cmp[i]).control([q_state[i], q_ctr[i]])
                circuit << _zero_control(q_ctr[i], _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i]))

        else:
            if function == 'eq' or function == 'neq':
                circuit << apply_QGate(q_ctr[i:], X) \
                << _zero_control(q_state[i], CNOT(q_anc_cmp[i - 1], q_anc_cmp[i])).control(q_ctr[i:]) \
                << X(q_ctr[i]) \
                << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i]).control(q_ctr[i:]) \
                << X(q_ctr[i]) \
                << apply_QGate(q_ctr[i:], X)
            else:
                circuit << apply_QGate(q_ctr[i:], X) \
                << _qor(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i]).control(q_ctr[i:]) \
                << X(q_ctr[i]) \
                << TOFFOLI(q_state[i], q_anc_cmp[i - 1], q_anc_cmp[i]).control(q_ctr[i:]) \
                << X(q_ctr[i]) \
                << apply_QGate(q_ctr[i:], X)
    return circuit


def _qor(qbit1, qbit2, q_res):
    circuit = QCircuit()
    circuit << apply_QGate([qbit1, qbit2, q_res], X) \
            << TOFFOLI(qbit1, qbit2, q_res) \
            << apply_QGate([qbit1, qbit2], X)
    return circuit


def _zero_control(q_ctr, circuit_input):
    circuit = QCircuit()
    if not isinstance(q_ctr, list):
        q_ctr = [q_ctr]

    circuit << apply_QGate(q_ctr, X) \
            << circuit_input.control(q_ctr)\
            << apply_QGate(q_ctr, X)
    return circuit


def qft_comparator(value, q_state, q_cmp, function='geq'):
    """
    This function provides qft_based comparators to compare basis states(can be superposition states)
    against a given classical integer.

    Parameters
        value : ``int``
            The given classical integer in range [0,N).
        q_state : ``Qubit``, ``QVec``
            State qubits.
        q_cmp : ``QVec``
            The comparison result qubit.
        function : ``str{'geq', 'g', 'seq', 's'}``, optional
            Evaluate conditions:
            - ``geq`` : evaluate a ``>=`` condition. (Default)
            - ``g``   : evaluate a ``>`` condition.
            - ``seq`` : evaluate a ``<=`` condition.
            - ``s``   : evaluate a ``<`` condition.

    Returns
        circuit : ``QCircuit``
            The result this function return is a quangtum circuit.
            The comparison result qubit would be in state  :math:`|1\\rangle` when the quantum state
            satisfies the comparison condition, otherwise  :math:`|0\\rangle`. Therefore we can get a
            probability outcome of the comparison.

    Examples
        An example of implementing a two-qubit uniform superposition state compared with 2.

    >>> from pyqpanda_alg import QCmp
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> value = 2
    >>> m = CPUQVM()
    >>> prog = QProg()
    >>> prog << H(0) << H(1)
    >>> cir = QCmp.qft_comparator(value, [0, 1], [2], function='g')
    >>> prog << cir
    >>> m.run(prog, 1000)
    >>> prob_dict_result = m.result().get_prob_dict([2])
    >>> print(prob_dict_result)
    {'0': 0.750000000000001, '1': 0.2500000000000002}

    """
    
    cir = QCircuit()

    if function == 'seq' or function == 'g':
        value += 1

    if not hasattr(q_cmp, '__len__'):
        q_cmp = [q_cmp]

    qlist = q_state + q_cmp
    n_list = len(qlist)
    factor_all = np.pi * 2 ** (1 - n_list)
    factor_remain = np.pi * 2 ** (1 - len(q_state))

    cir << QFT(qlist)
    for i, q_i in enumerate(qlist):
        cir << U1(q_i, - factor_all * 2 ** i * value)
    cir << QFT(qlist).dagger()

    cir << QFT(q_state)
    for i, q_i in enumerate(q_state):
        cir << U1(q_i, factor_remain * 2 ** i * value)
    cir << QFT(q_state).dagger()

    if function == 's' or function == 'seq':
        pass
    elif function == 'g' or function == 'geq':
        for q in q_cmp:
            cir << X(q)
        pass
    else:
        raise NameError('method is not recognized')
    return cir


def qft_qubit_comparator(q_state_1, q_state_2, q_cmp, function='geq'):
    """
    This function provides qft_based comparators to compare between two basis states(can be superposition states).

    Parameters
        q_state_1 : ``Qubit``, ``QVec``
            The first state qubits.
        q_state_2 : ``Qubit``, ``QVec``
            The second state qubits.
        q_cmp : ``QVec``
            Comparison result qubit.
        function : ``str{'geq', 'g', 'seq', 's'}``, optional
            Evaluate conditions:
            - ``geq`` : evaluate a ``>=`` condition. (Default)
            - ``g``   : evaluate a ``>`` condition.
            - ``seq`` : evaluate a ``<=`` condition.
            - ``s``   : evaluate a ``<`` condition.

    Returns
        circuit : ``QCircuit``
            The result this function return is a quangtum circuit.
            The comparison result qubit would be in state :math:`|1\\rangle` when the quantum state
            satisfies the comparison condition, otherwise :math:`|0\\rangle`. Therefore we can get a
            probability outcome of the comparison.

    Examples
        An example of implementing a two-qubit uniform superposition state compared with qubit state '01'.

    >>> from pyqpanda_alg import QCmp
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> m = CPUQVM()
    >>> prog = QProg()
    >>> prog << H(0) << H(1) << X(3)
    >>> cir = QCmp.qft_qubit_comparator([0, 1], [2, 3], [4], function='g')
    >>> prog << cir
    >>> m.run(prog, 1000)
    >>> prob_dict_result = m.result().get_prob_dict([4])
    >>> print(prob_dict_result)
    {'0': 0.7500000000000007, '1': 0.2500000000000002}
    
    """

    cir = QCircuit()


    if not hasattr(q_cmp, '__len__'):
        q_cmp = [q_cmp]

    qlist = q_state_2 + q_cmp
    n_list = len(qlist)
    factor_all = np.pi * 2 ** (1 - n_list)
    factor_remain = np.pi * 2 ** (1 - len(q_state_2))
    q_cmp = q_cmp[0]
    cir << QFT(qlist)
    bit_weights = []
    for j in range(len(q_state_1)):
        bit_weights += [2 ** j]

    for i, wi in enumerate(bit_weights):
        for j, qj in enumerate(qlist):
            cir << U1(qj, -factor_all * 2 ** j * wi).control(q_state_1[i])

    cir << QFT(qlist).dagger()

    cir << QFT(q_state_2)
    for i, wi in enumerate(bit_weights):
        for j, qj in enumerate(qlist):
            cir << U1(qj, factor_remain * 2 ** j * wi).control(q_state_1[i])


    cir << QFT(q_state_2).dagger()

    if function == 's' or function == 'seq':
        cir << X(q_cmp)
        pass
    elif function == 'g' or function == 'geq':
        pass
    else:
        raise NameError('method is not recognized')
    return cir

