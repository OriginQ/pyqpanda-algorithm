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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, X, H, Z, RX
import numpy as np
from numpy import pi

from pyqpanda_alg.Grover import Grover,amp_operator
from typing import Union, List

from .. plugin import *

class QAE:
    """
    This class provides a framework for original Quantum Amplitude Estimation(QAE) algorithm [1].

    Parameters
        operator_in : callable ``f(qubits)``
            Operator/Circuit of the estimated qubits state.
        qnumber : ``int``
            The number of all qubits used in circuit.
        res_index : ``int``, ``list``
            The index of the estimated qubit(s).
        epsilon : ``float``
            Estimated precision, i.e. the minimum error.
        target_state : ``str``
            Estimated target state.


    References
        [1] Brassard G, Hoyer P, Mosca M, et al. Quantum amplitude amplification and estimation[J].
        Contemporary Mathematics, 2002, 305: 53-74.

    >>> from pyqpanda_alg import QAE
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> def create_cir(qlist):
    >>>     cir = QCircuit()
    >>>     cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    >>>     return cir
    >>> W = QAE.QAE(operator_in=create_cir, qnumber=2, epsilon=0.01, res_index=[0, 1], target_state='11').run()
    >>> print(W)
    0.24294862790338914

    """
    def __init__(self, operator_in=None,
                 qnumber: int = 0,
                 res_index: Union[int, List[int]] = -1,
                 epsilon: float = 1e-3,
                 target_state: str = '1'
                 ):

        machine_type = 'CPU'
        self.operator = operator_in
        self.epsilon = epsilon
        self.qnumber = qnumber
        self.machine_type = machine_type
        if machine_type == 'CPU':
            self.machine = CPUQVM()
        self.n_anc = -int(np.floor(np.log2(epsilon)))
        self.max_anc_qubits = 14
        self.res_index = res_index
        self.target_state = target_state
        if type(self.res_index) != list:
            self.res_index = [self.res_index]
        if self.max_anc_qubits < self.n_anc:
            raise ValueError('Require lower epsilon')
        if len(self.res_index) != len(self.target_state):
            raise ValueError('the length of res_index and target_state not equal')
        if type(self.res_index) != list:
            self.res_index = [self.res_index]

    def __del__(self):
        pass

    def _Q_cir(self, q_operator):
        if type(self.res_index) != list:
            self.res_index = [self.res_index]
        q_target = []
        q_else = []
        q1 = []
        index = []
        for i in self.res_index:
            index += [(list(range(self.qnumber)))[i]]
        for i in range(self.qnumber):
            if i in index:
                q_target += [q_operator[i]]
            else:
                q_else += [q_operator[i]]
            if i != index[-1]:
                q1 += [q_operator[i]]

        Qcir = QCircuit()
        if len(q_target) == 1:
            Qcir << Z(q_target[0]) << RX(q_target[0], 2 * pi)
        elif len(q_target) > 1:
            Qcir << Z(q_target[-1]).control(q_target[:-1]) << RX(q_target[-1], 2 * pi)

        for k in range(len(self.target_state)):
            if self.target_state[k] == '0':
                Qcir << X(q_target[-k-1])
        Qcir << self.operator(q_operator).dagger()

        for q1idx in q1:
            Qcir << X(q1idx)
        Qcir << Z(q_target[-1]).control(q1) \
             << RX(q_target[-1], 2 * pi).control(q1) 
        for q1idx in q1:
            Qcir << X(q1idx)

        Qcir << self.operator(q_operator)
        for k in range(len(self.target_state)):
            if self.target_state[k] == '0':
                Qcir << X(q_target[-k-1])
        return Qcir

    def run(self):
        """
        Run the quantum amplitude estimation algorithm.

        Returns
            prob : ``float``
                A probability value as the amplitude estimation result.

        Examples
            An example for implementing an amplitude estimation for target state '11' of the following circuit.
        
        .. parsed-literal::
                      ┌────────────┐
            q_0:  \|0>─┤RY(1.047198)├ ─■─
                      └────────────┘ ┌┴┐
            q_1:  \|0>─────────────── ┤X├
                                     └─┘
        >>> from pyqpanda_alg import QAE
        >>> import numpy as np
        >>> from pyqpanda3.core import *
        >>> def create_cir(qlist):
        >>>     cir = QCircuit()
        >>>     cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
        >>>     return cir
        >>> W = QAE.QAE(operator_in=create_cir, qnumber=2, epsilon=0.01, res_index=[0, 1], target_state='11').run()
        >>> print(W)
        0.24294862790338914

        """
        q_operator = QProg(self.qnumber + self.n_anc).qubits()

        q_target = []
        index = []
        for i in self.res_index:
            index += [(list(range(self.qnumber)))[i]]
        for i in range(self.qnumber):
            if i in index:
                q_target += [q_operator[i]]

        prog = QProg()

        prog << self.operator(q_operator[:self.qnumber])
        for k in range(len(self.target_state)):
            if self.target_state[k] == '0':
                prog << X(q_target[-k-1])

        for qidx in q_operator[self.qnumber:]:
            prog << H(qidx)
        for i in range(self.n_anc):
            for j in range(2 ** i):
                prog << self._Q_cir(q_operator[:self.qnumber]).control([q_operator[self.qnumber:][i]])

        prog << QFT(q_operator[self.qnumber:]).dagger()
        self.machine.run(prog, 1000)
        result = self.machine.result().get_prob_dict(q_operator[self.qnumber:])
        res_state = max(result, key=result.get)
        amplitude = np.sin(int(res_state, 2) * pi / 2 ** self.n_anc)
        prob = amplitude ** 2
        return prob


class IQAE:
    """
    This class provides a framework for Iterative Quantum Amplitude Estimation(IQAE) algorithm [2].
    Estimated target state is  :math:`|1\\rangle`.

    Parameters
        operator_in : callable ``f(qubits)``
            Operator/Circuit of the estimated qubits state.\n
        qnumber : ``int``
            The number of all qubits used in circuit.\n
        res_index : ``int``
            The index of the estimated qubit.\n
        epsilon : ``float``
            Estimated precision, i.e. the minimum error.\n


    References
        [2] Grinko, D., Gacon, J., Zoufal, C. et al. Iterative quantum amplitude estimation.
        npj Quantum Inf 7, 52 (2021). https://doi.org/10.1038/s41534-021-00379-1

    >>> from pyqpanda_alg import QAE
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> def create_cir(qlist):
    >>>     cir = QCircuit()
    >>>     cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
    >>>     return cir
    >>> W = QAE.IQAE(operator_in=create_cir, qnumber=2, epsilon=0.01, res_index=-1).run()
    >>> print(W)
    0.25735228001322236

    """
    def __init__(self, operator_in=None,
                 qnumber: int = 0,
                 res_index: int = -1,
                 epsilon: float = 1e-3
                 ):
        alpha = 0.05  
        method = 'cher'  
        ratio = 2.0  
        machine_type = 'CPU'
        self.epsilon = epsilon
        self.alpha = alpha
        self.method = method
        self.ratio = ratio
        self.qnumber = qnumber
        self.res_index = res_index
        self.machine_type = machine_type
        self.n_sum = 0

        if machine_type == 'CPU':
            self.machine = CPUQVM()
        else:
            raise NameError('Support \'CPU\', \'QCloud\' only')
        self.qlist = QProg(self.qnumber).qubits()
        self.clist = QProg(self.qnumber).cbits()
        self.operatorA = operator_in
        self.a_l = 0
        self.a_u = 0
        self.N_max = 32 / (1 - 2 * np.sin(np.pi / 14)) ** 2 * np.log(
            2 / self.alpha * np.log(np.pi / 4 / self.epsilon) / np.log(self.ratio))
        self.draw = False

    def __del__(self):
        pass

    def run(self):
        """
        Run the iterative quantum amplitude estimation algorithm.

        Returns
            prob : ``float``
                A probability value as the iterative amplitude estimation result.

        Examples
            An example for implementing an iterative amplitude estimation for qubit q_1 of the following circuit.
        
        .. parsed-literal::
                      ┌────────────┐
            q_0:  \|0>─┤RY(1.047198)├ ─■─
                      └────────────┘ ┌┴┐
            q_1:  \|0>─────────────── ┤X├
                                     └─┘
        >>> from pyqpanda_alg import QAE
        >>> import numpy as np
        >>> from pyqpanda3.core import *
        >>> def create_cir(qlist):
        >>>     cir = QCircuit()
        >>>     cir << RY(qlist[0], np.pi / 3) << X(qlist[1]).control(qlist[0])
        >>>     return cir
        >>> W = QAE.IQAE(operator_in=create_cir, qnumber=2, epsilon=0.01, res_index=-1).run()
        >>> print(W)
        0.25735228001322236

        """
        i_count = 0
        k = [0]
        up = True
        theta_l = 0
        theta_u = 1. / 4

        rounds_m = int(np.log(np.pi / 8 / self.epsilon) / np.log(self.ratio)) + 1

        l_max = 0
        if self.method == 'cher':
            l_max = np.arcsin((2 / self.N_max * np.log(2 * rounds_m / self.alpha)) ** (1. / 4))
        elif self.method == 'clop':
            print('未完成')

        m_add = 0
        n_add = 0
        k_count = 0
        scaller = 1.5
        while (theta_u - theta_l > self.epsilon / np.pi) and i_count < 20:
            i_count += 1
            epsilon = (theta_u - theta_l) * np.pi / 2
            n_shots = int(scaller * rounds_m * np.log(
                (np.log(np.pi / (4 * np.minimum(epsilon, np.pi / 8))) / np.log(self.ratio)) * (
                        2 / self.alpha)))

            n_shots = np.minimum(n_shots, int(self.N_max))

            k_next, up = self._findnextk(k[i_count - 1], theta_l, theta_u, up)
            k.append(k_next)
            bigk = 4 * k_next + 2
            if bigk > int(l_max / self.epsilon) + 1:
                n_round = int(n_shots * l_max / self.epsilon / bigk / 10) + 1
            else:
                n_round = int(n_shots)

            self.n_sum += n_round
            m = self._measure(k_next, n_round)
            self.draw = False

            if k[i_count] == k[i_count - 1]:
                m_add += m
                n_add += n_round
                k_count += 1
                res = m_add / n_add

            else:
                m_add = 0
                n_add = 0
                res = m / n_round

            ai_min, ai_max = self._chernoff_confint(res, int(np.minimum(self.n_sum, self.N_max)), rounds_m)

            if up:
                thetai_min = np.arccos(1 - 2 * ai_min) / 2 / np.pi
                thetai_max = np.arccos(1 - 2 * ai_max) / 2 / np.pi
            else:
                thetai_min = 1 - np.arccos(1 - 2 * ai_max) / 2 / np.pi
                thetai_max = 1 - np.arccos(1 - 2 * ai_min) / 2 / np.pi

            theta_u = (int(bigk * theta_u) + thetai_max) / bigk
            theta_l = (int(bigk * theta_l) + thetai_min) / bigk

        a_l = np.sin(2 * np.pi * theta_l) ** 2
        a_u = np.sin(2 * np.pi * theta_u) ** 2
        self.a_l, self.a_u = a_l, a_u
        return (a_l + a_u) / 2

    def _measure(self, k: int, n_round: int) -> int:
        machine = self.machine
        qlist = self.qlist
        clist = self.clist

        operator_g = amp_operator(in_operator=self.operatorA, q_input=qlist)
        prog = QProg()
        prog << self.operatorA(qlist)
        for i in range(k):
            prog << operator_g
        if self.draw:
            print(prog)
        prog << measure_all([qlist[self.res_index]], [qlist[self.qnumber - 1]]) 
        if self.machine_type == 'CPU':
            machine.run(prog, n_round)
            res = machine.result().get_counts()
            m = res.get('1')
        elif self.machine_type == 'QCloud':
            res = machine.full_amplitude_measure(prog, n_round)
            m = int(res['1'] * n_round)
        if m is None:
            m = 0

        return m

    def _findnextk(self, k_i: int, theta_l: float, theta_u: float, up: bool) -> (int, bool):
        ratio = self.ratio
        if ratio <= 1:
            raise ValueError('ratio must be larger than 1')

        bigk_i = 4 * k_i + 2

        k_max = int(1 / (2 * (theta_u - theta_l)))
        k = k_max - (k_max - 2) % 4

        while k >= ratio * bigk_i:
            theta_min = k * theta_l
            theta_max = k * theta_u
            if (theta_min - int(theta_min)) <= (theta_max - int(theta_max)) <= 0.5:
                bigk_next = k
                up = True
                k_next = int((bigk_next - 2) / 4)
                return k_next, up

            elif (theta_max - int(theta_max)) >= (theta_min - int(theta_min)) >= 0.5:
                bigk_next = k
                up = False
                k_next = int((bigk_next - 2) / 4)
                return k_next, up

            k -= 4

        return k_i, up

    def _chernoff_confint(self, value: float, shots: int, max_rounds: int):
        eps = np.sqrt(np.log(2 * max_rounds / self.alpha) / (2 * shots))
        lower = value - eps
        if value - eps < 0:
            lower = 0

        upper = value + eps
        if value + eps > 1:
            upper = 1

        return lower, upper
