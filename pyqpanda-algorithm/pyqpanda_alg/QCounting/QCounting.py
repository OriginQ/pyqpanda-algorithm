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

from pyqpanda3.core import CPUQVM, QCircuit, QProg, H, RX
import numpy as np
from numpy import pi

from pyqpanda_alg.Grover import amp_operator, mark_data_reflection
from typing import Union, List

from .. plugin import *


class QCounting:
    """
    This class provides a framework for the Brassard-Hoyer-Tapp quantum counting algorithm [1].

    Quantum counting estimates the number of marked items M in a search space of
    :math:`N = 2 ^ {\\text {qnumber}}` items. Phase estimation is applied to the Grover
    amplitude amplification operator given by ``Grover.amp_operator``, whose eigenphases
    :math:`\\theta` satisfy :math:`\\sin^2(\\theta / 2) = M / N`. The operator has the two
    eigenphases :math:`\\theta` and :math:`2\\pi - \\theta`; both readout peaks give the
    same count, so the peak does not have to be disambiguated.

    Parameters
        qnumber : ``int``\n
            The number of qubits in the search space. Search space size:
            :math:`N = 2 ^ {\\text {qnumber}}`.
        flip_operator : callable ``f(qubits)``\n
            Operator/Circuit of marking the good states by phase-flip. The qubits passed
            in are the search qubits followed by the workspace qubits.
        mark_data : ``str``, ``list[str]``\n
            Marked target state(s). Only used when flip_operator is None. An empty list
            marks nothing, which is a valid input with count 0.
        counting_qubits : ``int``\n
            The number of qubits in the counting register. Default qnumber + 2. The
            estimate error is bounded by :math:`\\pi N / 2 ^ {\\text {counting_qubits} + 1}`,
            which is :math:`\\pi / 8` at the default, so rounding the result returns M.
            A larger counting register sharpens the unrounded estimate.
        ancilla_qubits : ``int``\n
            The number of workspace qubits used by flip_operator. Default 0. The
            flip_operator has to return the workspace to :math:`|0\\rangle`.

    References
        [1] Brassard G, Hoyer P, Tapp A. Quantum counting[C]. Automata, Languages and
        Programming, ICALP 1998, Lecture Notes in Computer Science, 1443: 820-831.
        https://arxiv.org/abs/quant-ph/9805082

    >>> from pyqpanda_alg.QCounting import QCounting
    >>> counter = QCounting(qnumber=3, mark_data=['001', '011', '111'])
    >>> print(round(counter.run(), 4))
    3.2196

    """
    def __init__(self, qnumber: int = 0,
                 flip_operator=None,
                 mark_data: Union[str, List[str]] = None,
                 counting_qubits: int = None,
                 ancilla_qubits: int = 0
                 ):

        machine_type = 'CPU'
        if not isinstance(qnumber, int) or qnumber < 1:
            raise ValueError('qnumber must be a positive integer')
        if not isinstance(ancilla_qubits, int) or ancilla_qubits < 0:
            raise ValueError('ancilla_qubits must be a non-negative integer')

        if isinstance(mark_data, str):
            mark_data = [mark_data]

        if flip_operator is None:
            if mark_data is None:
                raise ValueError('either flip_operator or mark_data is required')
            for state in mark_data:
                if not isinstance(state, str) or len(state) != qnumber or set(state) - {'0', '1'}:
                    raise ValueError('mark_data entries must be bit strings of length qnumber')

            def s_f(qubits):
                return mark_data_reflection(qubits, mark_data)
            self.flip_operator = s_f
        else:
            self.flip_operator = flip_operator

        if counting_qubits is None:
            counting_qubits = qnumber + 2
        if not isinstance(counting_qubits, int) or counting_qubits < 1:
            raise ValueError('counting_qubits must be a positive integer')
        self.max_counting_qubits = 14
        if counting_qubits > self.max_counting_qubits:
            raise ValueError('Require fewer counting_qubits')

        self.qnumber = qnumber
        self.mark_data = mark_data
        self.counting_qubits = counting_qubits
        self.ancilla_qubits = ancilla_qubits
        self.machine_type = machine_type
        if machine_type == 'CPU':
            self.machine = CPUQVM()
        else:
            raise NameError('Support \'CPU\' only')
        self.theta = None

    def __del__(self):
        pass

    def _G_cir(self, q_search, q_ancilla):
        circuit = QCircuit()
        circuit << amp_operator(q_input=q_search,
                                q_flip=q_search + q_ancilla,
                                q_zero=q_search,
                                flip_operator=self.flip_operator)
        # amp_operator builds the two reflections without the overall minus sign of the
        # Grover operator. RX(q, 2 * pi) is -1 on the register and restores it, which the
        # eigenphases depend on once the operator is controlled.
        circuit << RX(q_search[0], 2 * pi)
        return circuit

    def cir(self, q_search=None, q_count=None, q_ancilla=None):
        """
        Get the full circuit of quantum counting.

        Parameters
            q_search : ``QVec``\n
                Target qubit(s) for the search space, qnumber qubits.
            q_count : ``QVec``\n
                Target qubit(s) for the counting register, counting_qubits qubits.
            q_ancilla : ``QVec``\n
                Workspace qubit(s) for flip_operator, ancilla_qubits qubits.

        Returns
            circuit : ``QCircuit``\n
                Hadamards on the search and counting registers, the controlled powers
                :math:`G ^ {2 ^ k}` of the Grover operator, and the inverse QFT on the
                counting register.

        Examples
            An example for building the counting circuit of a 2 qubit search space with
            the single marked state '11', read out by 3 counting qubits. The readout
            register peaks at the two counting values y and 2 ** 3 - y, which both give
            the same estimate.

        >>> import numpy as np
        >>> from pyqpanda3.core import CPUQVM, QProg
        >>> from pyqpanda_alg.QCounting import QCounting
        >>> m = CPUQVM()
        >>> q_state = list(range(5))
        >>> counter = QCounting(qnumber=2, mark_data='11', counting_qubits=3)
        >>> prog = QProg()
        >>> _ = prog << counter.cir(q_search=q_state[:2], q_count=q_state[2:])
        >>> m.run(prog, 1000)
        >>> res = m.result().get_prob_dict(q_state[2:])
        >>> y = int(max(res, key=res.get), 2)
        >>> print(round(4 * np.sin(np.pi * y / 8) ** 2, 4))
        0.5858

        """
        if q_ancilla is None:
            q_ancilla = []
        q_search = list(q_search)
        q_count = list(q_count)
        q_ancilla = list(q_ancilla)

        if len(q_search) != self.qnumber:
            raise ValueError('q_search must hold qnumber qubits')
        if len(q_count) != self.counting_qubits:
            raise ValueError('q_count must hold counting_qubits qubits')
        if len(q_ancilla) != self.ancilla_qubits:
            raise ValueError('q_ancilla must hold ancilla_qubits qubits')

        operator_g = self._G_cir(q_search, q_ancilla)

        circuit = QCircuit()
        circuit << apply_QGate(q_search + q_count, H)
        for k in range(self.counting_qubits):
            controlled_g = operator_g.control([q_count[k]])
            for _ in range(2 ** k):
                circuit << controlled_g
        circuit << QFT(q_count).dagger()
        return circuit

    def run(self):
        """
        Run the quantum counting algorithm.

        Returns
            count : ``float``\n
                The estimated number of marked items in the search space. Rounding it to
                the nearest integer returns M for the default counting register size.
                The estimated eigenphase is kept in the ``theta`` attribute.

        Examples
            An example for counting the marked states of a 3 qubit search space, where
            the oracle marks '001', '011' and '111'.

        >>> from pyqpanda_alg.QCounting import QCounting
        >>> counter = QCounting(qnumber=3, mark_data=['001', '011', '111'])
        >>> count = counter.run()
        >>> print(round(count, 4))
        3.2196
        >>> print(int(round(count)))
        3

        """
        qubits = QProg(self.qnumber + self.ancilla_qubits + self.counting_qubits).qubits()
        q_search = qubits[:self.qnumber]
        q_ancilla = qubits[self.qnumber:self.qnumber + self.ancilla_qubits]
        q_count = qubits[self.qnumber + self.ancilla_qubits:]

        prog = QProg()
        prog << self.cir(q_search, q_count, q_ancilla)
        self.machine.run(prog, 1000)
        result = self.machine.result().get_prob_dict(q_count)
        res_state = max(result, key=result.get)
        phase = int(res_state, 2) / 2 ** self.counting_qubits
        self.theta = 2 * pi * phase
        count = 2 ** self.qnumber * np.sin(pi * phase) ** 2
        return count
