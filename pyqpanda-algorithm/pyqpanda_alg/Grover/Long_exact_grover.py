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

from pyqpanda3.core import QCircuit, Z, X, H, BARRIER, U1
import numpy as np

from ..plugin import *
from .Grover_core import Grover


def exact_iter_num(q_num, sol_num):
    """
    Calculate the optimal iteration number J_op for the exact Grover
    search, i.e. formula (1):

        beta = arcsin( sqrt(M / N) ),   N = 2^q_num,  M = sol_num
        J_op = floor( (pi/2 - beta) / (2 * beta) )

    Parameters
        q_num : ``int``
            The number of qubits in the search space. N = 2^q_num.
        sol_num : ``int``
            Number of target solution states M.

    Returns
        num : ``int``
            J_op of formula (1). In this implementation the circuit is
            applied (J_op + 1) times by default, see LongExactGrover.
    """
    beta = np.arcsin(np.sqrt(sol_num / (2 ** q_num)))
    return int(np.floor((np.pi / 2 - beta) / (2 * beta)))


def exact_phase(q_num, sol_num):
    """
    Calculate the matched phase phi for the exact Grover search,
    i.e. formula (2):

        phi = 2 * arcsin( sin(pi / (4*J + 6)) / sin(beta) ),   J = J_op

    The SAME phase is used in both the oracle (marking good states) and
    the zero-reflection (diffusion) operator: every Z gate of the
    standard circuit is replaced by U1(phi).

    Parameters
        q_num : ``int``
        sol_num : ``int``

    Returns
        phi : ``float``
    """
    beta = np.arcsin(np.sqrt(sol_num / (2 ** q_num)))
    J = exact_iter_num(q_num, sol_num)
    return 2 * np.arcsin(np.sin(np.pi / (4 * J + 6)) / np.sin(beta))


def phase_mark_reflection(qubits: list = None, mark_data=None,
                          phase: float = np.pi):
    """
    Arbitrary-phase generalization of mark_data_reflection: the marked
    states gain a phase exp(i*phase) instead of -1.
    Setting phase = pi recovers the original mark_data_reflection.

    Parameters
        qubits : ``QVec``
        mark_data : ``str``, ``list[str]``
            Marked target state(s).
        phase : ``float``
            Phase (radians) applied to the marked states.

    Returns
        flip_operator : ``QCircuit``
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
        # 关键改动：Z -> U1(phase)   （原版：Z(qubits[-1]).control(...)）
        flip_operator << U1(qubits[-1], phase).control(qubits[:-1])
        for j in range(n):
            if i[-(j + 1)] == '0':
                flip_operator << X(qubits[j])
            else:
                flip_operator << BARRIER([qubits[j]])
    return flip_operator


def phase_zero_flip(qubits: list = None, phase: float = np.pi):
    """
    Arbitrary-phase generalization of the default zero_flip: the state
    |0...0> gains a phase exp(i*phase) instead of -1.
    Setting phase = pi recovers the original default zero_flip.

    Structure: (X layer) -> (multi-controlled U1(phase)) -> (X layer)

    Parameters
        qubits : ``QVec``
        phase : ``float``

    Returns
        circuit : ``QCircuit``
    """
    if not hasattr(qubits, '__len__'):
        qubits = [qubits]
    cir = QCircuit()
    if len(qubits) == 1:
        cir << X(qubits[0]) << U1(qubits[0], phase) << X(qubits[0])
    else:
        cir << apply_QGate(qubits, X)
        # 关键改动：Z -> U1(phase)   （原版：apply_QGate([q_zero[-1]], Z).control(...)）
        cir << U1(qubits[-1], phase).control(qubits[:-1])
        cir << apply_QGate(qubits, X)
    return cir


class LongExactGrover:
    """ This class provides a framework for the exact (100% success
    probability) Grover search [1][2], by replacing the phase-flip (-1)
    in BOTH the oracle and the diffusion operator with a matched
    arbitrary phase exp(i*phi).

    The phase and the number of iterations are chosen so that the
    success probability is exactly 1 (up to numerical precision):

        beta = arcsin( sqrt(M / N) )                          with M = sol_num
        J_op = floor( (pi/2 - beta) / (2*beta) )                          (1)
        phi  = 2 * arcsin( sin(pi/(4*J+6)) / sin(beta) ),  J = J_op       (2)

    In this codebase's operator convention (oracle first, then
    A.dagger() * U0 * A), the amplification operator is applied
    (J_op + 1) times by default (RUN_OFFSET = 1); equivalently,
    phi = 2*arcsin(sin(pi/(4K+2))/sin(beta)) with K the number of
    applied iterations. This has been verified analytically (N=4) and
    numerically (N=8) to give success probability 1. If your reference
    counts iterations differently, adjust RUN_OFFSET or pass `iternum`
    explicitly and check with the provided example.

    Parameters
        mark_data : ``str``, ``list[str]``
            Marked target state(s). M is inferred as len(mark_data).
        flip_operator : callable ``f(qubits)``
            Custom oracle. NOTE: to preserve exactness it should mark
            good states with phase exp(i*phi) (use U1(q, phi) instead
            of Z); obtain phi via exact_phase(). Requires sol_num.
        sol_num : ``int``
            Number of solutions. Required when flip_operator is given;
            otherwise inferred from mark_data.
        in_operator : callable ``f(qubits)``
            Initial state preparation, default Hadamards.
        iternum : ``int``
            Override for the number of iterations (default J_op + 1).
        phi : ``float``
            Override for the phase (default from formula (2)).

    References
        [1] L. K. Grover, A fast quantum mechanical algorithm for database
        search. STOC 1996, pp. 212-219. https://arxiv.org/abs/quant-ph/9605043
        [2] G. L. Long, Grover algorithm with arbitrary phases.
        Phys. Rev. A 64, 022307 (2001). https://arxiv.org/abs/quant-ph/0107117

    Examples
        An exact search for the state '101' on 3 qubits. The success
        probability is 1, whereas standard Grover only reaches ~0.946:

    >>> from pyqpanda3.core import CPUQVM, QProg
    >>> from pyqpanda_alg import Grover
    >>> m = CPUQVM()
    >>> q_state = list(range(3))
    >>> demo = Grover.LongExactGrover(mark_data='101')
    >>> prog = QProg()
    >>> prog << demo.cir(q_input=q_state)
    >>> m.run(prog, 1000)
    >>> res = m.result().get_prob_dict(q_state)
    >>> print(res)
    {'000': 0.0, '001': 0.0, '010': 0.0, '011': 0.0,
     '100': 0.0, '101': 1.0, '110': 0.0, '111': 0.0}
    """
    # 实际迭代次数 = J_op + RUN_OFFSET。
    # 本代码算子约定下 RUN_OFFSET=1 使成功概率恰为 1（已验证）；
    # 若与你手头资料的计数约定不同，可改为 0 并用 example 验证。
    RUN_OFFSET = 1

    def __init__(self,
                 mark_data=None,
                 flip_operator=None,
                 sol_num=None,
                 in_operator=None,
                 iternum=None,
                 phi=None):
        self.mark_data = mark_data
        self.flip_operator = flip_operator
        self.sol_num = sol_num
        self.in_operator = hadamard_circuit if in_operator is None else in_operator
        self.iternum = iternum
        self.phi = phi
        # 诊断信息（cir() 调用后可查看）
        self.last_J = None
        self.last_iternum = None
        self.last_phase = None

    def cir(self, q_input=None, q_flip=None, q_zero=None):
        """
        Get the full circuit of the exact Grover search.

        Parameters
            q_input : ``QVec``
                Search register (defines N = 2^len(q_input)).
            q_flip : ``QVec``
                Target qubit(s) for the oracle, default q_input.
            q_zero : ``QVec``
                Target qubit(s) for the zero-reflection, default q_input.

        Returns
            circuit : ``QCircuit``
        """
        if not hasattr(q_input, '__len__'):
            q_input = [q_input]

        # ---------- 确定解的数量 M ----------
        if self.mark_data is not None:
            marks = [self.mark_data] if isinstance(self.mark_data, str) \
                else list(self.mark_data)
            M = len(marks)
        elif self.flip_operator is not None:
            if self.sol_num is None:
                raise ValueError("自定义 flip_operator 时需要同时给出 sol_num")
            marks = None
            M = self.sol_num
        elif self.sol_num is not None:
            raise ValueError("请提供 mark_data 或自定义 flip_operator")
        else:
            raise ValueError("LongExactGrover 需要 mark_data 或 flip_operator+sol_num")

        # ---------- 公式(1)(2)：迭代数与相位 ----------
        n = len(q_input)
        J = exact_iter_num(n, M)
        K = self.iternum if self.iternum is not None else J + self.RUN_OFFSET
        phi = self.phi if self.phi is not None else exact_phase(n, M)

        self.last_J = J
        self.last_iternum = K
        self.last_phase = phi

        # ---------- 构造任意相位算子（Z -> U1(phi)，两处同相位） ----------
        if self.flip_operator is not None:
            flip = self.flip_operator
        else:
            def flip(qubits, marks=marks, phi=phi):
                return phase_mark_reflection(qubits, marks, phi)

        def zero(qubits, phi=phi):
            return phase_zero_flip(qubits, phi)

        # ---------- 复用原有 Grover 的电路组装 ----------
        grover = Grover(in_operator=self.in_operator,
                        flip_operator=flip,
                        zero_flip=zero)
        return grover.cir(q_input=q_input, q_flip=q_flip,
                          q_zero=q_zero, iternum=K)
