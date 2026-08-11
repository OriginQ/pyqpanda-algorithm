"""
Prepare Dicke state D(n,k) in an n-qubit system with k-Hamming-weight.
When k=1, this module is equivalent to W state generation.
A build-in W state generation supports implementation on a linear architecture.
Ref. https://doi.org/10.1109/QCE53715.2022.00027
Ref. https://doi.org/10.1002/qute.201900015

"""
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

from math import ceil, acos
from scipy.special import comb
from pyqpanda3.core import QCircuit, X, RY, CNOT, SWAP, TOFFOLI, BARRIER




def _control_ry(q1, q2, ang, ctrl_q=None, compress=True):
    """
    Control RY gate and CC-RY gate
    :param q1: applied qubit 1, control qubit
    :param q2: applied qubit 2 with Ry rotation gates applied, target qubit
    :param ang: rotation beta
    :param ctrl_q: optional control qubits
    :param compress: compress the basic gate implementation, default is True
    :return: a pyqpanda circuit
    """
    circuit = QCircuit()
    # C-Ry gate or CC-Ry gate
    if compress:
        if ctrl_q is None:
            circuit << RY(q2, ang).control(q1)
        else:
            circuit << RY(q2, ang).control([q1, ctrl_q])
    else:
        if ctrl_q is None:
            circuit << RY(q2, ang / 2) << CNOT(q1, q2) \
                    << RY(q2, -ang / 2) << CNOT(q1, q2)
        else:
            circuit << RY(q2, ang / 2) << TOFFOLI(q1, ctrl_q, q2) \
                    << RY(q2, -ang / 2) << TOFFOLI(q1, ctrl_q, q2)
    return circuit


def _rotation_swap(q1, q2, ang, ctrl_q=None, compress=True):
    """
    Rotation SWAP gate. Identity to XY mixer. Implemented with ordinary C-RY methods
    :param q1: applied qubit 1, control qubit
    :param q2: applied qubit 2 with Ry rotation gates applied, target qubit
    :param ang: rotation beta
    :param ctrl_q: optional control qubits
    :param compress: compress the basic gate implementation, default is True
    :return: a pyqpanda circuit
    """
    circuit = QCircuit()
    circuit << CNOT(q2, q1)
    # C-Ry gate or CC-Ry gate
    if ctrl_q is None:
        circuit << _control_ry(q1, q2, ang, compress=compress)
    else:
        circuit << _control_ry(q1, q2, ang, ctrl_q=ctrl_q, compress=compress)
    circuit << CNOT(q2, q1)
    return circuit


def _padded_to_one_hot(qb, reverse=False):
    """
    Change from padded encoding to one-hot encoding. Also applied for the reversed progress
    :param qb: workspace of qubits
    :param reverse: False for padded encoding to one-hot encoding and vice vesa
    :return : the pyqpanda circuit
    """
    circuit = QCircuit()
    seq = list(zip(qb[1:], qb))
    if reverse:
        seq.reverse()
    for qc, qt in seq:
        circuit << CNOT(qc, qt)
    return circuit


def _generate_ry_angles(n, m, nl):
    """
    Calculate the rotation angles of the hamming weight distribution module
    :param n: total number of qubits
    :param m: number of the qubits in the upper regime
    :param nl: value of Hamming weight
    :return : the rotation angles of the weight distribution RY gates
    """
    ctrl_size = min([n - m, nl])
    ttw = comb(n, nl)
    pw = [comb(m, nl - i) * comb(n - m, i) for i in range(ctrl_size + 1)]
    acc_w = [ttw - sum(pw[:i]) for i in range(ctrl_size)]
    ang = [acos(2 * x / y - 1) for x, y in zip(pw, acc_w)]
    return ang


def _distribute_weight(qb1, qb2, n, m, k, classic=None, compress=True):
    """
    Distribute Hamming weight from upper block to lower block
    :param qb1: the qubits in the upper regime
    :param qb2: qubits in the lower
    :param n: the total number of qubits
    :param m: the size of the upper regime
    :param k: the maximum Hamming weight
    :param classic: int or None. If the input state is classical, a number of gates can be reduced
                    classic (int) specifies the Hamming weight of the input classical state
    :param compress: compress the basic gate implementation, default is True
    :return: a pyqpanda circuit
    """
    circuit = QCircuit()
    if type(classic) is int:
        if classic > k:
            raise ValueError('The given classic assumption should less than k.')
        ang = _generate_ry_angles(n, m, classic)
        ctrl_size = min([classic, n - m])
        for j in range(ctrl_size):
            if j == 0:
                circuit << RY(qb2[j], ang[j])
            else:
                circuit << _control_ry(qb2[j - 1], qb2[j], ang[j], compress=compress)
    else:
        for i in reversed(range(k)):
            qc_u = qb1[i]
            ang = _generate_ry_angles(n, m, i + 1)
            ctrl_size = min([i + 1, n - m])
            for j in range(ctrl_size):
                if j == 0:
                    circuit << _control_ry(qc_u, qb2[j], ang[j], compress=compress)
                else:
                    circuit << _control_ry(qc_u, qb2[j], ang[j], ctrl_q=qb2[j - 1], compress=compress)
    return circuit


def _ladder_plus(qts, qc):
    """
    Periodic bit-wise operation. 001 --> 100
    :param qts: working space, the target qubit list
    :param qc: control bit
    :return: a pyqpanda circuit
    """
    circuit = QCircuit()
    for q1, q2 in zip(qts, qts[1::]):
        circuit << SWAP(q1, q2).control(qc)
    return circuit


def _weight_reduction(qb1, qb2, k, classic=None):
    """
    Reduce Hamming weight in the upper regime after distributing Hamming weight correctly
    :param qb1: qubit list of the upper regime which involves only the top-l qubits
    :param qb2: the top-l qubits list of the lower regime
    :param k: the maximum Hamming weight
    :param classic: If a classical input is given, the reduction can be done with in exactly one depth
    :return: the pyqpanda circuit
    """
    circuit = QCircuit()
    if type(classic) is int:
        ctrl_size = min([len(qb2), classic])
        qb1 = qb1[(k - ctrl_size): k]
        qb2 = qb2[:ctrl_size]
        for qt, qc in zip(qb1[::-1], qb2):
            circuit << CNOT(qc, qt)
    else:
        size2 = len(qb2)
        for i in reversed(range(k)):
            if i < size2:
                qc = qb2[i]
                ql = qb1[i: k]
                if len(ql) > 1:
                    circuit << _ladder_plus(ql, qc)
                circuit << CNOT(qc, qb1[-1])
    return circuit


def _weight_distribution_block(q_list, n, m, k, classic=None, compress=True):
    """
    The general Hamming weight distribution block serving for Dicke state unitary
    :param q_list: the whole working space of the block
    :param n: the total number of qubits
    :param m: the size of the upper regime
    :param k: the maximum Hamming weight
    :param classic: int or None. If the input state is classical, a number of gates can be reduced
    :param compress: compress the basic gate implementation, default is True
    :return: the pyqpanda circuit
    """
    if m < (n - m):
        raise ValueError('Expect the upper regime is bigger.')

    qb_upper = q_list[:k]
    qb_lower = q_list[m:m + k]
    block = QCircuit()
    # switch to one-hot encoding, truncated for classical input
    if classic is None:
        block << _padded_to_one_hot(qb_upper, reverse=False)
    # distribute Hamming weight to lower block
    block << _distribute_weight(qb_upper, qb_lower, n, m, k, classic=classic, compress=compress)
    # switch back to padded-encoding, truncated for classical input
    if classic is None:
        block << _padded_to_one_hot(qb_upper, reverse=True)
    # reduce Hamming weight in the upper regime
    block << _weight_reduction(qb_upper, qb_lower, k, classic=classic)
    return block


def _cal_left(n, k):
    """
    Calculate value of left sub-node of the root  with value n, with Hamming weight k
    :param n: the total number of qubits
    :param k: the target Hamming weight of the Dicke state to be prepared
    :return: value of left node, int type
    """
    size, resi = n // k, n % k
    if size == 0:
        return None
    elif size % 2 == 0:
        return k * (size // 2) + resi
    else:
        return k * ceil(size / 2)


def _grow_wbd_tree(n, k):
    """
    Grow the tree of weight distribution block. The tree is supposed to be balanced and
    the weight of each node should be less than or equal to k, the target Hamming weight.
    Each node contains the qubits (int-labelled) of each block to work on
    :param n: the total number of qubits
    :param k: the target Hamming weight of the Dicke state to be prepared
    :return: a list form of tree nodes by depth
    """
    tree = [[[(0, n), _cal_left(n, k)]]]
    while True:
        # generate new layer
        new_layer = []
        for (qb_st, qb_end), lw in tree[-1]:
            if lw is not None:
                node = qb_end - qb_st
                l_end = qb_st + lw
                sub_lw = _cal_left(lw, k)
                sub_rw = _cal_left(node - lw, k)
                if l_end - qb_st > k:
                    new_layer.append([(qb_st, l_end), sub_lw])
                else:
                    new_layer.append([(qb_st, l_end), None])
                if qb_end - l_end > k:
                    new_layer.append([(l_end, qb_end), sub_rw])
                else:
                    new_layer.append([(l_end, qb_end), None])
        # generate block parameters
        if len(new_layer) > 0:
            tree.append(new_layer)
        else:
            return tree


def _movement_block(q_list, compress=True):
    """
    Elementary part of Ukk block
    :param q_list: qubits working space
    :param compress: compress the basic gate implementation, default is True
    :return: a pyqpanda circuit
    """
    n = len(q_list)
    circuit = QCircuit()
    if n == 1:
        return circuit
    elif n == 2:
        circuit << _rotation_swap(q_list[0], q_list[1], acos(2 / n - 1), compress=compress)
    else:
        circuit << _rotation_swap(q_list[0], q_list[1], acos(2 / n - 1), compress=compress)
        for i in range(2, n):
            ang = acos(2 * i / n - 1)
            circuit << _rotation_swap(q_list[0], q_list[i], ang, ctrl_q=q_list[i - 1], compress=compress)
    return circuit


def _smooth_block(q_list, compress=True):
    """
    Smooth concentrated Hamming weight in the qubit working space
    :param q_list: qubits working space
    :param compress: compress the basic gate implementation, default is True
    :return: a pyqpanda circuit
    """
    n = len(q_list)
    circuit = QCircuit()
    if n == 1:
        return circuit
    else:
        circuit << _movement_block(q_list, compress=compress) << _smooth_block(q_list[1::], compress=compress)
        return circuit


def prepare_dicke_state(q_list, k, compress=True):
    """Prepare Dicke state.

    The Dicke state is defined as :math:`D_{n}^{(k)} = \sum_{hmw(i)=k} |i \\rangle`,
    which is equally superposition state of all states with the same Hamming weight.
    The method prepare Dicke state with in :math:`O(k*log(n/k))` depth in
    all-to-all connectivity architecture.

    Parameters
        q_list: ``List[int]``, shape (n,)
            Qubit addresses. List size is supposed to be the :math:`n`
            of :math:`D_{n}^{(k)}`.
        k : ``int``, k>0
            The target Hamming weight of the Dicke state to be prepared,
            *i.e.*, the :math:`k` of :math:`D_{n}^{(k)}`.
        compress : ``bool``, ``optional``
            If True, compress the basic gate implementation with simulated control
            gates otherwise using basic gate implementation; default is True.

    Return
        circuit : ``pyqpanda QCircuit``
            A pyqpanda QCircuit which assumes the input state is all 0.

    Raises
        ValueError
            If the target Hamming weight is larger than the input qubit number (:math:`k<n`),
            or k is invalid (:math:`k<0`), or qubit number is 0 (:math:`n=0`).

    Reference
        [1] Bärtschi A, Eidenbenz S. Short-depth circuits for dicke state preparation[C]
        2022 IEEE International Conference on Quantum Computing and Engineering (QCE). IEEE, 2022: 87-96.
        https://doi.org/10.1109/QCE53715.2022.00027

    Examples
        .. code-block:: python

            from pyqpanda3.core import QProg
            from pyqpanda_alg.QAOA import dstate
            from pyqpanda_alg.execution import LocalBackend, ExecutionOptions

            n = 4
            k = 2
            qubits = list(range(n))
            prog = QProg()
            prog << dstate.prepare_dicke_state(qubits, k)
            backend = LocalBackend()
            state = backend.submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
            for index, amp in enumerate(state):
                key = format(index, '04b')[::-1]
                if key.count('1') == k and abs(amp) ** 2 > 0:
                    print(key, abs(amp) ** 2)


    The given example illustrates how to prepare the state :math:`D_4^{(2)}`.
    The corresponding quantum circuit is:

    .. parsed-literal::

                  ┌─┐     !                               ┌────┐         ! ┌────┐                ┌────┐
        q_0:  \|0>─┤X├ ────! ────────────── ────────────── ┤CNOT├──── ────! ┤CNOT├ ───────■────── ┤CNOT├
                  ├─┤     !                               └──┬┬┴───┐     ! └──┬─┘ ┌──────┴─────┐ └──┬─┘
        q_1:  \|0>─┤X├ ────! ────────────── ────────────── ───┼┤CNOT├ ────! ───■── ┤RY(1.570796)├ ───■──
                  └─┘     ! ┌────────────┐                   │└──┬─┘     ! ┌────┐ └────────────┘ ┌────┐
        q_2:  \|0>──── ────! ┤RY(2.300524)├ ───────■────── ───┼───■── ────! ┤CNOT├ ───────■────── ┤CNOT├
                          ! └────────────┘ ┌──────┴─────┐    │           ! └──┬─┘ ┌──────┴─────┐ └──┬─┘
        q_3:  \|0>──── ────! ────────────── ┤RY(0.927295)├ ───■────── ────! ───■── ┤RY(1.570796)├ ───■──
                          !                └────────────┘                !        └────────────┘

    And the probability of all possible state are (with possible floating errors):

    .. parsed-literal::
        0011 0.16666666666666663
        0101 0.1666666666666667
        0110 0.1666666666666667
        1001 0.1666666666666667
        1010 0.1666666666666667
        1100 0.16666666666666663

    which include all states with the same Hamming weight :math:`k=2`.

    """
    n = len(q_list)
    circuit = QCircuit()
    if k > n or k < 0 or n == 0:
        raise ValueError('Invalid Dicke state parameter: k > n')
    elif k == n:
        # circuit << X(q_list)
        for i in range(n):
            circuit << X(q_list[i])
        return circuit
    elif k > ceil(n / 2):
        k = n - k
        tail_x = True
    elif k == 0:
        return circuit
    else:
        tail_x = False

    # classical initial state preparation
    for qb in q_list[:k]:
        circuit << X(qb)
    circuit << BARRIER(q_list)
    # weight distribution blocks and smooth blocks are applied according to the tree
    for layer in _grow_wbd_tree(n, k):
        for (q_st, q_end), lw in layer:
            q_work = q_list[q_st: q_end]
            if lw is None:
                circuit << _smooth_block(q_work, compress=compress)
            else:
                w_size = len(q_work)
                if w_size == n:
                    circuit << _weight_distribution_block(q_work, w_size, lw, k, classic=k, compress=compress)
                else:
                    circuit << _weight_distribution_block(q_work, w_size, lw, k, compress=compress)
    # symmetry Dicke state
    if tail_x:
        # circuit << X(q_list)
        for q in q_list:
            circuit << X(q)
    return circuit


def linear_w_state(q_list, compress=True):
    """Prepare W state with a divide-and-conquer algorithm on the linear architecture device.

    W state is the special Dicke state where :math:`k=1`. The special case is compatible to
    Dicke state preparation while it can be formalized on a linear connectivity device with
    exactly :math:`n-1` depth and :math:`3n-3` CNOT gates.

    Parameters
         q_list: ``List[int]``, shape (n,)
            Qubit addresses. List size is supposed to be the :math:`n` of :math:`D_{n}^{(1)}`.
         compress  : ``bool``, ``optional``
            If True, compress the basic gate implementation with simulated control
            gates otherwise using basic gate implementation; default is True.

    Return
        circuit : ``pyqpanda QCircuit``
            A pyqpanda QCircuit which assumes the input state is all 0.

    Raises
        ValueError
            If the input qubit number is zero.

    Reference
        Cruz D, Fournier R, Gremion F, et al.
        Efficient quantum algorithms for ghz and w states, and implementation on the IBM quantum computer[J].
        Advanced Quantum Technologies, 2019, 2(5-6): 1900015.

    Example
        .. code-block:: python

            from pyqpanda3.core import QProg
            from pyqpanda_alg.QAOA import dstate
            from pyqpanda_alg.execution import LocalBackend, ExecutionOptions

            n = 3
            qubits = list(range(n))
            prog = QProg()
            prog << dstate.linear_w_state(qubits, compress=True)
            backend = LocalBackend()
            state = backend.submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
            for index, amp in enumerate(state):
                key = format(index, '03b')[::-1]
                if key.count('1') == 1 and abs(amp) ** 2 > 0:
                    print(key, abs(amp) ** 2)

    The example prepare W state on a 3-qubit system which is linearly connected.
    The corresponding circuit reads as:

    .. parsed-literal::
                  ┌─┐                ┌────┐
        q_0:  \|0>─┤X├ ───────■────── ┤CNOT├ ────────────── ──────
                  └─┘ ┌──────┴─────┐ └──┬─┘                ┌────┐
        q_1:  \|0>──── ┤RY(1.910633)├ ───■── ───────■────── ┤CNOT├
                      └────────────┘        ┌──────┴─────┐ └──┬─┘
        q_2:  \|0>──── ────────────── ────── ┤RY(1.570796)├ ───■──
                                            └────────────┘

    The resulting state should be like (with possible floating errors):

    .. parsed-literal::
        001 0.3333333333333333
        010 0.3333333333333334
        100 0.3333333333333334

    Each of them is one-Hamming-weight.

    """
    n = len(q_list)
    if n == 0:
        raise ValueError('Apply on at least one qubit.')

    circuit = QCircuit()
    circuit << X(q_list[0])
    if n > 1:
        for i in range(1, n):
            ang = acos(2 / (n-i+1) - 1)
            circuit << _control_ry(q_list[i - 1], q_list[i], ang, compress=compress) \
            << CNOT(q_list[i], q_list[i - 1])
    return circuit
