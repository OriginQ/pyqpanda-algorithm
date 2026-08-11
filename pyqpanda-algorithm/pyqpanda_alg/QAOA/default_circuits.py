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

import numbers

import numpy as np
from itertools import combinations
from pyqpanda3.core import QCircuit, RX, CNOT, RZ
from . import dstate





def iswap(q1, q2, angle):
    """
    Quantum circuits to generate a 2-qubit iswap gate, which can express as :math:`\exp(-i(X_i X_j + Y_i Y_j)t)`.
    
    Parameters
        q1 : ``Qubit``
            Qubit 1.\n
            
        q2 : ``Qubit``
            Qubit 2.\n
            
        angle : ``float``
            angle :math:`t` of :math:`e^{-i(X_i X_j + Y_i Y_j)t}`\n
            
    Return
        xycir : ``QCircuit``\n
            Circuit of simulation a 2-qubit iswap gate.\n
    """
    xycir = QCircuit()
    xycir << RX(q1, np.pi / 2) << RX(q2, np.pi / 2) << CNOT(q1, q2) << RX(q1, angle) << RZ(q2, angle) << CNOT(
        q1, q2) << RX(q1, -np.pi / 2) << RX(q2, -np.pi / 2)
    return xycir


def parity_partition_xy_mixer(qlist, beta):
    """
    Quantum circuits to approximate a parity-partition XY mixer.

    Parameters
        qlist : ``list``
            Qubits list.\n

        beta : ``float``
            angle :math:`t` of :math:`e^{-iHt}`\n

    Return
        cir : ``QCircuit``
            Circuit of simulation a parity-partition XY mixer :math:`e^{-iHt}`.\n

        
    Examples   
        Generate a circuit of simulation a parity-partition XY mixer :math:`e^{-iH\pi/2}`.

    
    >>> import numpy as np
    >>> from pyqpanda_alg.QAOA import default_circuits
    >>> qubits = list(range(4))
    >>> circuit = default_circuits.parity_partition_xy_mixer(qubits, np.pi/2)
    >>> print(circuit)

    .. parsed-literal::


                  ┌──────────────┐        ┌───────────────┐        ┌───────────────┐ ┌──────────────┐       ┌────┐ ┌───────────────┐       ┌────┐ ┌───────────────┐ 
        q_0:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ──────┤CNOT├ ┤RZ(-3.14159265)├ ──────┤CNOT├ ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ ├──────────────┤       └──┬─┘ ├───────────────┤       └──┬─┘ ├───────────────┤ 
        q_1:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───*─────┼── ┤RX(-3.14159265)├ ───*─────┼── ┤RX(-1.57079633)├ 
                  ├──────────────┤ └────┘ ├───────────────┤ └────┘ ├───────────────┤ ├──────────────┤ ┌──┴─┐   │   ├───────────────┤ ┌──┴─┐   │   ├───────────────┤ 
        q_2:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ┤CNOT├───┼── ┤RZ(-3.14159265)├ ┤CNOT├───┼── ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ ├──────────────┤ └────┘   │   ├───────────────┤ └────┘   │   ├───────────────┤ 
        q_3:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ─────────*── ┤RX(-3.14159265)├ ─────────*── ┤RX(-1.57079633)├ 
                  └──────────────┘ └────┘ └───────────────┘ └────┘ └───────────────┘ └──────────────┘              └───────────────┘              └───────────────┘ 
         c :   / ═
          

    Note
        For a given XY mixer Hamiltonian

         
        .. math::
            H_{XY,v} = \\frac{1}{2} \sum_{c,c'\in K} (\sigma_{v,c}^x \sigma_{v',c}^x + \sigma_{v,c}^y \sigma_{v',c}^y)

    When the mixer-set :math:`K` takes a one dimensional structure: :math:`c'=c+1`, and periodic boundary condition,
    it is termed a ring mixer. In order to simulate a ring mixer in quantum circuit, a one-order approximation by
    applying local XY-Hamiltonian on even paris first and local pairs next is used, which is called parity-partition XY
    mixer. The leading error term is in order of the number of qubits in the domain. See details in [1]

    Reference
        [1] WANG Z, RUBIN N C, DOMINY J M, et. XY-mixers: analytical and numerical results for QAOA[J/OL].
        Physical Review A, 2020, 101(1): 012320. DOI:10.1103/PhysRevA.101.012320.

    """
    q_num = len(qlist)
    cir = QCircuit()

    beta = - 2 * beta
    if q_num == 1:
        # cir << pq.RX(qlist, beta)
        cir << RX(qlist[0], beta)
    if q_num == 2:
        # cir << pq.iSWAP(qlist[0], qlist[1], beta)
        cir << iswap(qlist[0], qlist[1], beta)
    else:
        for i in range(0, q_num, 2):
            if i + 1 < q_num:
                # cir << pq.iSWAP(qlist[i], qlist[i+1], beta)
                cir << iswap(qlist[i], qlist[i + 1], beta)
        for i in range(1, q_num, 2):
            if i + 1 < q_num:
                # cir << pq.iSWAP(qlist[i], qlist[i + 1], beta)
                cir << iswap(qlist[i], qlist[i + 1], beta)
            else:
                # cir << pq.iSWAP(qlist[-1], qlist[0], beta)
                cir << iswap(qlist[-1], qlist[0], beta)

    return cir


def complete_xy_mixer(qlist, angle):
    """
    Quantum circuits to approximate a complete XY mixer.

    Parameters
        qlist : ``list``
            Qubits list.\n

        angle : ``float``
            beta :math:`t` of :math:`e^{-iHt}`\n

    Return
        cir : ``QCircuit``
            Circuit of simulation a complete XY mixer :math:`e^{-iHt}`.\n

    Examples
        Generate a circuit of simulation a complete XY mixer :math:`e^{-iH\pi/2}`.

        
    >>> import numpy as np
    >>> from pyqpanda_alg.QAOA import default_circuits
    >>> qubits = list(range(4))
    >>> circuit = default_circuits.complete_xy_mixer(qubits, np.pi/2)
    >>> print(circuit)

    .. parsed-literal::
                  ┌──────────────┐        ┌───────────────┐        ┌───────────────┐ ┌──────────────┐               ┌───────────────┐ >
        q_0:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───*── ────── ┤RX(-3.14159265)├ >
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ ├──────────────┤    │          ├───────────────┤ >
        q_1:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───┼── ───*── ┤RX(-3.14159265)├ >
                  ├──────────────┤ └────┘ ├───────────────┤ └────┘ ├───────────────┤ ├──────────────┤ ┌──┴─┐    │   ├───────────────┤ >
        q_2:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ┤CNOT├ ───┼── ┤RZ(-3.14159265)├ >
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ ├──────────────┤ └────┘ ┌──┴─┐ ├───────────────┤ >
        q_3:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ────── ┤CNOT├ ┤RZ(-3.14159265)├ >
                  └──────────────┘ └────┘ └───────────────┘ └────┘ └───────────────┘ └──────────────┘        └────┘ └───────────────┘ >
         c :   / ═
          

                               ┌───────────────┐ ┌──────────────┐               ┌───────────────┐               ┌───────────────┐ 
        q_0:  \|0>───*── ────── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───*── ────── ┤RX(-3.14159265)├ ───*── ────── ┤RX(-1.57079633)├ 
                    │          ├───────────────┤ ├──────────────┤    │          ├───────────────┤    │          ├───────────────┤ 
        q_1:  \|0>───┼── ───*── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───┼── ───*── ┤RX(-3.14159265)├ ───┼── ───*── ┤RX(-1.57079633)├ 
                 ┌──┴─┐    │   ├───────────────┤ ├──────────────┤    │   ┌──┴─┐ ├───────────────┤    │   ┌──┴─┐ ├───────────────┤ 
        q_2:  \|0>┤CNOT├ ───┼── ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ───┼── ┤CNOT├ ┤RZ(-3.14159265)├ ───┼── ┤CNOT├ ┤RX(-1.57079633)├ 
                 └────┘ ┌──┴─┐ ├───────────────┤ ├──────────────┤ ┌──┴─┐ └────┘ ├───────────────┤ ┌──┴─┐ └────┘ ├───────────────┤ 
        q_3:  \|0>────── ┤CNOT├ ┤RX(-1.57079633)├ ┤RX(1.57079633)├ ┤CNOT├ ────── ┤RZ(-3.14159265)├ ┤CNOT├ ────── ┤RX(-1.57079633)├ 
                        └────┘ └───────────────┘ └──────────────┘ └────┘        └───────────────┘ └────┘        └───────────────┘ 
         c :   / 

    Note
        For a given XY mixer Hamiltonian
    
        .. math::
            H_{XY,v}=\\frac{1}{2}\sum_{c,c'\in K} (\sigma_{v,c}^x \sigma_{v',c}^x + \sigma_{v,c}^y \sigma_{v',c}^y)

    When the mixer-set :math:`K` includes all pairs, it is termed a complete mixer. In order to simulate a complete
    mixer in quantum circuit, a partition of :math:`\kappa - 1, \kappa=\lceil \log_2 n \\rceil`  is applied, where
    :math:`n` is the number of qubits. See details in [1]

    Reference
        [1] WANG Z, RUBIN N C, DOMINY J M, et. XY-mixers: analytical and numerical results for QAOA[J/OL].
        Physical Review A, 2020, 101(1): 012320. DOI:10.1103/PhysRevA.101.012320.

    """

    q_num = len(qlist)
    angle = -2 * angle
    m = 0
    k1 = 1
    while k1 < q_num:
        k1 = k1 * 2
        m += 1

    combi_list = []
    for i in range(1, m + 1):
        combi_list += list(combinations(range(m), i))

    cir = QCircuit()
    pair_list = []
    for combi in combi_list:

        for first in range(q_num):
            second = first
            for j in combi:
                second = second ^ (1 << j)  # flip bit j of first

            if (first, second) not in pair_list and (second, first) not in pair_list \
                    and max(first, second) < q_num:
                pair_list.append((first, second))

    for q1, q2 in pair_list:
        # cir << pq.iSWAP(qlist[q1], qlist[q2], angle)
        cir << iswap(qlist[q1], qlist[q2], angle)

    return cir


def xy_mixer(domains, mixer_type='PXY'):
    """
    Generate XY mixer circuit.

    Parameters
        domains : ``integer`` or ``list[list]``
            Nodes of each XY mixer to be applied. If an integer n is given, the qubit list is divided into n parts. If a
            list is given, the mixer is applied to each domain.

        mixer_type: ``string`` or ``float``
            How the mixer is implemented. Should be one of

                    - ``PXY`` : Parity partition XY mixer.
                        See 'parity_partition_xy_mixer'
                    - ``CXY`` : Complete XY mixer
                        See 'complete_xy_mixer'

                If not given, default by ``PXY``.
                When a number is given instead, it is treated as the mixer
                angle :math:`t` and ``domains`` as a flat qubit list, and the
                parity-partition XY mixer circuit is returned directly.

    Return
        mixer_circuit : ``func(pq.QCircuit)`` or ``pyqpanda QCircuit``
            By default, a function which use qubit list and angles as input, output a circuit of simulation a XY mixer :math:`e^{-iHt}`.
            When the angle is given as the second argument, the XY mixer circuit is returned directly.

    Examples
        Generate a circuit of simulation a complete XY mixer :math:`e^{-iH\pi/2}` in qubits [0, 1] and [2, 3].

    >>> import numpy as np
    >>> from pyqpanda_alg.QAOA import default_circuits
    >>> qubits = list(range(4))
    >>> circuit1 = default_circuits.xy_mixer(2, 'PXY')(qubits, np.pi/2)
    >>> circuit2 = default_circuits.xy_mixer([[0,1], [2,3]], 'PXY')(qubits, np.pi/2)
    >>> print(circuit1)
    >>> print(circuit2)

    .. parsed-literal::

          ┌──────────────┐        ┌───────────────┐        ┌───────────────┐ 
        q_0:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ 
        q_1:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ 
                  ├──────────────┤ └────┘ ├───────────────┤ └────┘ ├───────────────┤ 
        q_2:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ 
        q_3:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ 
                  └──────────────┘ └────┘ └───────────────┘ └────┘ └───────────────┘ 
         c :   / ═


                  ┌──────────────┐        ┌───────────────┐        ┌───────────────┐ 
        q_0:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ 
        q_1:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ 
                  ├──────────────┤ └────┘ ├───────────────┤ └────┘ ├───────────────┤ 
        q_2:  \|0>─┤RX(1.57079633)├ ───*── ┤RX(-3.14159265)├ ───*── ┤RX(-1.57079633)├ 
                  ├──────────────┤ ┌──┴─┐ ├───────────────┤ ┌──┴─┐ ├───────────────┤ 
        q_3:  \|0>─┤RX(1.57079633)├ ┤CNOT├ ┤RZ(-3.14159265)├ ┤CNOT├ ┤RX(-1.57079633)├ 
                  └──────────────┘ └────┘ └───────────────┘ └────┘ └───────────────┘ 
         c :   / ═

    Note
        A XY mixer can enforce the state evolution in a feasible subspace which keep the hamming weight (total spin) of the
        state to be conserved.
        For example, in one-hot coding problem, a W state as initial state combine with the XY mixer
        can keep all solutions remain in one-hot form. See details in [1].

    Reference
        [1] WANG Z, RUBIN N C, DOMINY J M, et. XY-mixers: analytical and numerical results for QAOA[J/OL].
        Physical Review A, 2020, 101(1): 012320. DOI:10.1103/PhysRevA.101.012320.

    """

    if isinstance(mixer_type, numbers.Real):
        # Direct builder form: a flat qubit list and a mixer angle
        # produce the parity-partition XY mixer circuit directly.
        qlist = domains
        angle = mixer_type
        if not isinstance(qlist, list) or not all(isinstance(q, int) for q in qlist):
            raise TypeError(
                f"domains must be a list of qubit indices when the mixer "
                f"angle is given, rather than {type(domains)}"
            )
        return parity_partition_xy_mixer(qlist, angle)

    def mixer_circuit(qlist, angle):
        q_num = len(qlist)
        cir = QCircuit()
        domain_qlist = []

        if type(domains) is int:
            if q_num % domains != 0:
                # print('error')
                raise ValueError('q_num should be divisible by domains')
            else:
                features = int(q_num/domains)
                domain_qlist = list(range(q_num))
                domain_qlist = [domain_qlist[i:i+features] for i in range(0, q_num, features)]
        elif type(domains) is list:
            domain_qlist = domains
        elif domains is None:
            domain_qlist = [[i] for i in range(0, q_num)]
        for domain in domain_qlist:
            q_domain = [qlist[i] for i in domain]
            if mixer_type == 'PXY':
                cir << parity_partition_xy_mixer(q_domain, angle)
            elif mixer_type == 'CXY':
                cir << complete_xy_mixer(q_domain, angle)
            else:
                raise ValueError('mixer_type should be one of PXY or CXY')
        return cir

    return mixer_circuit


def init_d_state(domains, k=1, compress=True):
    """Prepare Dicke state.

    The Dicke state is defined as :math:`D_{n}^{(k)} = \sum_{hmw(i)=k} |i \\rangle`,
    which is equally superposition state of all states with the same Hamming weight.
    The method prepare an initial state, which is the product state of Dicke state in each domain,
    with in :math:`O(k*log(n/k))` depth in all-to-all connectivity architecture.

    Parameters
        domains : ``integer`` or ``list[list]``
            Nodes of each XY mixer to be applied. If an integer n is given, the qubit list is divided into n parts. If a
            list is given, the mixer is applied to each domain.
        k : ``integer``,  :math:`k>0`
            The target Hamming weight of the Dicke state to be prepared,
            *i.e.*, the :math:`k` of :math:`D_{n}^{(k)}`.
        compress : ``bool``, ``optional``
            If True, compress the basic gate implementation with simulated control
            gates otherwise using basic gate implementation; default is True.

    Return
        init_state_circuit : ``function``
            Return a function, which takes qubit list as input, and output a pyqpanda QCircuit which assumes
            the input state is all 0.

    Raises
        ValueError
            If the target Hamming weight is larger than the input qubit number (:math:`k<n`),
            or k is invalid (:math:`k<0`), or qubit number is 0 (:math:`n=0`).

    Reference
        Bärtschi A, Eidenbenz S. Short-depth circuits for dicke state preparation[C]
        2022 IEEE International Conference on Quantum Computing and Engineering (QCE). IEEE, 2022: 87-96.
        https://doi.org/10.1109/QCE53715.2022.00027

        
    Examples
        The given example illustrates how to prepare the state :math:`D_4^{(2)}`.

    >>> from pyqpanda3.core import QProg
    >>> from pyqpanda_alg.QAOA import default_circuits
    >>> from pyqpanda_alg.execution import LocalBackend, ExecutionOptions
    >>> n = 6
    >>> k = 2
    >>> qubits = list(range(n))
    >>> prog = QProg()
    >>> domain = [[0,1,2],[3,4,5]]
    >>> init_circuit = default_circuits.init_d_state(domain, k)
    >>> prog << init_circuit(qubits)
    >>> backend = LocalBackend()
    >>> state = backend.submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
    >>> for index, amp in enumerate(state):
    >>>     key = format(index, '06b')[::-1]
    >>>     prob = abs(amp) ** 2
    >>>     domain0_key = [key[i] for i in domain[0]]
    >>>     domain1_key = [key[i] for i in domain[1]]
    >>>     if prob > 0:
    >>>         print(key, domain0_key.count('1'), domain1_key.count('1'), prob)


    The corresponding quantum circuit is:

    .. parsed-literal::

                  ┌─┐                   ┌────┐                ┌────┐
        q_0:  \|0>─┤X├─────────── ────── ┤CNOT├ ───────■────── ┤CNOT├
                  ├─┤            ┌────┐ └──┬─┘ ┌──────┴─────┐ └──┬─┘
        q_1:  \|0>─┤X├─────────── ┤CNOT├ ───■── ┤RY(1.570796)├ ───■──
                  ├─┴──────────┐ └──┬─┘        └────────────┘
        q_2:  \|0>─┤RY(1.910633)├ ───■── ────── ────────────── ──────
                  ├─┬──────────┘        ┌────┐                ┌────┐
        q_3:  \|0>─┤X├─────────── ────── ┤CNOT├ ───────■────── ┤CNOT├
                  ├─┤            ┌────┐ └──┬─┘ ┌──────┴─────┐ └──┬─┘
        q_4:  \|0>─┤X├─────────── ┤CNOT├ ───■── ┤RY(1.570796)├ ───■──
                  ├─┴──────────┐ └──┬─┘        └────────────┘
        q_5:  \|0>─┤RY(1.910633)├ ───■── ────── ────────────── ──────
                  └────────────┘

    And the probability of all possible state are (with possible floating errors):

    .. parsed-literal::

            011011 2 2 0.1111111111111111
            011101 2 2 0.11111111111111113
            011110 2 2 0.11111111111111113
            101011 2 2 0.11111111111111113
            101101 2 2 0.11111111111111113
            101110 2 2 0.11111111111111113
            110011 2 2 0.11111111111111113
            110101 2 2 0.11111111111111113
            110110 2 2 0.11111111111111113

    which represents the state :math:`\ket{D_3^2}_{012}\otimes\ket{D_3^2}_{345}`.

    """

    def init_state_circuit(qlist):
        q_num = len(qlist)
        cir = QCircuit()
        domain_qlist = []

        if type(domains) is int:
            if q_num % domains != 0:
                raise ValueError('The number of qubits is not divisible by the number of domains.')
            else:
                features = int(q_num/domains)
                domain_qlist = list(range(q_num))
                domain_qlist = [domain_qlist[i:i+features] for i in range(0, q_num, features)]
        elif type(domains) is list:
            domain_qlist = domains
        elif domains is None:
            domain_qlist = [[i] for i in range(0, q_num)]

        for domain in domain_qlist:
            q_domain = [qlist[i] for i in domain]
            if k == 1:
                cir << linear_w_state(q_domain, compress)
            else:
                cir << prepare_dicke_state(q_domain, k, compress)
        return cir

    return init_state_circuit


def prepare_dicke_state(q_list, k, compress=True):
    """Prepare Dicke state.

    The Dicke state is defined as :math:`D_{n}^{(k)} = \sum_{hmw(i)=k} |i\\rangle` ,
    which is equally superposition state of all states with the same Hamming weight.
    The method prepare Dicke state with in :math:`O(k*log(n/k))` depth in
    all-to-all connectivity architecture.

    Parameters
        q_list : ``QVec``, ``List[Qubit]``, shape (n,)
            Qubit addresses. List size is supposed to be the :math:`n`
            of :math:`D_{n}^{(k)}`.\n
        k : ``integer``, k>0
            The target Hamming weight of the Dicke state to be prepared,
            *i.e.*, the :math:`k` of :math:`D_{n}^{(k)}`.
        compress : ``bool``, ``optional``
            If True, compress the basic gate implementation with simulated control
            gates otherwise using basic gate implementation; default is True.

    Return
        circuit : ``pyqpanda QCircuit``
            A pyqpanda QCircuit which assumes the input state is all 0.\n

    Raises
        ValueError
            If the target Hamming weight is larger than the input qubit number (:math:`k<n`),
            or k is invalid (:math:`k<0`), or qubit number is 0 (:math:`n=0`).

    Reference
        Bärtschi A, Eidenbenz S. Short-depth circuits for dicke state preparation[C]
        2022 IEEE International Conference on Quantum Computing and Engineering (QCE). IEEE, 2022: 87-96.
        https://doi.org/10.1109/QCE53715.2022.00027

    Examples
        The given example illustrates how to prepare the state :math:`D_4^{(2)}`.

    >>> from pyqpanda3.core import QProg
    >>> from pyqpanda_alg.QAOA.default_circuits import prepare_dicke_state
    >>> from pyqpanda_alg.execution import LocalBackend, ExecutionOptions
    >>> n = 4
    >>> k = 2
    >>> qubits = list(range(n))
    >>> prog = QProg()
    >>> prog << prepare_dicke_state(qubits, k)
    >>> backend = LocalBackend()
    >>> state = backend.submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
    >>> for index, amp in enumerate(state):
    >>>     key = format(index, '04b')[::-1]
    >>>     prob = abs(amp) ** 2
    >>>     if prob > 0:
    >>>         print(key, prob)


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
    return dstate.prepare_dicke_state(q_list, k, compress)


def linear_w_state(q_list, compress=True):
    """Prepare W state with a divide-and-conquer algorithm on the linear architecture device.

    W state is the special Dicke state where :math:`k=1`. The special case is compatible to
    Dicke state preparation while it can be formalized on a linear connectivity device with
    exactly :math:`n-1` depth and :math:`3n-3` CNOT gates.

    Parameters
        q_list : ``QVec``, ``List[Qubit]``, shape (n,)
            Qubit addresses. List size is supposed to be the :math:`n` of :math:`D_{n}^{(1)}`.\n

        compress : ``bool``, ``optional``
            If True, compress the basic gate implementation with simulated control
            gates otherwise using basic gate implementation; default is True.\n

    Return
        circuit : ``pyqpanda QCircuit``
            A pyqpanda QCircuit which assumes the input state is all 0.\n

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
            from pyqpanda_alg.QAOA import default_circuits
            from pyqpanda_alg.execution import LocalBackend, ExecutionOptions

            n = 3
            qubits = list(range(n))
            prog = QProg()
            prog << default_circuits.linear_w_state(qubits, compress=True)
            backend = LocalBackend()
            state = backend.submit_statevector(prog, options=ExecutionOptions()).result().single_statevector()
            for index, amp in enumerate(state):
                key = format(index, '03b')[::-1]
                if abs(amp) ** 2 > 0:
                    print(key, abs(amp) ** 2, key.count('1'))

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
        001 0.3333333333333333 1
        010 0.3333333333333334 1
        100 0.3333333333333334 1

    Each of them is one-Hamming-weight.

    """
    return dstate.linear_w_state(q_list, compress)
