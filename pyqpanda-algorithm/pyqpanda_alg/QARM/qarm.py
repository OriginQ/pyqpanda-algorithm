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

import os
import math
import warnings

import numpy as np
from pyqpanda3.core import QCircuit, QProg, X, H, U1, SWAP, draw_qprog
from pyqpanda3.intermediate_compiler import convert_qprog_to_originir
from itertools import chain

from ..execution import (
    AlgorithmInputError,
    AlgorithmTask,
    CompletedBackendTask,
    ExecutionOptions,
    register_algorithm,
    resolve_backend,
)

class QuantumAssociationRulesMining:
    """
    Association rule mining finds interesting associations or correlations between item sets in
    a large amount of data . Mining the implicit relationship between objects from large-scale
    data is called association analysis or association rule learning, which can reveal the hidden
    association pattern in data and help people to carry out market operation and decision support.
    For example, on the same trip to the supermarket, if the customer buys milk, what is the
    likelihood that he will also buy bread?

    Based on the famous classical association rule mining algorithm Apriori algorithm,
    a quantum association rule mining algorithm is proposed to realize the core task.
    Specifically, given a quantum black box accessing a trading database,
    the algorithm first uses the quantum parallel amplitude estimation algorithm
    to estimate the support of all candidate K term sets in a quantum parallel manner,
    and store it in a quantum superposition state.  Next, the quantum amplitude
    amplification algorithm is used to search the candidate K term sets that are not
    less than the predetermined threshold from the superposition quantum states.

    Parameters:
        show: ``string``
            Enumeration of the circuit show type,should be one of"None", "Picture"and"OriginIR"
            - None : no output;
            - Picture : Console output the quantum circuit,output default file name;
            - OriginIR : Console does not output,output the IR of the circuit.\n

        file_name: ``string``
            the output file namethat record the circuit information.

        machine_type: ``string``
            enumeration of QVM type, should be one of "CPU", and "QCloud" (deprecated).

        backend: execution backend, keyword-only
            The sampling backend driving the search.  Defaults to the local
            simulator.  Pass an explicit backend for the deprecated
            ``machine_type='QCloud'`` entry point.

    Returns:
        out: ``dict``
            confidence result

    Examples
        .. code-block:: python

            import os
            from pyqpanda_alg.QARM import QuantumAssociationRulesMining
            from pyqpanda_alg import QARM


            def read(file_path):
                if os.path.exists(file_path):
                    trans_data = []
                    with open(file_path, 'r', encoding='utf8') as f:
                        data_line = f.readlines()
                        if data_line:
                            for line in data_line:
                                if line:
                                    data_list = line.strip().split(',')
                                    trans_data.append([data.strip() for data in data_list])
                        else:
                            raise ValueError("The file {} has no any data!".format(file_path))
                else:
                    raise FileNotFoundError('The file {} does not exists!'.format(file_path))
                return trans_data


            if __name__ == '__main__':
                data_path = QARM.__path__[0]
                data_file = os.path.join(data_path, 'dataset/data2.txt')
                trans_data = read(data_file)
                support = 0.2
                conf = 0.5
                qarm = QuantumAssociationRulesMining(trans_data, support, conf)
                result = qarm.run()
                print(result)


    """
    def __init__(self, transaction_data, min_support=0.2, min_conf=0.3):
        if min_support < 0 or min_support > 1:
            raise ValueError("min_support must be between 0 and 1")
        if min_conf < 0 or min_conf > 1:
            raise ValueError("min_conf must be between 0 and 1")
        self.transaction_data = transaction_data
        self.information = self._get_information(self.transaction_data)
        self.transaction_number = self.information[0]
        self.items_length = self.information[1]
        self.items = self.information[2]
        self.items_dict = self.information[3]
        self.transaction_matrix = self.information[4]
        self.items_qubit_number = self._get_qubit_number(self.items_length - 1)
        self.transaction_qubit_number = self._get_qubit_number(self.transaction_number - 1)
        self.index_qubit_number = self.items_qubit_number + self.transaction_qubit_number
        self.digit_qubit_number = self._get_qubit_number(max(self.items))
        self.number_qubits = 2 * self.index_qubit_number + self.digit_qubit_number + 2
        self.min_support = min_support
        self.min_conf = min_conf


    def _get_information(self, transaction_data):
        transaction_number = len(transaction_data)
        items_set = set(chain.from_iterable(transaction_data))
        items = list(items_set)
        items.sort()
        items_length = len(items)
        items_dict = {i + 1: items[i] for i in range(items_length)}
        items = tuple([i + 1 for i in range(items_length)])
        transaction_matrix = []
        for data in transaction_data:
            temp = list(items)
            for i, key in enumerate(temp):
                if items_dict[key] not in data:
                    temp[i] = 0
            transaction_matrix.append(temp)

        return transaction_number, items_length, items, items_dict, transaction_matrix

    def _get_qubit_number(self, number):
        try:
            num_qubit = math.floor(math.log(number, 2) + 1)
        except:
            raise ValueError('The data in the file is out of format, please change the data and then try again!')
        return num_qubit

    def _create_c1(self, data_set):
        c1 = []
        for transaction in data_set:
            for item in transaction:
                if not [item] in c1 and item != 0:
                    c1.append([item])
        return list(map(tuple, c1))

    def _get_number_circuit(self, qlist, position, number, qubit_number):
        cir = QCircuit()
        bin_str = bin(number).replace('0b', '')
        bin_str = list((qubit_number - len(bin_str)) * '0' + bin_str)[::-1]
        for j, bit in enumerate(bin_str):
            if bit == '1':
                cir << X(qlist[position + j])
        return cir

    def _encode_circuit(self, qlist, position, index_qubit_number, items_length, transaction_number):
        cir = QCircuit()

        control_qubit = []
        for i in range(index_qubit_number):
            control_qubit.append(qlist[position + i])

        information_position = position + index_qubit_number
        for n in range(transaction_number):
            t_x_cir = self._get_number_circuit(qlist, position + self.items_qubit_number,
                                              2 ** self.transaction_qubit_number - 1 - n, self.transaction_qubit_number)
            for m in range(items_length):
                item_x_cir = self._get_number_circuit(qlist, position, 2 ** self.items_qubit_number - 1 - m,
                                                     self.items_qubit_number)
                cir << item_x_cir
                cir << t_x_cir
                sub_cir = self._get_number_circuit(qlist, information_position, self.transaction_matrix[n][m],
                                                  self.digit_qubit_number)
                cir << sub_cir.control(control_qubit)
                cir << t_x_cir
                cir << item_x_cir
        return cir

    def _query_circuit(self, qlist, position, target_number):
        cir = QCircuit()
        sub_cir = self._get_number_circuit(qlist, position, 2 ** self.digit_qubit_number - target_number - 1,
                                          self.digit_qubit_number)
        cir << sub_cir
        control_qubit = []
        for i in range(self.digit_qubit_number):
            control_qubit.append(qlist[position + i])
        cir << X(qlist[position + self.digit_qubit_number]).control(control_qubit)
        cir << sub_cir
        return cir

    def _transfer_to_phase(self, qlist, position):
        cir = QCircuit()
        cir << U1(qlist[position + 1], math.pi).control([qlist[position]])
        
        return cir

    def _oracle_cir(self, qlist, position, locating_number):
        u_cir = self._encode_circuit(qlist, position, self.index_qubit_number,
                                    self.items_length, self.transaction_number)
        s_cir = self._query_circuit(qlist, position + self.index_qubit_number, locating_number)
        transfer_cir = self._transfer_to_phase(qlist, position + self.index_qubit_number + self.digit_qubit_number)

        cir = QCircuit()
        cir << u_cir
        cir << s_cir
        cir << transfer_cir
        cir << s_cir.dagger()
        cir << u_cir.dagger()
        return cir

    def _coin_cir(self, qlist, position):
        u1_position = position + 1 + 2 * self.index_qubit_number + self.digit_qubit_number
        swap_interval = self.index_qubit_number
        coin_position = position
        cir = QCircuit()
        control_qubit = []
        for i in range(self.index_qubit_number):
            cir << H(qlist[coin_position + i])
            cir << X(qlist[coin_position + i])
            control_qubit.append(qlist[coin_position + i])
        control_cir = QCircuit()
        control_cir << U1(qlist[u1_position], math.pi)
        cir << control_cir.control(control_qubit)
        for i in range(self.index_qubit_number):
            cir << X(qlist[coin_position + i])
            cir << H(qlist[coin_position + i])
            cir << SWAP(qlist[coin_position + i], qlist[coin_position + i + swap_interval])
        return cir

    def _gk_cir(self, qlist, position, locating_number):
        cir = QCircuit()
        _oracle_cir = self._oracle_cir(qlist, position + self.index_qubit_number, locating_number)
        _coin_cir = self._coin_cir(qlist, position)
        cir << _oracle_cir
        cir << _coin_cir
        return cir

    def _search_prog(self, qlist, locating_number, show, file_name):
        """Build the search circuit for one locating number (no execution)."""
        _iter_number = self._iter_number()
        prog = QProg()
        for i in range(self.index_qubit_number):
            prog << H(qlist[self.index_qubit_number + i])
        prog << X(qlist[2 * self.index_qubit_number + self.digit_qubit_number])
        prog << H(qlist[2 * self.index_qubit_number + self.digit_qubit_number])
        prog << X(qlist[2 * self.index_qubit_number + self.digit_qubit_number + 1])

        cir = self._gk_cir(qlist, 0, locating_number)
        for n in range(_iter_number):
            prog << cir

        if locating_number == 1:
            line_length = 100
            if show is not None:
                if show == 'Picture':
                    if file_name:
                        abs_file_path = os.path.abspath(file_name)
                        f_path, f_name = os.path.split(abs_file_path)
                        if not os.path.exists(f_path):
                            os.makedirs(f_path)
                        pic_str = draw_qprog(prog, abs_file_path, line_length=line_length)
                        print(pic_str)
                    else:
                        pic_str = draw_qprog(prog, line_length=line_length)
                        print(pic_str)
                elif show == 'OriginIR':
                    ir = convert_qprog_to_originir(prog)
                    if file_name:
                        abs_file_path = os.path.abspath(file_name)
                        f_path, f_name = os.path.split(abs_file_path)
                        if not os.path.exists(f_path):
                            os.makedirs(f_path)
                        print(ir)
                        with open(abs_file_path, 'w') as f:
                            f.write(ir)
                    else:
                        with open('OriginIR.txt', 'w') as f:
                            f.write(ir)
                else:
                    raise NameError('Method is not recognized! Choose one method from \'Picture\',\'OriginIR\'.')
            else:
                if file_name:
                    abs_file_path = os.path.abspath(file_name)
                    f_path, f_name = os.path.split(abs_file_path)
                    if not os.path.exists(f_path):
                        os.makedirs(f_path)
                    draw_qprog(prog, abs_file_path, line_length=line_length)
        return prog

    def _iter_number(self):
        estimate_count = math.floor(math.pi * math.sqrt(2 ** self.index_qubit_number) / 2)
        if estimate_count % 2:
            count = estimate_count
        else:
            count = estimate_count + 1
        if count >= 9:
            count -= 4
        return count

    def _rows_from_counts(self, counts):
        """Extract the transaction rows for a locating number from counts.

        The legacy readout measured the first ``index_qubit_number`` qubits
        (qubit 0 = LSB), so the index value of a full-width measurement key
        is its projection ``int(key, 2) & (2 ** index_qubit_number - 1)``.
        Index values within one shot of the empirical maximum are selected,
        which reproduces the legacy 4-decimal argmax tie sets exactly for
        the deterministic fixtures.
        """
        total = sum(counts.values())
        projected = {}
        for key, count in counts.items():
            index_value = int(key, 2) & ((1 << self.index_qubit_number) - 1)
            projected[index_value] = projected.get(index_value, 0) + count / total
        max_val = max(projected.values())
        # The ``>= max_val - 1/total`` tolerance is a deliberate robustness
        # choice: empirical probabilities within one shot of the maximum
        # still count as ties, where the legacy readout compared exact
        # 4-decimal probabilities.  For the equal-count fixtures the two
        # select identical tie sets.
        epsilon = 1.0 / total
        index_list = [idx for idx, val in projected.items() if val >= max_val - epsilon]
        return self._get_index(index_list)

    def _get_index(self, index):
        result = []
        for idx in index:
            bin_str = bin(idx).replace('0b', '')
            bin_str = (self.index_qubit_number - len(bin_str)) * '0' + bin_str
            transaction_index = int(bin_str[:self.transaction_qubit_number], 2)
            item_index = int(bin_str[self.transaction_qubit_number:], 2)
            result.append((transaction_index, item_index))
        return result

    def _find_f1(self, ck_dict):
        f1_dict = {}
        f1 = []
        for key, val in ck_dict.items():
            support = len(val) / self.transaction_number
            if support >= self.min_support:
                # ``ck_dict`` keys are JSON-safe strings (the locating
                # number); convert back to an int and re-wrap each into a
                # 1-tuple so the apriori joins below keep their frozenset
                # semantics unchanged.
                key = (int(key),)
                f1_dict[key] = [val, support]
                f1.append(key)
        return f1, f1_dict

    def _find_fk(self, k, fk, fk_dict):
        cn = []
        fn_dict = {}
        fn = []
        len_fk = len(fk)
        for i in range(len_fk):
            for j in range(i + 1, len_fk):
                L1 = list(fk[i])[:k - 2]
                L2 = list(fk[j])[:k - 2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    c = tuple(frozenset(fk[i]) | frozenset(fk[j]))
                    cn.append(c)
                    set1_index = frozenset(fk_dict[fk[i]][0])
                    set2_index = frozenset(fk_dict[fk[j]][0])
                    set_index = set1_index & set2_index
                    index_list = list(set_index)
                    support = len(index_list) / self.transaction_number
                    if support >= self.min_support:
                        fn_dict[c] = [index_list, support]
                        fn.append(c)
        return fn, fn_dict

    def _fk_result(self, ck_dict):
        f1, f1_dict = self._find_f1(ck_dict)

        fn = []
        fn_dict = {}
        k = 2
        fk = f1
        fk_dict = f1_dict
        while fk:
            fn.append(fk)
            fn_dict.update(fk_dict)
            _fk_result = self._find_fk(k, fk, fk_dict)
            fk = _fk_result[0]
            fk_dict = _fk_result[1]
            k += 1
        return fn, fn_dict

    def _conf_x_y(self, supp_xy, supp_x):
        return round(supp_xy / supp_x, 2)

    def _get_all_conf(self, fn, fn_dict):
        len_fn = len(fn)
        if len_fn < 2:
            return None
        conf_dict = {}
        for i in range(1, len_fn):
            for backward in fn[i]:
                set_backward = frozenset(backward)
                for forward in fn[i - 1]:
                    set_forward = frozenset(forward)
                    if set_forward.issubset(set_backward):
                        supp_xy = fn_dict[backward][1]
                        supp_x = fn_dict[forward][1]
                        conf = self._conf_x_y(supp_xy, supp_x)
                        if conf >= self.min_conf:
                            cause = set_forward
                            effect = set_backward - set_forward
                            key = self._get_conf_key(cause, effect)
                            conf_dict[key] = conf
        return conf_dict

    def _get_conf_key(self, cause, effect):
        cause_list = list(cause)
        effect_list = list(effect)
        cause_str = ','.join([self.items_dict[i] for i in cause_list])
        effect_str = ','.join([self.items_dict[i] for i in effect_list])
        key = cause_str + '->' + effect_str
        return key

    def run(self, show=None, file_name="", machine_type="CPU", *, backend=None, execution_options=None):
        """
        Run the quantum association rule mining algorithm.

        Parameters
            show: ``string``
                Enumeration of the circuit show type, should be one of "None",
                "Picture" and "OriginIR".
            file_name: ``string``
                The output file name that records the circuit information.
            machine_type: ``string``
                Backend mode, should be "CPU".  ``machine_type='QCloud'`` is
                deprecated: it warns and requires an explicit ``backend``.
            backend : execution backend, keyword-only
                The sampling backend driving the search.  Defaults to the
                local simulator.
            execution_options : ``ExecutionOptions``, keyword-only
                Sampling options (shots, timeout, ...).  Defaults to 1000
                shots per search circuit.

        Returns
            conf_result : ``dict``
                confidence result
        """
        if machine_type == 'QCloud':
            warnings.warn(
                "machine_type='QCloud' is deprecated and will be removed in a "
                "future release; pass an explicit backend= instead of an api_key.",
                DeprecationWarning, stacklevel=2)
            if backend is None:
                raise AlgorithmInputError(
                    "machine_type='QCloud' requires an explicit backend= "
                    "argument; provide a logged-in execution backend instead "
                    "of an api_key.")
        elif machine_type != 'CPU':
            raise TypeError('No such mode! Choose one mode from \'CPU\',\'QCloud\'.')
        task = self.submit(show=show, file_name=file_name, backend=backend,
                           execution_options=execution_options)
        return task.result()

    def submit(self, show=None, file_name="", *, backend=None, execution_options=None):
        """Run the association rule mining as a resumable sampling task.

        ``submit()`` returns before any search circuit is sampled; every
        ``poll()`` samples the search circuit of one locating number
        (candidate 1-itemset) and records the sampled transaction rows.  Once
        all locating numbers are sampled the frequent sets and confidence
        rules are finalized classically.  Checkpointing preserves the round,
        and resuming continues from there without re-submitting completed
        rounds.

        The ``backend`` and ``execution_options`` parameters are keyword-only.
        """
        self._backend = resolve_backend(backend)
        options = execution_options if execution_options is not None else ExecutionOptions()
        prog = QProg(self.number_qubits)
        qlist = prog.qubits()
        c1 = self._create_c1(self.transaction_matrix)
        state = {"round": 0, "c1": c1, "ck_dict": {}}

        def make_advance(exec_backend):
            def advance(state):
                locating_number = state["c1"][state["round"]][0]
                search_prog = self._search_prog(qlist, locating_number, show, file_name)
                sample_task = exec_backend.submit_sample(search_prog, options=options)
                counts = sample_task.result().single_counts()
                rows = self._rows_from_counts(counts)
                # JSON checkpoint keys are always strings, so ``ck_dict``
                # keys are stored as strings end-to-end (fresh and resumed)
                # to survive the checkpoint round-trip losslessly.
                state["ck_dict"][str(locating_number)] = [row[0] for row in rows]
                state["round"] += 1
                if state["round"] < len(state["c1"]):
                    return CompletedBackendTask(None, task_id=sample_task.id), False
                fn, fn_dict = self._fk_result(state["ck_dict"])
                conf_result = self._get_all_conf(fn, fn_dict)
                if not conf_result:
                    raise ValueError("""The data in the file could not mine any rules to satisfy
the conditions of supporting greater than the min_support {}
and confidence greater than the min_conf {}!
You can change the data in the file or decrease the min_support and the min_conf,
then try again!""".format(self.min_support, self.min_conf))
                return CompletedBackendTask(conf_result, task_id=sample_task.id), True
            return advance

        register_algorithm("qarm", make_advance)
        return AlgorithmTask(algorithm="qarm", initial_state=state,
                             advance=make_advance(self._backend), backend=self._backend)
