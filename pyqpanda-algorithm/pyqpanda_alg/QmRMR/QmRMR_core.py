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

import matplotlib.pyplot as plt
import numpy as np
from pyqpanda3.core import QCircuit, QProg, CNOT, X, RY
from pyqpanda3.hamiltonian import Hamiltonian, PauliOperator

from .. plugin import *
from ..execution import AlgorithmTask, CompletedBackendTask, ExecutionOptions, resolve_backend
from ..QAOA.qaoa import p_1

def plot_bar(dic):
    """
    Plot a bar chart from a dictionary.

    Highlights the bar with the highest value in red, while the rest are shown in blue.

    Parameters
    ----------
    dic : dict
        A dictionary where keys are labels and values are numerical quantities to be plotted.
    """
    keys = list(dic.keys())
    values = list(dic.values())
    max_value_index = values.index(max(values))
    colors = ['red' if i == max_value_index else 'blue' for i in range(len(keys))]
    plt.bar(keys, values, color=colors)
    plt.show()
    plt.clf()


def plot_loss(los):
    """
    Plot a line chart showing the loss over iterations.

    Useful for visualizing the optimization process.

    Parameters
    ----------
    los : list of float
        A list of loss values, where each entry corresponds to one optimization step.
    """
    index = list(range(1, 1 + len(los)))
    plt.plot(index, los)
    plt.show()
    plt.clf()

def control(qubits1, qubits2, theta):

    """
    Construct a multi-controlled RY rotation gate.

    Creates an RY rotation gate on the target qubit with multiple control qubits.

    Parameters
    ----------
    qubits1 : list or Qubit
        Control qubit(s). Can be a single Qubit or a list of Qubits.
    qubits2 : Qubit
        Target qubit on which the RY rotation will be applied.
    theta : float
        Rotation angle for the RY gate.

    Returns
    -------
    QGate
        A controlled RY gate that can be appended to a quantum circuit.
    """
    qvec = qubits1
    RY_control = RY(qubits2, theta).control(qvec)
    return RY_control

class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) Optimizer.

    This class implements the SPSA algorithm for optimizing a scalar-valued
    objective function that depends on multiple parameters. SPSA is useful
    in cases where the objective function is noisy, non-differentiable, or
    expensive to evaluate, such as in variational quantum algorithms.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to be minimized. It must take a parameter vector as input
        and return a scalar loss value.
    a : float, optional
        Scaling factor for the learning rate (step size), default is 0.1.
    c : float, optional
        Scaling factor for the perturbation, default is 0.1.
    alpha : float, optional
        Exponent controlling the decay rate of the learning rate, default is 0.602.
    beta : float, optional
        Exponent controlling the decay rate of the perturbation size, default is 0.101.
    max_iters : int, optional
        Maximum number of iterations to perform, default is 1000.
    """

    def __init__(self, objective_function, a=0.1, c=0.1, alpha=0.602, beta=0.101, max_iters=1000):
        self.objective_function = objective_function
        self.a = a
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.max_iters = max_iters

    def optimize(self, initial_params):
        """
        Run the SPSA optimization to minimize the objective function.

        Parameters
        ----------
        initial_params : np.ndarray
            Initial values for the parameters to be optimized.

        Returns
        -------
        np.ndarray
            The optimized parameter vector.
        list[float]
            History of objective function values over iterations.
        """
        m = len(initial_params)
        history = []
        params = initial_params
        history.append(float(self.objective_function(initial_params.copy())))
        for k in range(1, self.max_iters + 1):
            ak = self.a / (k + 1) ** self.alpha
            ck = self.c / (k + 1) ** self.beta

            delta = np.random.choice([-1, 1], size=m)

            params_plus = params + ck * delta
            params_minus = params - ck * delta

            f_plus = self.objective_function(params_plus)
            f_minus = self.objective_function(params_minus)
            grad_estimate = (f_plus - f_minus) / (2 * ck * delta)

            params = params - ak * grad_estimate
            x = self.objective_function(params.copy())
            history.append(float(x))
        return params, history



class Feature_Selection:
    """
    Quantum-based Feature Selection using parameterized quantum circuits.

    This class builds a variational quantum circuit to model a feature selection
    task. The objective is to optimize the selection of a subset of features
    based on a given linear and quadratic objective function.

    Parameters
    ----------
    quadratic : np.ndarray
        Quadratic coefficient matrix used in the objective function.
    linear : list[float]
        Linear coefficient list used in the objective function.
    select_num : int
        Number of features to be selected (i.e., number of bits set to 1).

    Attributes
    ----------
    qb_num : int
        Number of qubits required, equal to the number of features.
    l1 : list[float]
        History of the total expectation value (the full objective
        E[x^T Q x] - E[x . l]) during optimization.
    l2 : list[float]
        History of the total expectation value (the full objective
        E[x^T Q x] - E[x . l]) during optimization.
        
    >>> from pyqpanda_alg import QmRMR
    >>> import numpy as np
    >>> from pyqpanda3.core import *
    >>> import matplotlib.pyplot as plt
    
    >>> m = 6
    >>> u = np.random.random(m)
    >>> cor = np.random.random([m, m])
    >>> cor = (cor.T + cor)/2
    >>> ini_par = np.random.random(int(m/2)*m)*np.pi
    >>> loss, choice, dic = QmRMR.Feature_Selection(cor, u, 3).get_his_res(ini_par)
    >>> print(choice)
    >>> plt.plot(loss)
    >>> plt.show()
    [0, 1, 0, 0, 1, 1]
    
    """

    def __init__(self, quadratic, linear, select_num):
        self.quadratic = quadratic
        self.linear = linear
        self.qb_num = len(linear)
        self.select_num = select_num
        self.l1 = []
        self.l2 = []
        self._backend = None
        self._execution_options = None
        linear_arr = np.asarray(linear, dtype=float)
        quadratic_arr = np.asarray(quadratic, dtype=float) if quadratic is not None else None
        # Objective: minimize E[x^T Q x] - E[x . l] (redundancy - relevance),
        # the standard QmRMR/mRMR form.  The linear coefficients are
        # relevance weights (higher = more preferred), so relevance
        # SUBTRACTS from the loss; hence the minus sign below.
        # Bit strings are most-significant first, so feature p maps to
        # qubit (m - 1 - p) in the observable.
        observable = 0 * PauliOperator({"": 1})
        for index in range(self.qb_num):
            coefficient = -linear_arr[index]
            if quadratic_arr is not None:
                coefficient += quadratic_arr[index][index]
            observable += coefficient * p_1(self.qb_num - 1 - index)
        if quadratic_arr is not None:
            for row in range(self.qb_num):
                for col in range(row + 1, self.qb_num):
                    coefficient = quadratic_arr[row][col] + quadratic_arr[col][row]
                    observable += coefficient * p_1(self.qb_num - 1 - row) * p_1(self.qb_num - 1 - col)
        self._observable = Hamiltonian(observable)

    def Circuit(self, qbs, para):
        cir = QCircuit()
        n = len(qbs)
        for i in range(self.select_num):
            cir << X(qbs[i])

        cz_list_even = [[t, t + 1] for t in range(0, n - 1, 2)]
        cz_list_odd = [[t + 1, t + 2] for t in range(0, n - 2, 2)]
        k = -1
        for i in range(int(n/2)):
            for kf in cz_list_even:
                k += 1
                cir << CNOT(qbs[kf[1]], qbs[kf[0]]) << control(qbs[kf[0]], qbs[kf[1]], para[k]) << CNOT(
                    qbs[kf[1]], qbs[kf[0]])
            for kf in cz_list_odd:
                k += 1
                cir << CNOT(qbs[kf[1]], qbs[kf[0]]) << control(qbs[kf[0]], qbs[kf[1]], para[k]) << CNOT(
                    qbs[kf[1]], qbs[kf[0]])
        return cir


    def select_key(self, dit):
        new_dit = {}
        for key in dit:
            if key.count('1') == self.select_num:
                new_dit[key] = dit[key]
        top_10_items = dict(sorted(new_dit.items(), key=lambda item: item[1], reverse=True)[:10])
        top_10_items = {key: round(value, 3) for key, value in top_10_items.items()}
        return next(iter(top_10_items)), top_10_items

    def get_theory(self, para):
        backend = self._backend if self._backend is not None else resolve_backend(None)
        options = self._execution_options if self._execution_options is not None else ExecutionOptions()
        prog = QProg(self.qb_num)
        qv = prog.qubits()
        prog << self.Circuit(qbs=qv, para=para)
        if backend.capabilities.statevector:
            statevector = backend.submit_statevector(
                prog, options=options
            ).result().single_statevector()
            res = {
                format(i, '0%db' % self.qb_num): abs(statevector[i]) ** 2
                for i in range(len(statevector))
            }
            return parse_quantum_result_dict(res, qv, select_max=-1)
        prog << measure_all(qv, qv)
        counts = backend.submit_sample(prog, options=options).result().single_counts()
        total = sum(counts.values())
        return {key: count / total for key, count in counts.items()}


    def cal_loss(self, para):
        backend = self._backend if self._backend is not None else resolve_backend(None)
        options = self._execution_options if self._execution_options is not None else ExecutionOptions()
        prog = QProg(self.qb_num)
        qv = prog.qubits()
        prog << self.Circuit(qbs=qv, para=para)
        res = backend.submit_estimate((prog, self._observable),
                                      options=options).result().single_value()

        self.l1.append(res)
        self.l2.append(res)

        return res


    def submit(self, ini_para, *, backend=None, execution_options=None):
        """
        Run the feature selection optimization and return an execution-layer task.

        The whole SPSA optimization runs inside a single advance step, so
        the task completes on the first ``poll()``; checkpoint/resume
        applies to the multi-round state machines (Grover, QAE, QARM),
        not here.  The synchronous :meth:`get_his_res` drives this task to
        completion; the backend and execution_options parameters are
        keyword-only.

        Parameters
            ini_para : ``array-like``
                Initial parameters of the variational circuit.
            backend : ``ExecutionBackend``, ``optional``
                The execution backend to run on. Keyword-only. If not given, the CPU
                LocalBackend is used.
            execution_options : ``ExecutionOptions``, ``optional``
                Options controlling the backend submissions. Keyword-only. If not
                given, defaults are used.

        Returns
            task : ``AlgorithmTask``
                Execution-layer task holding the optimization run.  The
                task is not checkpointable/resumable: the SPSA optimization
                runs in one advance step (checkpoint/resume applies to the
                multi-round state machines, Grover/QAE/QARM).
                ``task.result()`` returns ``(his, choice, dic)``.
        """
        self._backend = resolve_backend(backend)
        self._execution_options = execution_options

        def advance(state):
            np.random.seed(1234)
            optimizer = SPSAOptimizer(self.cal_loss, max_iters=200)
            optimal_params, his = optimizer.optimize(ini_para)
            x = self.get_theory(optimal_params)
            key, dic = self.select_key(x)
            choice = [int(t) for t in key]
            return CompletedBackendTask((his, choice, dic), task_id='qmrmr-result'), True

        return AlgorithmTask(algorithm='qmrmr', initial_state={}, advance=advance, backend=self._backend)


    def get_his_res(self, ini_para, *, backend=None, execution_options=None):
        """
        Run the feature selection optimization.

        This is the synchronous counterpart of :meth:`submit`: the whole
        optimization is executed immediately and the completed
        ``(his, choice, dic)`` tuple is returned. See :meth:`submit` for
        the parameter documentation.
        """
        task = self.submit(ini_para, backend=backend, execution_options=execution_options)
        return task.result()
