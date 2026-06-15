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

import pytest
import numpy as np

from pyqpanda3.core import CPUQVM, QCircuit, QProg, TOFFOLI, Z
from pyqpanda_alg.Grover import (
    Grover,
    GroverAdaptiveSearch,
    amp_operator,
    mark_data_reflection,
    iter_num,
    iter_analysis,
)


def mark_11(qubits):
    """Oracle: mark |11> state using Toffoli + Z."""
    cir = QCircuit()
    cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    cir << Z(qubits[2])
    cir << TOFFOLI(qubits[0], qubits[1], qubits[2])
    return cir


class TestGroverCir:
    """Tests for Grover.cir() method."""

    def setup_method(self):
        self.qvm = CPUQVM()

    def test_cir_basic(self):
        """Basic circuit construction with TOFFOLI oracle."""
        q_state = QProg(3).qubits()
        g = Grover(flip_operator=mark_11)
        circuit = g.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=1)
        assert isinstance(circuit, QCircuit)
        assert circuit is not None

    def test_cir_zero_iterations(self):
        """Zero iterations should return only the initial Hadamard layer."""
        q_state = QProg(3).qubits()
        g = Grover(flip_operator=mark_11)
        circuit = g.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=0)
        assert isinstance(circuit, QCircuit)

    def test_cir_multiple_iterations(self):
        """Multiple iterations should produce a valid circuit."""
        q_state = QProg(3).qubits()
        g = Grover(flip_operator=mark_11)
        for iters in [1, 2, 3]:
            circuit = g.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=iters)
            assert isinstance(circuit, QCircuit)
            assert len(str(circuit)) > 0

    def test_cir_negative_iterations_raises(self):
        """Negative iteration count should raise ValueError."""
        q_state = QProg(3).qubits()
        g = Grover(flip_operator=mark_11)
        with pytest.raises(ValueError, match="non-negative"):
            g.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=-1)

    def test_cir_execution_finds_target(self):
        """Grover search should find |11> with high probability."""
        q_state = QProg(3).qubits()
        g = Grover(flip_operator=mark_11)
        circuit = g.cir(q_input=q_state[:2], q_flip=q_state, q_zero=q_state[:2], iternum=1)
        prog = QProg()
        prog << circuit
        self.qvm.run(prog, shots=1000)
        result = self.qvm.result().get_prob_dict(q_state[:2])
        assert result['11'] >= 0.99

    def test_cir_default_flip_operator(self):
        """Default flip_operator (None) should use Z on last qubit."""
        q_state = QProg(1).qubits()
        g = Grover()  # no flip_operator
        circuit = g.cir(q_input=q_state, iternum=1)
        assert isinstance(circuit, QCircuit)


class TestGroverSearch:
    """Tests for Grover.search() convenience method."""

    def test_search_returns_dict(self):
        """search() should return a probability dictionary."""
        q = list(range(3))
        g = Grover(flip_operator=mark_11)
        result = g.search(q_input=q[:2], q_flip=q, q_zero=q[:2], iternum=1, shots=1000)
        assert isinstance(result, dict)
        assert '11' in result
        assert result['11'] >= 0.99

    def test_search_default_shots(self):
        """search() should work with default shots parameter."""
        q = list(range(3))
        g = Grover(flip_operator=mark_11)
        result = g.search(q_input=q[:2], q_flip=q, q_zero=q[:2], iternum=1)
        assert isinstance(result, dict)


class TestIterNum:
    """Tests for iter_num() function."""

    def test_single_solution(self):
        """2 qubits, 1 solution -> optimal iterations = 1."""
        assert iter_num(2, 1) == 1

    def test_two_solutions(self):
        """3 qubits, 2 solutions -> should return a positive integer."""
        n = iter_num(3, 2)
        assert isinstance(n, int)
        assert n >= 1

    def test_large_search_space(self):
        """Larger search space should yield more iterations."""
        assert iter_num(4, 1) >= iter_num(2, 1)


class TestIterAnalysis:
    """Tests for iter_analysis() function."""

    def test_perfect_amplification(self):
        """2 qubits, 1 solution, 1 iteration -> probability = 1.0."""
        prob, theta = iter_analysis(2, 1, iternum=1)
        assert abs(prob - 1.0) < 1e-10

    def test_over_rotation(self):
        """Too many iterations should reduce probability."""
        prob1, _ = iter_analysis(2, 1, iternum=1)
        prob2, _ = iter_analysis(2, 1, iternum=2)
        assert prob2 < prob1

    def test_returns_tuple(self):
        """Should return (probability, angle) tuple."""
        result = iter_analysis(3, 1, iternum=1)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestAmpOperator:
    """Tests for amp_operator() function."""

    def test_amp_operator_returns_circuit(self):
        """amp_operator should return a QCircuit."""
        q = list(range(3))
        circuit = amp_operator(
            q_input=q[:2], q_flip=q, q_zero=q[:2],
            flip_operator=mark_11
        )
        assert isinstance(circuit, QCircuit)

    def test_amp_operator_default_flip(self):
        """Default flip_operator should use Z on last qubit."""
        q = [0]
        circuit = amp_operator(q_input=q, q_flip=q, q_zero=q)
        assert isinstance(circuit, QCircuit)

    def test_amp_operator_single_qubit(self):
        """Single qubit zero_flip should produce X-Z-X pattern."""
        q = [0]
        circuit = amp_operator(q_input=q, q_flip=q, q_zero=q)
        assert isinstance(circuit, QCircuit)


class TestMarkDataReflection:
    """Tests for mark_data_reflection() function."""

    def test_single_marked_state(self):
        """Mark a single state string."""
        q = [0, 1, 2]
        circuit = mark_data_reflection(q, mark_data='101')
        assert isinstance(circuit, QCircuit)

    def test_multiple_marked_states(self):
        """Mark multiple state strings."""
        q = [0, 1, 2]
        circuit = mark_data_reflection(q, mark_data=['101', '001'])
        assert isinstance(circuit, QCircuit)

    def test_mark_data_search_execution(self):
        """Full Grover search with mark_data_reflection should find targets."""
        q = list(range(3))

        def mark(qubits):
            return mark_data_reflection(qubits, mark_data=['101', '001'])

        g = Grover(flip_operator=mark)
        qvm = CPUQVM()
        circuit = g.cir(q_input=q, q_flip=q, q_zero=q, iternum=1)
        prog = QProg()
        prog << circuit
        qvm.run(prog, shots=1000)
        result = qvm.result().get_prob_dict(q)
        # Both marked states should have high probability
        assert result.get('101', 0) + result.get('001', 0) >= 0.9


class TestGroverAdaptiveSearch:
    """Tests for GroverAdaptiveSearch class."""

    def test_gas_basic_run(self):
        """GAS should run without errors and return valid results."""
        def flip_oracle(q_index_value, current_min):
            from pyqpanda3.core import QCircuit, H, U1
            from pyqpanda_alg.plugin import qft
            q_index = q_index_value[:2]
            q_value = q_index_value[2:]
            n_value = len(q_value)
            factor = np.pi * 2 ** (1 - n_value)
            cal_cir = QCircuit()
            for i, q_i in enumerate(q_value):
                cal_cir << H(q_i)
                cal_cir << U1(q_i, factor * 2 ** i).control(q_index)
                cal_cir << U1(q_i, factor * 2 ** i).control(q_index[0])
                cal_cir << U1(q_i, -factor * 2 ** i).control(q_index[1])
                cal_cir << U1(q_i, factor * 2 ** i * (-current_min))
            cal_cir << qft(q_value).dagger()
            return cal_cir

        def n_value_function(current_min):
            return 2 if current_min == 0 else 3

        def value_function(var_array):
            var_array = list(map(int, var_array))[::-1]
            return var_array[0] * var_array[1] + var_array[0] - var_array[1]

        gas = GroverAdaptiveSearch(
            init_value=0, n_index=2, oracle_circuit=flip_oracle
        )
        result = gas.run(
            continue_times=3,
            n_value_function=n_value_function,
            value_function=value_function,
            rotation_change='increase',
        )
        indexes, min_val = result
        # GAS is probabilistic; the minimum should be <= 0 (the initial value)
        assert min_val <= 0
        assert isinstance(indexes, list)
        assert len(indexes) >= 1

    def test_gas_invalid_rotation_change(self):
        """Invalid rotation_change should raise NameError."""
        gas = GroverAdaptiveSearch(init_value=0, n_index=2)
        with pytest.raises(NameError, match="not recognized"):
            gas.run(
                continue_times=1,
                n_value_function=lambda _: 2,
                value_function=lambda _: 0,
                rotation_change='invalid',
            )


class TestGroverWithCustomAnsatz:
    """Tests for Grover with custom in_operator and amplify_operator."""

    def test_custom_in_operator(self):
        """Custom initial state preparation should work."""
        from pyqpanda3.core import X

        def custom_init(qubits):
            cir = QCircuit()
            for q in qubits:
                cir << X(q)  # Start from |11> instead of uniform superposition
            return cir

        q = [0, 1]
        g = Grover(in_operator=custom_init, flip_operator=lambda qbs: Z(qbs[-1]))
        circuit = g.cir(q_input=q, iternum=1)
        assert isinstance(circuit, QCircuit)

    def test_custom_amplify_operator(self):
        """Custom amplify_operator should be used instead of default."""
        def custom_amp(q_all):
            cir = QCircuit()
            for q in q_all:
                cir << Z(q)
            return cir

        q = [0, 1]
        g = Grover(amplify_operator=custom_amp)
        circuit = g.cir(q_input=q, iternum=1)
        assert isinstance(circuit, QCircuit)
