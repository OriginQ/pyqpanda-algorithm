"""Restored from ``test/legacy_disabled/QAOA/qaoa_pauli_z_operator_to_circuit.py``.

Pure circuit-construction assertion; the unused ``CPUQVM`` from the legacy
test is dropped (the public behavior asserted here is the returned circuit).
"""

import pytest
import sympy as sp

from pyqpanda3.core import QProg

from pyqpanda_alg.QAOA import qaoa


class TestPauliZOperatorToCircuit:

    def setup_method(self):
        self.prog = QProg(3)
        self.qubits = self.prog.qubits()

    def test_basic_functionality(self):
        # f = 2*x0*x1 + 3*x2 - 1
        vars = sp.symbols('x0:3')
        f = 2 * vars[0] * vars[1] + 3 * vars[2] - 1

        operator = qaoa.problem_to_z_operator(f)

        gamma = 1.0
        circuit, _ = qaoa.pauli_z_operator_to_circuit(
            operator, self.qubits, gamma
        )

        assert circuit is not None
        originir_str = circuit.originir()
        assert isinstance(originir_str, str)
        assert len(originir_str) > 0
