"""Restored from ``test/legacy_disabled/QAlgBase/QUBO_cir.py``.

Pure circuit-construction assertion of ``QuadraticBinary.cir``; the unused
``CPUQVM`` from the legacy test is dropped.
"""

import pytest
import sympy as sp

from pyqpanda3.core import QProg, QCircuit

from pyqpanda_alg.QUBO import QUBO


class Test_QUBO_cir:

    def test_cir_basic_functionality(self):
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        function = (
            -0.5 * x0 * x1 - 0.7 * x0 * x1
            + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
        )
        test0 = QUBO.QuadraticBinary(function)
        n_key, n_res = test0.query_qnumber()
        q_key = QProg(n_key + n_res).qubits()
        circuit = test0.cir(q_key[:n_key], q_key[n_key:])

        assert circuit is not None, "cir should return a non-None quantum circuit"
        assert isinstance(circuit, QCircuit), \
            f"should return QCircuit, got {type(circuit)}"
        assert len(circuit.qubits()) == n_key + n_res, \
            f"circuit should use {n_key + n_res} qubits, " \
            f"got {len(circuit.qubits())}"
