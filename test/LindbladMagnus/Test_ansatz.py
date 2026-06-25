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

"""Unit tests for the variational ansatz module."""

import numpy as np
import pytest
from pyqpanda3.core import CNOT, H, RX, RY, RZ

from pyqpanda_alg.LindbladMagnus.ansatz import (HardwareEfficientAnsatz,
                                                VariationalAnsatz)


def test_ansatz_parameter_count():
    """Adding parameterised gates must consume parameters in order."""
    ansatz = VariationalAnsatz(n_qubits=2)
    assert ansatz.n_parameters == 0
    ansatz.add_gate(RX, 0)
    ansatz.add_gate(RZ, 0)
    ansatz.add_gate(CNOT, (0, 1))
    ansatz.add_gate(RY, 1)
    assert ansatz.n_parameters == 3


def test_ansatz_statevector_bell():
    """The ansatz must reproduce a Bell state for the right theta."""
    ansatz = VariationalAnsatz(n_qubits=2)
    ansatz.add_gate(H, 0)
    ansatz.add_gate(CNOT, (0, 1))
    sv = ansatz.get_statevector(np.zeros(0))
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(sv, expected, atol=1e-10)


def test_ansatz_init_state_prepared():
    """The reference state must be loaded at theta = 0."""
    init = np.array([0, 0, 0, 1], dtype=complex)  # |11>
    ansatz = VariationalAnsatz(n_qubits=2, init_state=init)
    ansatz.add_gate(RX, 0)
    ansatz.add_gate(RZ, 1)
    sv = ansatz.get_statevector(np.zeros(2))
    np.testing.assert_allclose(sv, init, atol=1e-10)


def test_hardware_efficient_ansatz_identity_at_zero():
    """HE ansatz with parameterised RZZ entanglers must reduce to identity
    when ``theta = 0`` so that the initial state is preserved."""
    init = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)  # |001>
    ansatz = HardwareEfficientAnsatz(n_qubits=3, layers=2, init_state=init)
    sv = ansatz.get_statevector(np.zeros(ansatz.n_parameters))
    np.testing.assert_allclose(sv, init, atol=1e-10)


def test_jacobian_parameter_shift_factor():
    """The Jacobian computed by parameter shift must match the analytic
    derivative of ``R_P(theta)|0>`` at a non-trivial ``theta``."""
    ansatz = VariationalAnsatz(n_qubits=1)
    ansatz.add_gate(RX, 0)
    theta = np.array([0.3])
    jac = ansatz.get_jacobian(theta)
    # d/dtheta RX(theta)|0> = (-i/2) X RX(theta)|0>
    psi = ansatz.get_statevector(theta)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = (-1j / 2.0) * (X @ psi)
    np.testing.assert_allclose(jac[:, 0], expected, atol=1e-10)


def test_ansatz_validates_qubits():
    """``n_qubits`` must be positive."""
    with pytest.raises(ValueError):
        VariationalAnsatz(n_qubits=0)
