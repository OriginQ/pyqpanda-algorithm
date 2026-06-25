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

"""Tests for the new user-facing API: pickling, ``LindbladResult``, validation,
``__repr__`` and parallel execution."""

import pickle

import numpy as np
import pytest

from pyqpanda_alg.LindbladMagnus import (HardwareEfficientAnsatz,
                                         LindbladMagnusSolver, LindbladResult,
                                         solve, tfim_model)
from pyqpanda3.core import RX, RY, H


# ----------------------------------------------------------------------
#  Pickling of the ansatz
# ----------------------------------------------------------------------
def test_ansatz_pickle_roundtrip():
    """A pickled ansatz must reproduce the same state-vectors."""
    from pyqpanda_alg.LindbladMagnus.ansatz import VariationalAnsatz
    init = np.array([0, 0, 0, 1], dtype=complex)
    ansatz = VariationalAnsatz(n_qubits=2, init_state=init)
    ansatz.add_gate(RX, 0)
    ansatz.add_gate(RY, 1)

    blob = pickle.dumps(ansatz)
    rebuilt = pickle.loads(blob)

    theta = np.array([0.3, 0.7])
    sv_original = ansatz.get_statevector(theta)
    sv_rebuilt = rebuilt.get_statevector(theta)
    np.testing.assert_allclose(sv_original, sv_rebuilt, atol=1e-12)
    assert rebuilt.n_parameters == ansatz.n_parameters


def test_hardware_efficient_ansatz_pickle():
    """``HardwareEfficientAnsatz`` must pickle/unpickle across processes."""
    init = np.array([0, 0, 0, 1], dtype=complex)
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=init)
    rebuilt = pickle.loads(pickle.dumps(ansatz))
    theta = np.linspace(0, 1, ansatz.n_parameters)
    np.testing.assert_allclose(ansatz.get_statevector(theta),
                               rebuilt.get_statevector(theta), atol=1e-12)


# ----------------------------------------------------------------------
#  __repr__
# ----------------------------------------------------------------------
def test_ansatz_repr_informative():
    """``repr(ansatz)`` must report qubit/parameter counts."""
    ansatz = HardwareEfficientAnsatz(n_qubits=3, layers=2)
    text = repr(ansatz)
    assert "n_qubits=3" in text
    assert "n_parameters" in text


def test_solver_repr_informative():
    """``repr(solver)`` must report the key configuration knobs."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=2, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz, magnus_order=2,
                                  integrator="euler")
    text = repr(solver)
    assert "magnus_order=2" in text
    assert "integrator='euler'" in text


# ----------------------------------------------------------------------
#  LindbladResult
# ----------------------------------------------------------------------
def test_result_object_properties():
    """``LindbladResult`` must expose shape and metadata accessors."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz)
    times = np.linspace(0, 0.5, 6)
    result = solver.solve(psi0, times, e_ops, traj_num=3, seed=1)
    assert isinstance(result, LindbladResult)
    assert result.n_ops == len(e_ops)
    assert result.n_times == len(times)
    assert result.traj_num == 3
    assert result.expect.shape == (len(e_ops), len(times))
    assert result.std.shape == result.expect.shape
    assert result.norms.shape == (len(times),)
    assert "magnus_order" in result.solver_info
    # repr must not raise.
    assert isinstance(repr(result), str)


def test_solve_functional_api_returns_result():
    """The functional ``solve`` API must return a ``LindbladResult``."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    times = np.linspace(0, 0.5, 5)
    result = solve(H, c_ops, ansatz, psi0, times, e_ops,
                   traj_num=2, magnus_order=1, eps=1e-10)
    assert isinstance(result, LindbladResult)
    assert result.solver_info["eps"] == 1e-10


# ----------------------------------------------------------------------
#  Input validation
# ----------------------------------------------------------------------
def test_solver_rejects_non_increasing_tlist():
    """A non-increasing time grid must raise ``ValueError``."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz)
    with pytest.raises(ValueError):
        solver.solve(psi0, np.array([0.0, 1.0, 1.0, 2.0]), e_ops, traj_num=1)


def test_solver_rejects_short_tlist():
    """A time grid with fewer than two points must raise ``ValueError``."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz)
    with pytest.raises(ValueError):
        solver.solve(psi0, np.array([0.0]), e_ops, traj_num=1)


def test_solver_rejects_mismatched_hamiltonian_dimension():
    """The Hamiltonian dimension must match the ansatz qubit count."""
    from pyqpanda_alg.LindbladMagnus import HardwareEfficientAnsatz
    H_big = np.eye(8, dtype=complex)
    ansatz_small = HardwareEfficientAnsatz(n_qubits=2)
    with pytest.raises(ValueError):
        LindbladMagnusSolver(H_big, [], ansatz_small)


def test_solver_rejects_bad_c_ops_shape():
    """Collapse operators must have the same shape as ``H``."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    bad_ops = [np.eye(8, dtype=complex)]  # wrong size
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    with pytest.raises(ValueError):
        LindbladMagnusSolver(H, bad_ops, ansatz)


def test_solver_rejects_bad_init_params_length():
    """``init_params`` must match the ansatz parameter count."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    bad_params = np.zeros(ansatz.n_parameters + 5)
    with pytest.raises(ValueError):
        LindbladMagnusSolver(H, c_ops, ansatz, init_params=bad_params)


# ----------------------------------------------------------------------
#  Parallel execution
# ----------------------------------------------------------------------
def test_parallel_matches_serial():
    """Parallel and serial runs with the same seeds must agree."""
    H, c_ops, e_ops, psi0, _ = tfim_model()
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1, init_state=psi0)
    solver = LindbladMagnusSolver(H, c_ops, ansatz)
    times = np.linspace(0, 0.5, 5)
    serial = solver.solve(psi0, times, e_ops, traj_num=4, seed=7, n_jobs=1)
    parallel = solver.solve(psi0, times, e_ops, traj_num=4, seed=7, n_jobs=2)
    np.testing.assert_allclose(serial.expect, parallel.expect, atol=1e-12)


# ----------------------------------------------------------------------
#  to_qprog
# ----------------------------------------------------------------------
def test_ansatz_to_qprog_returns_qprog():
    """``to_qprog()`` must expose a concrete :class:`QProg`."""
    from pyqpanda3.core import QProg
    ansatz = HardwareEfficientAnsatz(n_qubits=2, layers=1)
    prog = ansatz.to_qprog()
    assert isinstance(prog, QProg)
    assert prog.qubits_num() == 2
