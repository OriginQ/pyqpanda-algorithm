"""VQE ground-state search on the local CPU backend, no runtime needed.

This example finds the ground state of a one-qubit Pauli-Z Hamiltonian
with the built-in hardware-efficient ansatz, entirely on pyqpanda3's
local CPU simulator: no qpanda3-runtime installation and no credentials
are required.  It walks through the stable public surface of
``pyqpanda_alg.VQE`` -- the immutable ``VQEConfig`` knobs, an explicit
``LocalBackend``, a custom ansatz following the callable protocol, a
custom optimizer, and checkpoint recovery through ``AlgorithmTask``.
The remote qpanda3-runtime path is shown at the end as commented code
that reads credentials from environment variables only.

Run it standalone from the repository root::

    python pyqpanda-algorithm/example/VQE/vqe_hamiltonian.py
"""

import os
import tempfile

import numpy as np
from pyqpanda3.core import CNOT, RY, RZ
from pyqpanda3.hamiltonian import Hamiltonian
from pyqpanda3.vqcircuit import VQCircuit

from pyqpanda_alg.VQE import VQE, VQEConfig
from pyqpanda_alg.execution import AlgorithmTask, LocalBackend

# -- 1. The observable ------------------------------------------------------
# VQE consumes a pyqpanda3 Hamiltonian directly; the qubit count is
# inferred from the Pauli operator.  The one-qubit Z observable has
# ground-state energy -1.0, reached by the ansatz at parameters (0, 0).
hamiltonian = Hamiltonian({"Z0": 1.0})

# -- 2. Default solver: built-in ansatz, SciPy optimizer, local CPU --------
# Omitting the backend argument means the local CPU backend.  The
# default ansatz for one qubit is the RY/RZ rotation pair; the parameter
# ordering is layer-major, qubit-major, RY before RZ per qubit.
solver = VQE(hamiltonian)
result = solver.run(
    initial_parameters=np.array([0.2, 0.0]),
    config=VQEConfig(max_iterations=80, tolerance=1e-6),
)
print(f"default solver: energy = {result.energy:.6f}, converged = {result.converged}")

# -- 3. Explicit LocalBackend and configuration knobs -----------------------
# backend=LocalBackend() is the explicit form of the default.  VQEConfig
# carries the iteration budget, the convergence tolerance, the classical
# optimizer method, and the per-evaluation sampling budget (shots).
# Energies are sampled estimates, so a converged run sits within shot
# noise of the exact value; more shots shrink the fluctuation.
explicit_solver = VQE(hamiltonian)
explicit_result = explicit_solver.run(
    initial_parameters=np.array([0.2, 0.0]),
    backend=LocalBackend(),
    config=VQEConfig(max_iterations=80, tolerance=1e-6, shots=1000),
)
print(f"explicit backend: energy = {explicit_result.energy:.6f}")

# -- 4. Custom ansatz: the callable contract --------------------------------
# Any ansatz exposing the callable contract works: ansatz(parameters)
# returns the bound circuit through .circuits()[0], and
# mutable_parameter_total() reports the parameter count.  A pyqpanda3
# VQCircuit built with set_Param fulfills both.
ansatz = VQCircuit(2)
ansatz.set_Param([4])
ansatz << RY(0, ansatz.Param([0])) << RZ(0, ansatz.Param([1]))
ansatz << RY(1, ansatz.Param([2])) << RZ(1, ansatz.Param([3]))
ansatz << CNOT(0, 1)

custom_ansatz_solver = VQE(Hamiltonian({"Z0 Z1": 1.0, "X0": 0.5}), ansatz=ansatz)
custom_ansatz_result = custom_ansatz_solver.run(
    initial_parameters=np.array([0.1, -0.2, 0.3, 0.4]),
)
print(f"custom ansatz:   energy = {custom_ansatz_result.energy:.6f}")


# -- 5. Custom optimizer: the minimize contract ------------------------------
class SimpleOptimizer:
    """Custom optimizer adapter implementing ``minimize(evaluate, initial_parameters)``.

    A resumable adapter additionally implements ``step(evaluate)`` (one
    classical iteration), the built-in state attributes, and the
    ``state_dict()`` / ``load_state_dict()`` protocol; the VQE
    documentation covers the full contract.  Without the protocol the
    run works but cannot be resumed, which VQE reports explicitly.
    """

    method = "simple"

    def __init__(self):
        self.energy = None
        self.parameters = None
        self.energy_history = []
        self.task_ids = []
        self.iterations = 0
        self.converged = False

    def minimize(self, evaluate, initial_parameters):
        parameters = np.asarray(initial_parameters, dtype=float).copy()
        for _ in range(20):
            self.parameters = parameters
            self.energy = float(evaluate(parameters))
            self.energy_history.append(self.energy)
            self.iterations += 1
            # The Z ground state is |1>, reached by the RY rotation at
            # angle pi; halve the distance to pi each iteration.
            parameters = parameters + (np.pi - parameters) * 0.5
        self.converged = True


custom_solver = VQE(hamiltonian, optimizer=SimpleOptimizer())
custom_result = custom_solver.run(
    initial_parameters=np.array([0.4, 0.4]),
    config=VQEConfig(max_iterations=20, tolerance=1e-6),
)
print(f"custom optimizer: energy = {custom_result.energy:.6f}")
print(f"                  resumable = {custom_result.metadata['resume_supported']}")

# -- 6. Checkpoint and resume -------------------------------------------------
# submit() returns a resumable AlgorithmTask: each poll completes one
# classical optimization iteration.  Checkpoint while the task is
# RUNNING, then reconstruct it with AlgorithmTask.resume; the checkpoint
# holds the optimizer state and the solver fingerprint, never a live
# session or credentials.
with tempfile.TemporaryDirectory(prefix="vqe_example_") as tmp_dir:
    checkpoint_path = os.path.join(tmp_dir, "vqe.json")
    task = VQE(hamiltonian).submit(
        initial_parameters=np.array([0.2, 0.0]),
        config=VQEConfig(max_iterations=3, tolerance=1e-12),
    )
    task.poll()  # one classical iteration completed
    task.checkpoint(checkpoint_path)
    restored = AlgorithmTask.resume(checkpoint_path, backend=LocalBackend())
    resumed_result = restored.result()  # continues from the checkpoint
    print(f"checkpoint resume: iterations = {resumed_result.iterations}")
    print(f"                    energy = {resumed_result.energy:.6f}, "
          f"converged = {resumed_result.converged}")

# -- 7. Remote runtime execution (optional) -----------------------------------
# The same solver runs on a qpanda3-runtime service by passing a
# QPandaRuntimeBackend instead of LocalBackend.  Install the optional
# dependency (pip install pyqpanda-algorithm[runtime]) and read the
# credentials from environment variables -- never embed API keys in
# source:
#
#     from qpanda3_runtime import RuntimeService
#     from pyqpanda_alg.execution import QPandaRuntimeBackend
#
#     service = RuntimeService(url_or_cfgfile=os.environ["QPANDA3_SERVER_URL"])
#     service.login(api_key=os.environ["QPANDA3_API_KEY"])
#     device = service.device(os.environ["QPANDA3_DEVICE_ID"])
#     remote_backend = QPandaRuntimeBackend(service, device)
#     remote_result = VQE(hamiltonian).run(
#         initial_parameters=np.array([0.2, 0.0]),
#         backend=remote_backend,
#         config=VQEConfig(max_iterations=80, tolerance=1e-6),
#     )
