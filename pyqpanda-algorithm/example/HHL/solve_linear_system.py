"""HHL linear-system solving on the local CPU backend, no runtime needed.

This example solves a small Hermitian linear system with
``pyqpanda_alg.HHL`` entirely on pyqpanda3's local CPU simulator: no
qpanda3-runtime installation and no credentials are required.  It walks
through the stable public surface -- the object API solve (the default
success-probability run and explicit reconstruction), the
backend-independent resource estimate, and the legacy notebook
compatibility wrappers -- and shows the remote qpanda3-runtime
success-probability path at the end as commented code that reads
credentials from environment variables only.

Run it standalone from the repository root::

    python pyqpanda-algorithm/example/HHL/solve_linear_system.py
"""

import numpy as np

from pyqpanda_alg.HHL import (
    HHL,
    HHLConfig,
    HHL_solve_linear_equations,
    build_HHL_circuit,
    estimate_hhl_resources,
    expand_linear_equations,
)

# The system diag([1, 2]) x = [1, 1] has the exact solution x = [1, 0.5].
matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
vector = np.array([1.0, 1.0])

# -- 1. CPU solve: the default run reports the success probability -------------
# The default run reports the ancilla success probability and the
# post-selected data state; it never assumes a cheap full-vector
# reconstruction (the runtime default contract).
solver = HHL(matrix, vector, precision=1e-3)
default_result = solver.run()
print(f"default run: success probability = {default_result.success_probability:.4f}")

# -- 2. Explicit reconstruction ------------------------------------------------
# reconstruct=True computes the classical solution direction (unit norm,
# truncated to the original dimension) and the residual
# ||A x - b|| / ||b|| against the original unpadded system.
result = solver.run(reconstruct=True)
print(f"reconstructed direction = {np.round(result.classical_vector.real, 6)}")
print(f"residual                = {result.residual:.4f}")

# -- 3. Backend-independent resource estimate ----------------------------------
# estimate_hhl_resources validates the system and reports the qubit
# counts, controlled evolutions, approximate gate budget, and the
# tomography basis-circuit count before any circuit is synthesized.
estimate = estimate_hhl_resources(matrix, vector, HHLConfig())
print(f"data/phase/ancilla qubits = {estimate.data_qubits}/{estimate.phase_qubits}/{estimate.ancilla_qubits}")
print(f"controlled evolutions     = {estimate.controlled_evolutions}")
print(f"tomography circuits       = {estimate.tomography_circuits}")

# -- 4. Legacy compatibility wrappers ------------------------------------------
# The legacy notebooks called build_HHL_circuit / expand_linear_equations /
# HHL_solve_linear_equations with a flat matrix list and a decimal-digit
# precision count as the third positional argument; the wrappers keep
# those calls working on the stable API.
prog = build_HHL_circuit([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], 1)
print(f"legacy circuit program:\n{prog}")

padded_matrix, padded_vector = expand_linear_equations(matrix, vector)
print(f"padded dimension         = {padded_matrix.shape[0]}")

legacy_result = HHL_solve_linear_equations([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], 1)
print(f"legacy solver direction  = {np.round(legacy_result.real, 6)}")

# -- 5. Remote runtime success probability (optional) ---------------------------
# The same solver runs on a qpanda3-runtime service by passing a
# QPandaRuntimeBackend instead of LocalBackend.  Install the optional
# dependency (pip install pyqpanda-algorithm[runtime]) and read the
# credentials from environment variables -- never embed API keys in
# source.  On the sampling path the default run measures the success
# ancilla and the requested data observables and reports the success
# probability from the counts; full-vector reconstruction requires the
# explicit reconstruct=True tomography plan.
#
#     from qpanda3_runtime import RuntimeService
#     from pyqpanda_alg.execution import QPandaRuntimeBackend
#
#     service = RuntimeService(url_or_cfgfile=os.environ["QPANDA3_SERVER_URL"])
#     service.login(api_key=os.environ["QPANDA3_API_KEY"])
#     device = service.device(os.environ["QPANDA3_DEVICE_ID"])
#     remote_result = solver.run(backend=QPandaRuntimeBackend(service, device))
#     print(remote_result.success_probability)
