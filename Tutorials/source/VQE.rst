VQE
===

``pyqpanda_alg.VQE`` is a stable Hamiltonian-level variational quantum
eigensolver.  It validates a pyqpanda3 ``Hamiltonian``, builds a
hardware-efficient ansatz (or accepts a custom one), and minimizes the
energy expectation with a classical optimizer through the shared
execution layer described in :doc:`execution`.  The default CPU path
runs on pyqpanda3's local simulator and needs no qpanda3-runtime
installation; the same solver can be pointed at a qpanda3-runtime
service through ``QPandaRuntimeBackend`` (see :doc:`runtime_algorithms`
for the shared backend contract).

The public surface is exactly three names, importable from
``pyqpanda_alg.VQE``:

* ``VQE`` -- the solver: validates the observable, owns the ansatz,
  and drives the optimization.
* ``VQEConfig`` -- the immutable run knobs: iteration budget,
  convergence tolerance, optimizer method, and sampling budget.
* ``VQEResult`` -- the frozen snapshot of a completed or interrupted
  run: energy, optimal parameters, convergence bookkeeping, energy
  history, the materialized circuit at the optimum, and execution
  provenance.

Hamiltonian input
-----------------

``VQE`` consumes a pyqpanda3 ``Hamiltonian`` directly; chemistry
preprocessing (molecular integrals, fermion-to-qubit mappings) is out
of scope.  The observable is validated on construction: it must be
non-empty, Hermitian (every coefficient in the Pauli basis must be
real), and have finite coefficients.  The qubit count is inferred from
the Pauli operator, so the ansatz spans the full Hilbert space of the
observable.

.. code-block:: python

    from pyqpanda3.hamiltonian import Hamiltonian
    from pyqpanda_alg.VQE import VQE

    solver = VQE(Hamiltonian({"Z0 Z1": 1.0, "X0": 0.5}))

The default ansatz
------------------

When no ansatz is supplied, ``VQE`` builds the built-in
hardware-efficient ansatz: alternating RY/RZ single-qubit rotation
layers followed by a linear CNOT chain.  The parameter ordering is a
stable API contract -- layer-major, qubit-major, and RY before RZ per
qubit.  For parameter ``i`` the gate it feeds is

``index = layer * (2 * num_qubits) + qubit * 2 + gate``

with ``gate`` 0 for RY and 1 for RZ: all parameters of layer 0 come
first, then layer 1, and so on; within a layer every qubit contributes
RY then RZ, and each layer ends with the linear CNOT chain ``(0,1)``,
``(1,2)``, ..., ``(n-2, n-1)``.  The parameter count is
``2 * num_qubits * layers``.  A single qubit has no entanglement
partner, so the default reduces to the same ordering without the CNOT
chain.

Running the solver
------------------

``run(initial_parameters, *, backend=None, execution_options=None,
config=None)`` minimizes the energy and returns the frozen
``VQEResult``.  ``initial_parameters`` is the one-dimensional starting
point and must match the ansatz parameter count.

.. code-block:: python

    import numpy as np
    from pyqpanda3.hamiltonian import Hamiltonian
    from pyqpanda_alg.VQE import VQE, VQEConfig

    solver = VQE(Hamiltonian({"Z0": 1.0}))
    result = solver.run(
        initial_parameters=np.array([0.2, 0.0]),
        config=VQEConfig(max_iterations=80, tolerance=1e-6),
    )
    print(result.energy)  # ≈ -1.0, the Z ground state

``VQEConfig`` carries the classical optimization knobs --
``max_iterations`` bounds the optimization loop, ``tolerance`` is the
convergence criterion on the energy change, ``optimizer`` names the
classical method (SLSQP by default) -- and the sampling budget
``shots`` (default 1000), which overrides any ``shots`` in the
``execution_options``.

The completed run is snapshotted into the immutable ``VQEResult``:
``energy`` and ``optimal_parameters`` describe the best point found,
``converged`` and ``iterations`` summarize the optimization,
``energy_history`` records every evaluated energy in iteration order,
``optimal_circuit`` is the concrete circuit at the optimum,
``task_ids`` keeps the execution provenance, and ``metadata`` carries
the optimizer method, the backend name, and the
``resume_supported`` flag.

Statistical tolerance
---------------------

Every energy evaluation is a sampled estimate with the configured
``shots`` budget, so energies fluctuate run to run; the fluctuation
shrinks as ``shots`` grows and vanishes for observables whose
eigenstate the ansatz reaches exactly (for example the one-qubit Z
ground state, whose measurement is deterministic at the optimum).  A
converged run therefore sits within shot noise of the exact value --
the example below reliably lands below -0.99 for a true ground-state
energy of -1.0 at the default 1000 shots.

The local CPU backend
---------------------

``LocalBackend`` executes circuits synchronously on pyqpanda3's
``CPUQVM`` and is the default when ``backend`` is omitted, so existing
CPU calls keep working unchanged.  Passing it explicitly is the same
path:

.. code-block:: python

    from pyqpanda_alg.execution import LocalBackend

    result = solver.run(
        initial_parameters=np.array([0.2, 0.0]),
        backend=LocalBackend(),
        config=VQEConfig(max_iterations=80, tolerance=1e-6),
    )

A custom ansatz
---------------

Any ansatz following the callable contract can replace the built-in
one: ``ansatz(parameters)`` must return the bound circuit through
``ansatz(parameters).circuits()[0]`` (a ``QProg``), and the ansatz
must expose ``mutable_parameter_total()`` so the solver can validate
the initial parameters.  A pyqpanda3 ``VQCircuit`` built with
``set_Param`` fulfills both.

.. code-block:: python

    from pyqpanda3.core import CNOT, RY, RZ
    from pyqpanda3.hamiltonian import Hamiltonian
    from pyqpanda3.vqcircuit import VQCircuit
    from pyqpanda_alg.VQE import VQE

    ansatz = VQCircuit(2)
    ansatz.set_Param([4])
    ansatz << RY(0, ansatz.Param([0])) << RZ(0, ansatz.Param([1]))
    ansatz << RY(1, ansatz.Param([2])) << RZ(1, ansatz.Param([3]))
    ansatz << CNOT(0, 1)

    solver = VQE(Hamiltonian({"Z0 Z1": 1.0}), ansatz=ansatz)

A custom optimizer
------------------

The built-in optimizer is a SciPy adapter (SLSQP by default).  Any
adapter exposing ``minimize(evaluate, initial_parameters)`` replaces
it, where ``evaluate`` returns the energy of one parameter point.  To
be resumable, the adapter additionally implements ``step(evaluate)``
(one classical iteration), the built-in state attributes (``method``,
``parameters``, ``energy``, ``energy_history``, ``iterations``,
``max_iterations``, ``converged``), and the ``state_dict()`` /
``load_state_dict()`` protocol.  A custom optimizer without the
protocol runs fine but cannot be resumed: recovery is refused loudly
with an explicit explanation at resume time, and the limitation is
flagged in ``VQEResult.metadata["resume_supported"]``.

.. code-block:: python

    from pyqpanda_alg.VQE import VQE

    class MyOptimizer:
        def minimize(self, evaluate, initial_parameters):
            ...  # one-shot minimization of evaluate(parameters)

    solver = VQE(Hamiltonian({"Z0": 1.0}), optimizer=MyOptimizer())

Remote runtime construction
---------------------------

The same solver runs on a qpanda3-runtime service by passing a
``QPandaRuntimeBackend`` instead of ``LocalBackend``.  The dependency
is optional: install it with ``pip install pyqpanda-algorithm[runtime]``.
The solver never authenticates and never selects a device -- it only
calls the backend it was given.  Application code reads credentials
from environment variables and never embeds API keys in source.

.. code-block:: python

    import os

    from qpanda3_runtime import RuntimeService
    from pyqpanda_alg.execution import QPandaRuntimeBackend

    service = RuntimeService(url_or_cfgfile=os.environ["QPANDA3_SERVER_URL"])
    service.login(api_key=os.environ["QPANDA3_API_KEY"])
    device = service.device(os.environ["QPANDA3_DEVICE_ID"])

    result = solver.run(
        initial_parameters=np.array([0.2, 0.0]),
        backend=QPandaRuntimeBackend(service, device),
    )

Variational-session behavior
----------------------------

When the backend advertises variational sessions, ``VQE`` prefers one:
a single session is created lazily on the first evaluation and reused
for the whole run, then released idempotently when the run finishes or
fails.  When the backend does not advertise sessions, every evaluation
submits an estimate batch instead.  A runtime failure never silently
falls back to the CPU backend: it surfaces as the execution layer's
public error (see :doc:`execution`).

Checkpoint recovery
-------------------

``submit(initial_parameters, ...)`` returns the resumable execution
layer ``AlgorithmTask`` instead of blocking: each ``poll()`` completes
exactly one classical optimization iteration.  A task can be
checkpointed while RUNNING and reconstructed with
``AlgorithmTask.resume``.  The checkpoint holds the optimizer state
(method, parameters, energy history, iteration, convergence criteria,
task IDs), the resume-support flag, and a solver fingerprint -- never
a live session or credentials.  Resuming validates the checkpoint
against the solver: a checkpoint created by a solver with a different
observable or ansatz raises ``TaskRecoveryError`` instead of silently
optimizing the wrong problem.

.. code-block:: python

    from pyqpanda_alg.execution import AlgorithmTask

    task = solver.submit(initial_parameters=np.array([0.2, 0.0]))
    task.poll()  # one classical iteration completed
    path = task.checkpoint("vqe.json")  # checkpoint while RUNNING

    # later, in another process (after re-login), resume from the file:
    restored = AlgorithmTask.resume(path, backend=backend)
    result = restored.result()  # continues from the checkpoint

A completed task's immutable result is not JSON-serializable, so
checkpointing a finished task is rejected loudly.

A runnable example
------------------

The worked example at ``pyqpanda-algorithm/example/VQE/
vqe_hamiltonian.py`` walks through the whole surface on the local CPU
backend -- default solver, explicit ``LocalBackend`` and configuration
knobs, custom ansatz, custom optimizer, and checkpoint recovery -- and
shows the remote runtime path as commented code reading environment
variables.  Run it standalone from the repository root:

.. code-block:: bash

    python pyqpanda-algorithm/example/VQE/vqe_hamiltonian.py
