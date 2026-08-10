Execution Layer
================

``pyqpanda_alg.execution`` is the capability-based execution layer shared by
every backend: CPU simulation through pyqpanda3, and optional remote
submission through qpanda3-runtime.  Backends always submit work and return
a task object; callers poll the task and collect the result through one of
the batch result wrappers.  The whole public surface is importable from
``pyqpanda_alg.execution`` in one place.

Every backend follows the same submission contract:

* ``submit_sample(circuit, *, options)`` -> task whose result is a
  ``SampleBatchResult`` (counts dictionaries).
* ``submit_estimate((circuit, observable), *, options)`` -> task whose
  result is an ``EstimateBatchResult`` (expectation values).
* ``submit_statevector(circuit, *, options)`` -> task whose result is a
  ``StatevectorBatchResult`` (read-only complex arrays).
* ``create_variational_session(ansatz, observable, *, options)`` -> a
  context-managed ``VariationalSession`` whose ``run(parameters)`` returns
  an expectation-value task.

A submitted task exposes ``id``, ``status()``, ``try_result()``,
``result(timeout=None)``, and ``checkpoint(path=None)``.  ``ExecutionOptions``
carries the runtime-independent knobs (``shots``, ``timeout``,
``preflight``, ``specified_block``, and the qpanda3-runtime boolean flags).

The CPU default
----------------

Existing CPU calls keep working unchanged: when no backend is supplied,
``resolve_backend(None)`` returns ``LocalBackend``, which executes circuits
synchronously on pyqpanda3's ``CPUQVM``.  A supplied backend is never
replaced, so a runtime failure surfaces to the caller instead of silently
falling back to the CPU.

.. code-block:: python

    from pyqpanda3.core import CNOT, H, QProg, measure
    from pyqpanda_alg.execution import ExecutionOptions, resolve_backend

    prog = QProg()
    prog << H(0) << CNOT(0, 1) << measure([0, 1], [0, 1])

    backend = resolve_backend()  # no argument: the local CPU backend
    task = backend.submit_sample(prog, options=ExecutionOptions(shots=200))
    counts = task.result().single_counts()
    print(counts)  # e.g. {"00": 101, "11": 99} -- statistics vary

Sampling, estimation, and state vectors
-----------------------------------------

Sampling returns counts through ``SampleBatchResult.single_counts()`` when
exactly one circuit was submitted.

.. code-block:: python

    from pyqpanda3.core import H, QProg
    from pyqpanda3.hamiltonian import Hamiltonian
    from pyqpanda_alg.execution import ExecutionOptions, LocalBackend

    prog = QProg()
    prog << H(0)
    task = LocalBackend().submit_estimate(
        (prog, Hamiltonian({"X0": 1.0})),
        options=ExecutionOptions(shots=1),
    )
    value = task.result().single_value()  # ≈ 1.0

State-vector execution is LocalBackend-only.  The returned array is a
read-only copy, so mutating it never corrupts the result wrapper.

.. code-block:: python

    from pyqpanda3.core import H, QProg
    from pyqpanda_alg.execution import ExecutionOptions, LocalBackend

    prog = QProg()
    prog << H(0)
    state = LocalBackend().submit_statevector(
        prog, options=ExecutionOptions()
    ).result().single_statevector()
    # |0⟩ and |1⟩ each carry amplitude 1/√2

Estimation and state-vector execution reject circuits that contain
measurements with ``AlgorithmInputError``.

The optional qpanda3-runtime backend
--------------------------------------

``QPandaRuntimeBackend`` submits sampling and estimation to a
qpanda3-runtime service.  The dependency is optional: install it with the
``runtime`` extra

.. code-block:: bash

    pip install pyqpanda-algorithm[runtime]

Without it, importing ``pyqpanda_alg`` still works; only constructing the
backend raises ``MissingRuntimeDependencyError`` naming that exact install
command.

The backend never authenticates and never selects a device: it receives an
already logged-in service and an explicit device.  Application code reads
credentials from environment variables -- never embed API keys in source.

.. code-block:: python

    import os

    from qpanda3_runtime import RuntimeService
    from pyqpanda_alg.execution import (
        ExecutionOptions,
        QPandaRuntimeBackend,
    )

    service = RuntimeService(url_or_cfgfile=os.environ["QPANDA3_SERVER_URL"])
    service.login(api_key=os.environ["QPANDA3_API_KEY"])
    device = service.device(os.environ["QPANDA3_DEVICE_ID"])

    backend = QPandaRuntimeBackend(service, device)
    task = backend.submit_sample(prog, options=ExecutionOptions(shots=1000))
    counts = task.result().single_counts()

State-vector submission on the runtime backend fails fast with
``DeviceCapabilityError`` before any service call, because the runtime
surface does not advertise state-vector support.

.. code-block:: python

    from pyqpanda_alg.execution import (
        DeviceCapabilityError,
        ExecutionOptions,
        QPandaRuntimeBackend,
    )

    backend = QPandaRuntimeBackend(service, device)
    try:
        backend.submit_statevector(prog, options=ExecutionOptions())
    except DeviceCapabilityError:
        print("state-vector execution is LocalBackend-only")

FakeBackend constraints
------------------------

Before every runtime submission, the preflight steps selected by
``ExecutionOptions.preflight`` run against the device's ``FakeBackend``,
which mirrors the real device's gate set and topology:

* ``PreflightMode.NONE`` submits verbatim, with no preflight work.
* ``PreflightMode.TRANSPILE_ONLY`` (the default) validates qubit capacity,
  observable qubits, gate set, topology, and placement block, then runs the
  fake backend's transpile path.
* ``PreflightMode.FAKE_EXECUTE`` additionally runs the fake task and records
  its metadata on the returned task as ``fake_execution`` before the real
  submission.

On some platforms the fake backend's multiprocessing entry point requires
an ``if __name__ == "__main__":`` guard.  Such environment failures surface
as ``BackendUnavailableError`` naming ``fake_execute`` in the message, so
callers can switch the preflight mode instead of guessing.

.. code-block:: python

    from pyqpanda_alg.execution import (
        BackendUnavailableError,
        ExecutionOptions,
        PreflightMode,
    )

    options = ExecutionOptions(preflight=PreflightMode.FAKE_EXECUTE)
    try:
        task = backend.submit_sample(prog, options=options)
    except BackendUnavailableError as exc:
        if "fake_execute" in str(exc):
            task = backend.submit_sample(
                prog,
                options=ExecutionOptions(
                    preflight=PreflightMode.TRANSPILE_ONLY
                ),
            )

A rejected or failed preflight raises a public execution-layer error
(``DeviceCapabilityError``, ``TranspilationError``) before the service is
touched, so no half-submitted task is ever left behind.

Variational sessions
---------------------

A variational session binds an ansatz and an observable once and evaluates
many parameter sets.  ``run(parameters)`` returns an expectation-value task
whose result is a float.  Every session is a context manager: the
underlying session is released exactly once on exit -- also when the body
raises -- and release is idempotent.

.. code-block:: python

    with backend.create_variational_session(
        ansatz, observable, options=ExecutionOptions(shots=1000)
    ) as session:
        task = session.run([0.1, 0.2])
        value = task.result()  # expectation value as a float

Checkpointing and recovery
---------------------------

An ``AlgorithmTask`` wraps one algorithm run as a deterministic state
machine: each ``poll()`` executes exactly one step through a process-local
``advance`` callback that submits backend work and reports completion.
The callback itself is never serialized; a JSON checkpoint holds only the
state, backend task IDs, backend identity, and redacted metadata.

.. code-block:: python

    from pyqpanda_alg.execution import (
        AlgorithmTask,
        CompletedBackendTask,
        ExecutionOptions,
        TaskStatus,
        register_algorithm,
        resolve_backend,
    )

    def build_counter_advance(backend):
        """Rebuild the step callback when a checkpoint is resumed."""
        def advance(state):
            task = backend.submit_sample(
                prog, options=ExecutionOptions(shots=100)
            )
            step_counts = sum(task.result().single_counts().values())
            state["total"] += step_counts
            return CompletedBackendTask(step_counts), state["total"] >= 200
        return advance

    register_algorithm("sample-counter", build_counter_advance)

    task = AlgorithmTask(
        algorithm="sample-counter",
        initial_state={"total": 0},
        advance=build_counter_advance(resolve_backend()),
    )
    while task.poll() is TaskStatus.RUNNING:
        pass
    path = task.checkpoint("task.json")

    # Later, in another process (after re-login), resume from the file:
    restored = AlgorithmTask.resume(path, backend=resolve_backend())
    restored.result()  # 200

``AlgorithmTask.resume`` rebuilds the callback only through the factory
registered under the checkpoint's algorithm name; arbitrary Python objects
are never deserialized.  Checkpoints never contain API keys, tokens, live
``RuntimeService`` or ``QDevice`` handles, or live task objects -- metadata
is filtered recursively for credential-named keys before anything is
written, and ``backend`` only contributes its identity.

No silent fallback
-------------------

The execution layer has one non-negotiable rule: a runtime failure never
silently falls back to the CPU backend or to a classical substitute result.
Every failure surfaces as a public exception derived from
``AlgorithmExecutionError`` (``TaskSubmissionError``,
``BackendUnavailableError``, ``TaskTimeoutError``, ``DeviceCapabilityError``,
``ResultDecodingError``, ``TaskRecoveryError``, ...), and transport failures
always retain the original exception as ``__cause__``.  Submission is never
retried automatically; only idempotent status queries may use bounded
backoff.  ``resolve_backend`` never replaces a supplied backend, and
``QPandaRuntimeBackend`` never catches a runtime error to run the work
locally instead.
