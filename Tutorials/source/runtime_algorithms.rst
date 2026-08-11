Runtime Algorithms
==================

Every public algorithm in ``pyqpanda_alg`` runs through the shared
execution layer described in :doc:`execution`.  Since release 2.1.0 the
algorithms no longer construct pyqpanda3's ``CPUQVM`` directly: each
executable entry point takes a ``backend`` keyword argument (keyword-only,
so existing positional calls keep working), defaults to the local CPU
backend when it is omitted, and never silently replaces a supplied
backend.  A runtime failure therefore surfaces to the caller instead of
falling back to the CPU.

The migrated surface

All backend arguments are keyword-only and accept ``ExecutionOptions``
through the sibling ``execution_options`` keyword:

.. code-block:: python

    from pyqpanda_alg.execution import ExecutionOptions, LocalBackend

    backend = LocalBackend()
    result = algorithm.run(..., backend=backend,
                           execution_options=ExecutionOptions(shots=1000))

Omitting both arguments keeps the historical CPU behaviour: work executes
synchronously on pyqpanda3's ``CPUQVM`` inside ``LocalBackend``.

Algorithms with ``submit()``
----------------------------

The following classes expose both a convenience ``run(...)`` method and a
``submit(...)`` method that returns the execution layer's
``AlgorithmTask`` (see :doc:`execution`).  Checkpointing and recovery
(``resume()``) apply to the multi-round state machines --
``GroverAdaptiveSearch``, ``QAE``/``IQAE``, and ``QARM`` -- which run one
round per ``poll()``; ``QAOA``, ``QUBO``, and ``QmRMR`` complete their
whole optimization in a single step, so their tasks are execution handles
rather than resumable state machines:

* ``QAOA`` -- ``QAOA(problem).run(layer=1, ...)`` / ``.submit(...)``
* ``QUBO`` -- ``QUBO_QAOA(problem).run(...)`` / ``QUBO_GAS_origin(problem).run(...)``
* ``QAE`` -- ``QAE(operator_in=..., qnumber=..., epsilon=...).run(...)``
* ``QARM`` -- ``QuantumAssociationRulesMining(transactions, min_sup, min_conf).run(...)``
* ``QmRMR`` -- ``Feature_Selection(quadratic, linear, n_features).submit(ini_para, ...)``
* ``GroverAdaptiveSearch`` -- ``GroverAdaptiveSearch(init_value=0, n_index=...).run(continue_times=..., ...)``

Example -- QAE with an explicit backend:

.. code-block:: python

    from pyqpanda3.core import QCircuit, RY, X
    from pyqpanda_alg.QAE import QAE
    from pyqpanda_alg.execution import (
        ExecutionOptions,
        LocalBackend,
        TaskStatus,
    )

    def operator(qlist):
        cir = QCircuit()
        cir << RY(qlist[0], 3.1415926 / 3) << X(qlist[1]).control(qlist[0])
        return cir

    task = QAE(operator_in=operator, qnumber=2, epsilon=0.01,
               res_index=[0, 1], target_state="11").submit(
        backend=LocalBackend(), execution_options=ExecutionOptions())
    while task.poll() is TaskStatus.RUNNING:
        pass
    print(task.result())

Sampling entry points
---------------------

The remaining algorithms build circuits and submit sampling work
internally; their entry points take the same ``backend`` /
``execution_options`` keywords but do not expose ``submit()``:

* ``Grover`` -- ``Grover(flip_operator=...).cir(q_input=...)`` builds the
  search circuit; submit it with ``LocalBackend().submit_sample(...)``.
* ``QKmeans`` -- ``QuantumKmeans(k=...).fit(data, backend=...)``
* ``QPCA`` -- ``qpca(sample_A, k, backend=...)``
* ``QSVM`` -- ``QuantumKernel_vqnet(n_qbits=...).evaluate(X, backend=...)``
* ``QSVR`` -- ``Quantum_SVR(X, y).get_res(backend=...)``
* ``QSVD`` -- ``SVD(matrix).QSVD_min(backend=..., maxiter=...)``
* ``QSEncode`` -- ``QSpare_Code(data, cut_length=...).Quantum_Res(backend=...)``

Example -- Grover sampling through the local backend:

.. code-block:: python

    from pyqpanda3.core import QProg
    from pyqpanda_alg.Grover import Grover, mark_data_reflection
    from pyqpanda_alg.execution import ExecutionOptions, LocalBackend

    def mark(qubits):
        return mark_data_reflection(qubits=qubits, mark_data=["101", "001"])

    prog = QProg()
    prog << Grover(flip_operator=mark).cir(q_input=list(range(3)))

    counts = LocalBackend().submit_sample(
        prog, options=ExecutionOptions(shots=1000)
    ).result().single_counts()
    print(counts)  # the marked states dominate, statistics vary

Remote submission
-----------------

Passing a ``QPandaRuntimeBackend`` (see :doc:`execution`) instead of
``LocalBackend`` submits the same algorithm work to the qpanda3-runtime
service.  The algorithm code itself never authenticates and never selects
a device: it only calls the backend it was given.
