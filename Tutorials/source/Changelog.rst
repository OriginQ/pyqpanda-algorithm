2.1.0 - development
====================

This entry documents the 2.1.0 development baseline. The package version
is single-sourced from ``pyqpanda_alg.__version__``, and the Sphinx
documentation metadata reports the same version.

New algorithms
--------------

This release ships three new stable, validated algorithm modules. All
three accept any positive/compatible input, validate it up front with
the execution layer's public errors, run on the local CPU backend by
default, and can be pointed at a qpanda3-runtime service through
``QPandaRuntimeBackend``. They are imported as submodules (``from
pyqpanda_alg.VQE import VQE``); the top-level package does not
re-export them.

* ``pyqpanda_alg.VQE`` -- a stable Hamiltonian-level variational quantum
  eigensolver.  It validates a pyqpanda3 ``Hamiltonian`` (non-empty,
  Hermitian with real Pauli-basis coefficients, finite coefficients),
  builds the built-in hardware-efficient ansatz (alternating RY/RZ
  rotation layers followed by a linear CNOT chain) or accepts a custom
  ansatz, and minimizes the energy expectation with a classical
  optimizer.  The public surface is exactly three names: ``VQE``,
  ``VQEConfig`` (immutable run knobs: iteration budget, convergence
  tolerance, optimizer method, sampling budget), and ``VQEResult`` (a
  frozen snapshot: energy, optimal parameters, convergence bookkeeping,
  energy history, the materialized circuit at the optimum, and execution
  provenance).  ``VQE.run(initial_parameters, *, backend=None,
  execution_options=None, config=None)`` also exposes ``submit()`` and
  is resumable.
* ``pyqpanda_alg.HHL`` -- a stable solver for finite Hermitian linear
  systems ``A x = b`` through the HHL quantum subroutine.  The input is
  validated at construction: the matrix must be finite, square, and
  Hermitian (``A == A.conj().T``), the vector must match it with nonzero
  norm, and the system must be invertible with a condition number under
  ``HHLConfig.max_condition_number`` (default ``1e6``).  Systems whose
  dimension is not a power of two are padded by reversible block-diagonal
  augmentation.  The public surface is ``HHL``, ``HHLConfig``,
  ``HHLSolution``, ``NormalizedLinearSystem`` and
  ``normalize_linear_system``, ``build_hhl_circuit``, and
  ``estimate_hhl_resources`` (a backend-independent resource estimate).
  The runtime path reconstructs the solution from Pauli-basis sampled
  observables and reports a reconstruction fidelity clamped to
  ``[0, 1]``.  Legacy wrappers ``build_HHL_circuit``,
  ``expand_linear_equations``, and ``HHL_solve_linear_equations`` keep
  old notebook entry points working.
* ``pyqpanda_alg.Shor`` -- a stable small-scale Shor factorization solver
  for odd composite moduli.  ``classical_preprocess`` resolves even,
  prime, and perfect-power moduli without submitting any quantum task
  (``ShorResult.used_quantum`` is reported honestly as ``False``); every
  other modulus drives a bounded number of quantum order-finding
  attempts, and ``recover_order`` turns a measured phase-register sample
  into a candidate multiplicative order via continued fractions.  The
  public surface is ``Shor``, ``ShorConfig`` (attempt budget and
  phase-register size), ``ShorResult``, ``PreprocessOutcome``,
  ``classical_preprocess``, and ``recover_order``.  Lower-level builders
  ``Shor.circuit.build_order_finding_circuit`` and
  ``Shor.resources.estimate_shor_resources`` are stable module-level
  entry points.

Execution layer
---------------

* The new ``pyqpanda_alg.execution`` package provides a capability-based
  execution layer shared by every backend.  By default, algorithms
  execute synchronously on the local CPU backend (``LocalBackend``)
  through pyqpanda3's ``CPUQVM``; existing valid CPU calls remain
  compatible and keep this default.
* Every backend follows the same submission contract:
  ``submit_sample`` (counts), ``submit_estimate`` (expectation values),
  ``submit_statevector`` (read-only complex arrays, ``LocalBackend``
  only), and ``create_variational_session``.  Submitted tasks expose
  ``id``, ``status()``, ``try_result()``, ``result(timeout=None)``, and
  ``checkpoint(path=None)``; ``ExecutionOptions`` carries the
  runtime-independent knobs (``shots``, ``timeout``, ``preflight``,
  ``specified_block``, and the qpanda3-runtime boolean flags).
* ``AlgorithmTask`` adds ``submit()``/``checkpoint()``/``resume()`` for
  multi-round state machines, with factory registration through
  ``register_algorithm``; checkpoints never contain credentials or live
  service/device handles.
* An optional qpanda3-runtime backend (``QPandaRuntimeBackend``) submits
  sampling and estimation to a remote qpanda3-runtime service.  The
  dependency is optional and installed via the ``runtime`` extra
  (``pip install pyqpanda_alg[runtime]``); constructing the backend
  without it raises ``MissingRuntimeDependencyError``.  Remote execution
  requires explicit injection of an already logged-in ``RuntimeService``
  and an explicitly selected device; requesting an unadvertised
  capability (such as statevector) raises ``DeviceCapabilityError``, and
  a runtime failure never silently falls back to the CPU backend or to a
  classical substitute result.
* The layer's public errors (``DeviceCapabilityError``,
  ``BackendUnavailableError``, ``MissingRuntimeDependencyError``,
  ``TaskSubmissionError``, ``TaskTimeoutError``, ``TaskRecoveryError``,
  ``ResultDecodingError``, ``TranspilationError``,
  ``AlgorithmExecutionError``, ``AlgorithmInputError``) are all
  importable from ``pyqpanda_alg.execution``.

Migrated algorithms
-------------------

* Since 2.1.0 no public algorithm constructs pyqpanda3's ``CPUQVM``
  directly: every executable entry point runs through the execution
  layer.  All backend arguments are keyword-only ``backend`` and
  ``execution_options`` keywords, so existing positional calls keep
  working and omitting both keeps the historical synchronous CPU
  behaviour.
* ``QAOA``, ``QUBO_QAOA``, ``QUBO_GAS_origin``, ``QAE``, ``IQAE``,
  ``QARM``, ``QmRMR``, and ``GroverAdaptiveSearch`` expose a
  ``submit(...)`` method returning the execution layer's
  ``AlgorithmTask`` in addition to ``run(...)``.  Checkpointing and
  recovery (``resume()``) apply to the multi-round state machines --
  ``GroverAdaptiveSearch``, ``QAE``/``IQAE``, ``QARM``, and
  ``QUBO_GAS_origin`` -- which run one round per ``poll()``; the
  resumability claims in the documentation are scoped to QUBO/QAOA
  accordingly.
* ``Grover``, ``QKmeans``, ``QPCA``, ``QSVM``, ``QSVR``, ``QSVD``, and
  ``QSEncode`` build circuits and submit sampling/estimation work
  internally; their entry points take the same ``backend`` /
  ``execution_options`` keywords but do not expose ``submit()``.
* ``QUBO_QAOA`` and ``QmRMR`` now obtain their final distributions
  through sampling when the backend exposes no statevector capability,
  closing the statevector requirement for runtime backends.
* Circuit-building helpers (comparators in ``QCmp``, Grover operators,
  and the other circuit components) remain backend-independent.

Behavior changes and deprecations
---------------------------------

* ``QPCA.qpca`` default sampling shots changed from 8192 to the
  execution layer's 1000.  Results are normalized by the actual shot
  count, so the output shape is preserved.
* ``QuantumAssociationRulesMining`` (QARM) entry points: the legacy
  ``machine_type='QCloud'`` mode is deprecated.  It warns and now
  requires an explicit ``backend`` argument; ``machine_type='CPU'``
  remains the default.
* ``QSVD``: the singular-vector overlap observables were corrected for
  rectangular matrices.
* ``QKmeans``: iterations are capped and the sampling paths hardened.
* ``QmRMR``: the objective sign is pinned and covered by tests.
* ``HHL``: the reconstruction fidelity is clamped to ``[0, 1]`` and the
  observable construction was hardened.

Release qualification and release gate
--------------------------------------

* Real-QPU release qualification is implemented and mandatory for
  2.1.0.  A fixed case inventory of 18 small-scale cases (Bell plus
  QAOA, QARM, QKmeans, QPCA, QSVM, QSVR, QUBO_GAS, QUBO_QAOA, QAE,
  QSVD, QSEncode, QmRMR, Grover, GroverAdaptiveSearch, VQE, HHL, and
  Shor) declares, for each case, the builder, domain predicate, shots,
  statistical threshold, seed, attempt budget, timeout, and required
  capabilities; the thresholds are committed and never retro-edited.
* The manually dispatched ``runtime-rc`` workflow installs the candidate
  wheel with the ``[runtime]`` extra, runs the fixed preflight
  (FakeBackend) and real-QPU cases against an explicitly selected
  device, and uploads a sanitized qualification manifest (raw service
  responses are stored only as SHA-256 digests; credential-shaped values
  fail the workflow).
* The tag release job refuses to create a GitHub Release unless
  ``check_release.py`` verifies that the manifest attests the exact
  commit, wheel digest, version, device record, and all-passed case
  verdicts.  Qualification output is never committed (committing it
  would change the commit the manifest attests to), and guarantees are
  scoped to the fixed small-scale cases: this release does not claim any
  particular device has passed beyond what the manifest records.

Packaging, CI, and documentation
--------------------------------

* Packaging metadata moved to ``pyproject.toml`` with the version
  single-sourced from ``pyqpanda_alg._version`` and the optional
  ``runtime`` extra for qpanda3-runtime.
* CI builds the wheel and tests the installed wheel on a 3 OS x 3 Python
  (3.11/3.12/3.13) matrix, enforces a committed coverage baseline
  (``tools/check_coverage_baseline.py`` + ``test/coverage-baseline.json``),
  and builds the Sphinx documentation (with a docs gate).
* New tutorial pages document the execution layer (``execution.rst``),
  the per-algorithm backend contract (``runtime_algorithms.rst``), and
  the three new algorithms (``VQE.rst``, ``HHL.rst``, ``Shor.rst``).

2.0 - 2025-10-25
===================

Features
---------------------
With the upgrade of the pyqpanda technical architecture, we have decided to migrate the underlying SDK version of the pyqpanda_alg algorithm package from pyqpanda to pyqpanda3. 

This upgrade aims to improve algorithm execution efficiency, enhance functional support capabilities, and optimize compatibility with quantum computing platforms. 

To ensure a smooth transition for users and full utilization of the new version's features, the online documentation for pyqpanda_alg will undergo comprehensive updates to reflect interface adjustments, functional differences, and best practices resulting from the SDK change.


0.2 - 2025-05-20
===================

In our latest update, we have introduced four new algorithms.

Features
---------------------

 **Quantum Singular Value Decomposition (QSVD)** : QSVD is a quantum adaptation of the classical singular value decomposition technique, enabling efficient extraction of principal components from quantum data. It offers exponential speedup for certain matrix operations and plays a foundational role in quantum machine learning and quantum image processing.

 **Quantum Support Vector Regression (QSVR)** : QSVR brings the principles of support vector regression into the quantum domain, enabling efficient fitting of nonlinear functions using quantum-enhanced kernels. It holds promise for accelerated regression tasks in data modeling, finance, and scientific forecasting.

 **Quantum Sparse State Encoding** : This algorithm provides an efficient method for encoding sparse classical data into quantum states with logarithmic resource requirements. It is essential for loading structured datasets into quantum memory, supporting a variety of algorithms such as quantum search, simulation, and optimization.

 **Quantum Minimum Redundancy Maximum Relevance (QmRMR)** : QmRMR is a quantum feature selection algorithm that identifies the most informative and least redundant features in a dataset. Designed for quantum dimensionality reduction, it boosts the performance of quantum classifiers and regression models while reducing noise and computation overhead.


0.1 - 2023-11-16
===================

In our latest update, we have introduced two new algorithms: Shor's algorithm and the HHL algorithm.

Features
---------------------

 **Shor's algorithm** , pioneered by mathematician Peter Shor, is a quantum algorithm that revolutionizes the field of number theory by efficiently factoring large numbers. This breakthrough has profound implications for cryptography, as it challenges the security of widely-used encryption schemes relying on the difficulty of factoring large numbers.

 **The HHL (Harrow-Hassidim-Lloyd) algorithm** , spearheaded by researchers Aram Harrow, Avinatan Hassidim, and Seth Lloyd, is a quantum algorithm designed to solve linear systems of equations. This algorithm offers a quadratic speedup over the best-known classical algorithms for specific types of problems. Its applications extend across diverse fields, including optimization, machine learning, and scientific simulations.





