HHL
===

``pyqpanda_alg.HHL`` is a stable solver for finite Hermitian linear
systems ``A x = b`` through the HHL quantum subroutine: it validates
the system, synthesizes the HHL circuit, and solves it on the shared
execution layer described in :doc:`execution`.  The default CPU path
runs on pyqpanda3's local simulator and needs no qpanda3-runtime
installation; the same solver can be pointed at a qpanda3-runtime
service through ``QPandaRuntimeBackend`` (see :doc:`runtime_algorithms`
for the shared backend contract).

The public surface is importable from ``pyqpanda_alg.HHL``:

* ``HHL`` -- the solver: validates the system, synthesizes the circuit,
  and runs it on a backend.
* ``HHLConfig`` -- the immutable run knobs: the ill-conditioning
  threshold and the phase-estimation register size.
* ``HHLSolution`` -- the frozen snapshot of a solved system: success
  probability, optional state vector, and optional reconstructed
  classical solution.
* ``NormalizedLinearSystem`` and ``normalize_linear_system`` -- the
  validated, padded, normalized form of an input system and the
  function that produces it.
* ``build_hhl_circuit`` -- synthesizes the HHL circuit for a validated
  system and returns the build with its register and scale metadata.
* ``estimate_hhl_resources`` -- a backend-independent resource estimate
  (qubit counts, gate budget, tomography circuits, shots).
* ``build_HHL_circuit``, ``expand_linear_equations``,
  ``HHL_solve_linear_equations`` -- compatibility wrappers keeping the
  legacy notebook entry points working on the stable surface.

Matrix input and validation
---------------------------

``HHL`` consumes a matrix and a right-hand vector directly; the matrix
may be a 2-D array or a flat list whose length is a perfect square
(the legacy flattened form).  The input is validated at construction:
the matrix must be finite, square, and Hermitian (``A == A.conj().T``,
so real symmetric matrices are fine), the vector must be a matching
finite sequence with nonzero norm, and the system must be invertible.
Singular and over-threshold ill-conditioned matrices are rejected with
the execution layer's public error before any task submission; the
rejection message reports the estimated condition number.  The
threshold is ``HHLConfig.max_condition_number`` (default ``1e6``).

.. code-block:: python

    import numpy as np
    from pyqpanda_alg.HHL import HHL

    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    vector = np.array([1.0, 1.0])
    solver = HHL(matrix, vector, precision=1e-3)

Padding rule
------------

Systems whose dimension is not a power of two are padded to the next
power of two by block-diagonal augmentation:

.. code-block:: text

    A_padded = [[A, 0], [0, I]]      b_padded = [b, 0, ..., 0]

The identity block contributes only unit eigenvalues, so padding never
introduces a zero eigenvalue into an invertible system.  The rule is
reversible: a computed solution is truncated to its first
``original_dimension`` entries and scaled by ``original_vector_norm``
to recover the solution of the original unnormalized system.

Precision and the phase register
--------------------------------

``precision`` is the QPE phase-resolution target: the phase register
must resolve phase fractions down to ``1 / 2**p <= precision``, so the
default phase register size is ``ceil(log2(1 / precision))`` (at least
one qubit).  A smaller precision yields a finer eigenvalue estimate and
a more accurate reciprocal rotation at the cost of more phase qubits;
the smallest eigenvalue must be resolvable by that register for the
reciprocal rotation to invert it faithfully, so a coarse register can
return an inaccurate direction for an ill-conditioned spectrum.  The
low-level builders take the register size directly through
``HHLConfig.phase_qubits`` (default 4); the ``HHL`` facade derives it
from ``precision``.

Resource estimates
------------------

``estimate_hhl_resources(matrix, vector, config)`` validates the raw
system exactly as the solver does and reports the qubit counts (data,
phase, ancilla, total), the number of controlled evolutions, an
approximate pre-transpilation gate count (the only approximate field,
flagged by ``approximate_labels``), the tomography basis-circuit count
``3 ** data_qubits``, and the default shot request -- before any
circuit is synthesized.

.. code-block:: python

    from pyqpanda_alg.HHL import HHLConfig, estimate_hhl_resources

    estimate = estimate_hhl_resources(matrix, vector, HHLConfig())
    print(estimate.total_qubits)          # e.g. 6
    print(estimate.tomography_circuits)   # 3 ** data_qubits

Solving on the local CPU backend
--------------------------------

``run(*, backend=None, execution_options=None, reconstruct=False,
observables=None, checkpoint_path=None)`` solves the system and
returns the frozen ``HHLSolution``.  ``backend`` is keyword-only; when
None, the local CPU backend is used.  On the local state-vector path
the success probability is the weight of the ancilla success branch,
and the post-selected data register is the solution of the padded
system.  With the default ``reconstruct=False`` the run reports the
success probability and the post-selected data state, matching the
runtime contract of not reconstructing the full vector; with
``reconstruct=True`` the classical solution direction (unit norm,
truncated to the original dimension) and the residual
``||A x - b|| / ||b||`` against the original unpadded system are
computed as well.

.. code-block:: python

    result = solver.run(reconstruct=True)
    print(result.success_probability)    # weight of the success branch
    print(result.classical_vector)       # solution direction, unit norm
    print(result.residual)               # error against the original system

Runtime execution
-----------------

On any other (runtime) backend the run is executed through sampling:
the success ancilla and the requested data observables are measured,
and the success probability is the weight of the success outcomes in
the counts.  The runtime default is success probability and the
requested observables, never full-vector reconstruction.

With ``reconstruct=True`` the solution is recovered by Pauli-basis
tomography: exactly ``3 ** data_qubits`` X/Y/Z basis-measurement
circuits -- the count declared by ``estimate_hhl_resources`` -- are
submitted one basis batch per step of a resumable execution-layer
``AlgorithmTask``, with a checkpoint after every completed batch when
``checkpoint_path`` is given.  The density matrix reconstructed from
the post-selected counts yields the solution direction, and the
reconstruction fidelity and uncertainty are reported in
``metadata["tomography"]``; the uncertainty is a self-consistency
measure (``1 - lambda_max``) of the reconstructed density matrix, not a
statistical error bar, and both are clamped into ``[0, 1]`` because
shot noise can make the linear-inversion estimate non-physical.
Observables are rejected on the local state-vector path and cannot be
combined with ``reconstruct=True`` on the sampling path, which measures
the full Pauli basis instead.  The sampling path requires a backend
advertising sampling capability; the capability is checked before any
submission.

Compatibility wrappers
----------------------

The legacy notebooks called ``build_HHL_circuit``,
``expand_linear_equations``, and ``HHL_solve_linear_equations`` with
the matrix given as a flat list and an integer as the third positional
argument -- the number of decimal digits the solution is accurate to.
The wrappers keep that meaning: the count maps to the QPE
phase-resolution target ``precision = 10 ** (-precision_cnt)``, which
yields the phase register sizes the legacy package's documentation
reported (4 qubits for one decimal digit, 7 for two, ...).  The
wrappers are thin: they delegate to the stable ``HHL`` surface and
implement no solving logic of their own.

.. code-block:: python

    from pyqpanda_alg.HHL import (
        HHL_solve_linear_equations,
        build_HHL_circuit,
        expand_linear_equations,
    )

    prog = build_HHL_circuit([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], 1)
    # measurement-free QProg of the HHL circuit

    padded_matrix, padded_vector = expand_linear_equations(
        [1.0, 0.0, 0.0, 2.0], [1.0, 1.0]
    )
    # padded matrix and unit-norm padded vector the circuit consumes

    result = HHL_solve_linear_equations([1.0, 0.0, 0.0, 2.0], [1.0, 1.0], 1)
    # solution direction of diag([1, 2]) x = [1, 1], unit norm: ~ [0.894, 0.447]

``build_HHL_circuit`` returns the measurement-free ``QProg`` the legacy
notebooks printed, ``expand_linear_equations`` returns the padded
``(matrix, vector)`` pair (the vector normalized to unit norm, as the
circuit amplitude-encodes it), and ``HHL_solve_linear_equations``
returns the reconstructed classical solution direction -- unit norm,
truncated to the original dimension -- so the legacy callers' iteration
over the solution entries keeps working.

A runnable example
------------------

The worked example at ``pyqpanda-algorithm/example/HHL/
solve_linear_system.py`` walks through the whole surface on the local
CPU backend -- the default success-probability run, explicit
reconstruction with the residual, the resource estimate, and the
legacy compatibility wrappers -- and shows the remote qpanda3-runtime
success-probability path as commented code reading environment
variables.  Run it standalone from the repository root:

.. code-block:: bash

    python pyqpanda-algorithm/example/HHL/solve_linear_system.py
