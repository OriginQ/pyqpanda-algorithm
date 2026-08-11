Shor
====

``pyqpanda_alg.Shor`` is a stable small-scale Shor factorization solver
for odd composite moduli, driven through the shared execution layer
described in :doc:`execution`.  The public API accepts any positive
integer, but quantum execution is only promised when the resources fit
the backend: even, prime, and perfect-power moduli are resolved
explicitly by classical preprocessing, and every other modulus drives a
bounded number of quantum order-finding attempts on a backend.  The
default CPU path runs on pyqpanda3's local simulator and needs no
qpanda3-runtime installation; the same solver can be pointed at a
qpanda3-runtime service through ``QPandaRuntimeBackend`` (see
:doc:`runtime_algorithms` for the shared backend contract).

The public surface, importable from ``pyqpanda_alg.Shor``:

* ``Shor`` -- the solver facade: validates the modulus, owns the
  attempt RNG, and drives the bounded attempt pipeline.
* ``ShorConfig`` -- the immutable run knobs: the attempt budget and
  the phase-register size.
* ``ShorResult`` -- the frozen snapshot of a completed run: the factor
  pair (or prime marker), the honest ``used_quantum`` flag, the task
  IDs, the recovered order, and provenance metadata.
* ``PreprocessOutcome`` and ``classical_preprocess`` -- the
  discriminated result of the classical preprocessing step and the
  function that produces it.
* ``recover_order`` -- turns a measured phase-register sample into a
  candidate multiplicative order via continued fractions.

The lower-level builders live in their own modules, exactly as the test
suite imports them:

* ``pyqpanda_alg.Shor.circuit.build_order_finding_circuit`` --
  synthesizes the order-finding circuit for a base coprime to the
  modulus and returns the build with its register layout, gate counts,
  and depth.
* ``pyqpanda_alg.Shor.resources.estimate_shor_resources`` -- a
  backend-independent resource estimate: qubit counts, controlled
  multiplications, and approximate gate and depth budgets.

Classical fast paths
--------------------

``classical_preprocess`` classifies a modulus up front.  Prime moduli
resolve with ``is_prime=True`` and no factor pair; even moduli with the
factor pair ``(2, modulus // 2)``; perfect powers with their smallest
nontrivial root pair (81 resolves to ``(3, 27)``); odd composites are
marked as needing quantum order finding.  A classically resolved run
reports ``used_quantum=False`` and an empty ``task_ids`` -- no quantum
task is ever submitted:

.. code-block:: python

    from pyqpanda_alg.Shor import Shor

    print(Shor(12).run().factors)   # (2, 6), used_quantum=False
    print(Shor(13).run().is_prime)  # True, used_quantum=False
    print(Shor(81).run().factors)   # (3, 27), used_quantum=False

Only results that executed quantum order finding count toward QPU
qualification: ``used_quantum`` is True exactly when the result emerged
from the quantum attempt pipeline -- an order-finding success or
attempt exhaustion -- regardless of the later classical resolution of
the factorization.

Solving
-------

``run(*, backend=None, execution_options=None)`` factors the modulus
and returns the frozen ``ShorResult``.  ``backend`` is keyword-only;
when None, the local CPU backend is used.  Odd composites drive a
bounded attempt loop: each attempt draws a base from the injected RNG,
applies gcd preprocessing (a base sharing a factor with the modulus
resolves that attempt classically), submits one order-finding sample
task, parses every nonzero phase sample of the histogram by descending
count, recovers a candidate order with ``recover_order``, applies the
standard halving rule (a recovered order can be a proper multiple of
the true order), and derives the factors as
``gcd(base**(order // 2) - 1, modulus)`` and
``gcd(base**(order // 2) + 1, modulus)``.

``ShorConfig.max_attempts`` bounds the number of attempts (default 6);
when the budget runs out the run reports ``factors=None`` with
``used_quantum=True`` and the full attempt history in the metadata.
``ShorConfig.phase_qubits`` is the phase-register size; None selects
the derived default of twice the modulus bit length.

Generic circuit construction
----------------------------

``build_order_finding_circuit(base, modulus, phase_qubits=None)``
synthesizes the textbook order-finding circuit for a base coprime to an
odd modulus: a phase register of twice the modulus bit length (unless
explicitly configured), a value register wide enough for the modulus
and prepared to ``|1>``, Hadamards on the phase register, one
controlled modular multiplication per phase qubit (phase qubit ``k``
gates the multiplication by ``base**(2**k) mod modulus``), an inverse
QFT, and a phase measurement.  The returned build reports the register
layout, the pre-transpilation depth, and the per-gate counts -- it is
a construction and inspection tool, not an execution path.

.. warning::

    Release-candidate circuit examples for ``N=15`` -- or any modulus --
    must not hardcode the factors or the order.  Construction derives
    every classical constant from ``pow(base, 2**k, modulus)`` alone,
    and the factorization and order-recovery helpers are never called:
    a circuit that embeds 3, 5, or the order 4 proves nothing about
    the solver.  The implementation's no-answer-injection guarantee is
    exercised by the test suite.

Seeded, reproducible runs
-------------------------

The attempt bases are drawn from ``rng``, a ``random.Random`` by
default.  Injecting a seeded instance makes runs reproducible, and any
object exposing ``randrange`` is sufficient -- a fixed-base double
makes the drawn base deterministic, which is how the worked example
pairs base 2 (order 4 modulo 15) with its scripted device response:

.. code-block:: python

    import random

    from pyqpanda_alg.Shor import Shor

    solver = Shor(15, rng=random.Random(42))

Resource estimates
------------------

``estimate_shor_resources(modulus, phase_qubits=None)`` validates the
modulus exactly as circuit construction does and reports the qubit
counts (phase, value, ancilla, total), the number of controlled modular
multiplications (exact), and an order-of-magnitude pre-transpilation
gate and depth budget (the only approximate fields, flagged by
``approximate_labels``) -- all before any circuit is synthesized.

What is and is not promised
---------------------------

The public API accepts any positive integer, but quantum execution is
only promised when the resources fit the backend.  The phase register
alone is twice the modulus bit length, and the modular exponentiation
scales the whole circuit polynomially in the modulus size, so modest
moduli already need dozens of qubits and millions of gates: the
estimate for the 14-bit composite ``10403 = 101 * 103`` already reports
57 qubits and over a million gates.  Larger moduli are therefore not
promised to run on any particular backend, and even the local CPU
sampling of the N=15 order-finding circuit takes minutes.  Use the
resource estimate to check the qubit budget against a backend before
submitting; the runtime backend rejects a device with fewer available
qubits than the circuit needs before any task is submitted.

Runtime execution
-----------------

The same solver runs on a qpanda3-runtime service by passing a
``QPandaRuntimeBackend`` instead of ``LocalBackend``.  The dependency
is optional: install it with ``pip install pyqpanda-algorithm[runtime]``.
The solver never authenticates and never selects a device -- it only
calls the backend it was given.  Application code reads credentials
from environment variables and never embeds API keys in source.

.. code-block:: python

    import os
    import random

    from qpanda3_runtime import RuntimeService
    from pyqpanda_alg.Shor import Shor
    from pyqpanda_alg.execution import QPandaRuntimeBackend

    service = RuntimeService(url_or_cfgfile=os.environ["QPANDA3_SERVER_URL"])
    service.login(api_key=os.environ["QPANDA3_API_KEY"])
    device = service.device(os.environ["QPANDA3_DEVICE_ID"])

    result = Shor(15, rng=random.Random(42)).run(
        backend=QPandaRuntimeBackend(service, device),
    )

On the sampling path each attempt submits one order-finding sample task
and parses the returned phase histogram exactly as on the CPU;
``result.task_ids`` records the submitted task IDs in submission order.

Checkpoint recovery
-------------------

``submit(*, backend=None, execution_options=None)`` returns the
resumable execution-layer ``AlgorithmTask`` instead of blocking: each
``poll()`` executes exactly one attempt of the same state machine
``run`` drives synchronously.  A task can be checkpointed while RUNNING
and reconstructed with ``AlgorithmTask.resume``.  The checkpoint holds
the modulus, the config, the RNG draw position, and the full attempt
history (bases, completed task IDs, histograms, candidate orders,
rejection reasons) -- never the backend, credentials, or live task
handles -- and resuming rebuilds the attempt machine through the
factory registered under the algorithm name ``shor``, continuing the
exact base sequence and never resubmitting a completed task.

.. code-block:: python

    from pyqpanda_alg.execution import AlgorithmTask

    task = Shor(15).submit()
    task.poll()  # one attempt completed
    path = task.checkpoint("shor.json")  # checkpoint while RUNNING

    # later, in another process (after re-login), resume from the file:
    restored = AlgorithmTask.resume(path, backend=backend)
    snapshot = restored.result()  # JSON-safe dict snapshot of the result

A completed Shor task's accumulated result is the same JSON-safe
snapshot, so a finished task can be checkpointed too.

A runnable example
------------------

The worked example at ``pyqpanda-algorithm/example/Shor/factor_15.py``
walks through the whole surface on the CPU -- the classical fast paths,
generic circuit construction and inspection, the resource estimate, a
seeded quantum order-finding run against a scripted device-response
stub (documented inside the example: a real local sample of the N=15
circuit takes minutes, so only the device counts are scripted while the
whole solver pipeline runs untouched), and checkpoint recovery -- and
shows the remote qpanda3-runtime path as commented code reading
environment variables.  Run it standalone from the repository root:

.. code-block:: bash

    python pyqpanda-algorithm/example/Shor/factor_15.py
