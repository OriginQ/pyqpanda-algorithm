"""Shor factorization of 15: classical fast paths and a scripted device stub.

This example walks the stable public surface of ``pyqpanda_alg.Shor``
entirely on the CPU: no qpanda3-runtime installation and no credentials
are required.  It demonstrates the classical fast paths (even, prime,
and perfect-power moduli resolve without any quantum task), the generic
order-finding circuit construction and its inspection, the
backend-independent resource estimate, a seeded reproducible quantum
order-finding run, and checkpoint recovery through the shared execution
layer.

One deliberate substitution keeps this example fast and honest at the
same time.  A real local order-finding sample of the N=15 circuit on
pyqpanda3's CPU QVM takes minutes, so the quantum run below drives the
complete solver pipeline -- base draw, gcd preprocessing, circuit
synthesis, sample-task submission, phase parsing, order recovery, and
factor derivation -- against :class:`ScriptedDeviceStub`, a stand-in
backend that returns the phase histogram a device would produce (see
its definition in section 4).  ``result.used_quantum`` is therefore
True exactly as it would be for a real device run: a quantum
order-finding task was submitted, and only the counts were scripted.
Replace the stub with a ``QPandaRuntimeBackend`` (section 6) for real
execution.

Run it standalone from the repository root::

    python pyqpanda-algorithm/example/Shor/factor_15.py
"""

import os
import tempfile

from pyqpanda_alg.Shor import Shor, ShorConfig, classical_preprocess
from pyqpanda_alg.Shor.circuit import build_order_finding_circuit
from pyqpanda_alg.Shor.resources import estimate_shor_resources
from pyqpanda_alg.execution import (
    AlgorithmTask,
    CompletedBackendTask,
    ExecutionOptions,
    SampleBatchResult,
)

# -- 1. Classical fast paths: no quantum task is ever submitted ---------------
# Even, prime, and perfect-power moduli resolve during preprocessing with
# used_quantum=False and an empty task list; classical_preprocess is the
# function behind the resolution and classifies any modulus up front.
for modulus in (12, 13, 81):
    resolved = Shor(modulus).run()
    outcome = classical_preprocess(modulus)
    print(
        f"classical {modulus}: factors={resolved.factors} "
        f"is_prime={resolved.is_prime} used_quantum={resolved.used_quantum} "
        f"({outcome.status})"
    )

# -- 2. Generic circuit construction and inspection -----------------------------
# build_order_finding_circuit synthesizes the textbook order-finding
# circuit for a base coprime to the modulus: a phase register of twice
# the modulus bit length (8 qubits for 15), a value register prepared
# to |1>, one controlled modular multiplication per phase qubit, an
# inverse QFT, and a phase measurement.  Every classical constant is
# derived from pow(base, 2**k, modulus) alone -- the factors 3 and 5
# and the order 4 never appear anywhere in the circuit, and the
# factorization and order-recovery helpers are never called during
# construction.
build = build_order_finding_circuit(2, 15)
print(
    f"circuit (2, 15): phase={len(build.phase_qubits)} "
    f"value={len(build.value_qubits)} ancilla={len(build.ancilla_qubits)} "
    f"total={build.total_qubits} qubits, depth={build.depth}"
)
print(f"circuit gate counts: {build.gate_counts}")

# -- 3. Backend-independent resource estimate ----------------------------------
# estimate_shor_resources validates the modulus and reports the qubit
# counts and controlled multiplications exactly (the only quantum
# arithmetic blocks), plus an order-of-magnitude pre-transpilation gate
# and depth budget flagged by approximate_labels -- all before any
# circuit is synthesized.  The estimate for a 14-bit odd composite
# shows why arbitrary large moduli are not promised: resources grow
# polynomially and quickly outgrow any real device.
estimate = estimate_shor_resources(15)
print(
    f"estimate N=15: total={estimate.total_qubits} qubits "
    f"(phase {estimate.phase_qubits} + value {estimate.value_qubits} "
    f"+ ancilla {estimate.ancilla_qubits}), "
    f"controlled multiplies={estimate.controlled_multiplies}, "
    f"gates~{estimate.synthesized_gates}, depth~{estimate.estimated_depth}"
)
large = estimate_shor_resources(10403)  # 101 * 103, a 14-bit modulus
print(
    f"estimate N=10403: total={large.total_qubits} qubits, "
    f"gates~{large.synthesized_gates}"
)

# -- 4. Seeded quantum order finding against a scripted device stub -------------
# The run below is a real Shor attempt pipeline; only the device
# response is scripted.  The stub wraps its histograms exactly as a
# completed backend sample task is wrapped, so the solver's phase
# parsing, order recovery, and factor derivation run untouched.


class FixedBaseRng:
    """RNG stand-in that always draws the same base.

    Injecting it makes the run reproducible: the drawn base is 2,
    which is coprime to 15 and has order 4, matching the scripted
    histograms below.  Any object exposing ``randrange`` works; a
    seeded ``random.Random`` is the usual alternative.
    """

    def __init__(self, value):
        self.value = value

    def randrange(self, a, b):
        return self.value


class ScriptedDeviceStub:
    """Device-response stand-in for this example, not a real backend.

    ``submit_sample`` is the only method the solver calls; each call
    returns the next scripted phase histogram (the last one repeats
    for overflow), wrapped exactly as a real backend would wrap a
    completed sample task.  The histograms below are what a noiseless
    device would return for base 2 modulo 15: the order-4 phases 1/4
    and 3/4 appear as the 8-bit samples 01000000 (64) and 11000000
    (192), and the zero histogram of section 5 carries no phase.
    Only the counts are scripted -- the circuit submitted to this stub
    is the real order-finding circuit.  Replace this stub with a
    ``QPandaRuntimeBackend`` (section 6) for real device execution.
    """

    def __init__(self, histograms):
        self.histograms = list(histograms)
        self.submitted_tasks = []

    def submit_sample(self, circuit, *, options):
        index = min(len(self.submitted_tasks), len(self.histograms) - 1)
        counts = self.histograms[index]
        self.submitted_tasks.append(counts)
        return CompletedBackendTask(
            SampleBatchResult(counts=(counts,), shots=options.shots),
            task_id=f"stub-sample-{len(self.submitted_tasks)}",
        )


stub = ScriptedDeviceStub(
    [
        {"01000000": 600, "11000000": 400},
        {"01000000": 600, "11000000": 400},
    ]
)
result = Shor(15, rng=FixedBaseRng(2)).run(
    backend=stub, execution_options=ExecutionOptions(shots=8192)
)
print(
    f"quantum stub run: factors={result.factors} order={result.order} "
    f"used_quantum={result.used_quantum}"
)
print(
    f"  task_ids={result.task_ids} "
    f"submitted histograms={stub.submitted_tasks}"
)

# -- 5. Checkpoint and resume ---------------------------------------------------
# submit() returns a resumable AlgorithmTask running one attempt per
# poll.  The first attempt below draws the zero sample (carrying no
# phase) and is rejected, leaving the task RUNNING; the checkpoint then
# captures the attempt history and the RNG draw position, and resume
# rebuilds the attempt machine through the factory registered under the
# algorithm name "shor" -- never a live backend or credentials.  The
# resumed run re-draws base 2 and the stub's second histogram factors
# 15; the completed task's result is the JSON-safe snapshot dict.
checkpoint_stub = ScriptedDeviceStub(
    [
        {"00000000": 1000},
        {"01000000": 600, "11000000": 400},
    ]
)
with tempfile.TemporaryDirectory(prefix="shor_example_") as tmp_dir:
    checkpoint_path = os.path.join(tmp_dir, "shor.json")
    task = Shor(
        15, rng=FixedBaseRng(2), config=ShorConfig(max_attempts=2)
    ).submit(backend=checkpoint_stub)
    task.poll()  # one rejected attempt, task stays RUNNING
    task.checkpoint(checkpoint_path)
    restored = AlgorithmTask.resume(checkpoint_path, backend=checkpoint_stub)
    resumed = restored.result()  # continues from the checkpoint
    print(
        f"checkpoint resume: factors={resumed['factors']} "
        f"used_quantum={resumed['used_quantum']} "
        f"task_ids={resumed['task_ids']}"
    )

# -- 6. Remote runtime execution (optional) -------------------------------------
# The same solver runs on a qpanda3-runtime service by passing a
# QPandaRuntimeBackend instead of the stub.  Install the optional
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
#     remote_result = Shor(15, rng=FixedBaseRng(2)).run(
#         backend=QPandaRuntimeBackend(service, device),
#         execution_options=ExecutionOptions(shots=8192),
#     )
