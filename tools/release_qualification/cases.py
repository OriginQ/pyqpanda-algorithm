"""Fixed release-qualification case inventory.

Plan 7 Task 2: the fixed case catalog for one release candidate.  Every
executable algorithm has exactly one case (asserted by
``test/release/test_case_inventory.py``), and each case declares the
builder, expected domain predicate, shots, threshold, seed, maximum
algorithm attempts, timeout, required capabilities, and whether
FakeBackend execution is required.  All statistical values are
committed before QPU execution and are never retuned after results are
seen.

Builder contract
----------------
``case.builder()`` returns one of:

1. a :class:`QCircuit`/:class:`QProg` -- a pure sampling request
   (measurements already attached);
2. a ``(circuit, observable)`` tuple -- an expectation-estimation
   request;
3. a callable ``invoke(backend)`` -- an algorithm invocation whose
   circuit construction happens at invocation time.

Builders are import-safe (no credentials, no network) and fast (no
local simulation): the inventory test only builds requests.  Algorithm
cases use the callable form and apply
``ExecutionOptions(shots=case.shots, timeout=case.timeout)`` so the
committed shots travel with every invocation.

Threshold semantics
-------------------
``threshold`` is the fixed pass bound committed before QPU execution: a
probability floor for sampling outcomes (Grover, bell), a tolerance for
estimation/optimization values (QAE, QSEncode, VQE), or the confidence
floor for qualitative outcomes (Shor, QUBO_GAS).  The verdict layer
(Plan 7 Task 4) compares each parsed result against this committed
bound.

Domain predicate contract
-------------------------
``case.domain(parsed)`` gates the natural result shape of the case:
the counts dict for pure-circuit cases, the algorithm's returned tuple
or object for algorithm cases, and a fixed dict (``{"factors": ...}``)
for Shor.  The verdict layer must satisfy it for the case to pass.

Known Plan 3 gaps (flagged, not patched here)
---------------------------------------------
``QUBO_QAOA`` and ``QmRMR`` require the ``statevector`` capability for
their final-distribution paths; runtime backends advertise
``statevector=False``, so those two cases honestly declare
``"statevector"`` in ``required_capabilities`` and cannot pass on a
runtime backend until Plan 3 provides a sampling-based final
distribution (see docs/superpowers/plans/2026-08-10-release-qualification.md).
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import sympy as sp
from pyqpanda3.core import H, QCircuit, QProg, RY, X
from pyqpanda3.hamiltonian import Hamiltonian

from pyqpanda_alg import QARM
from pyqpanda_alg.Grover.Grover_core import Grover, GroverAdaptiveSearch
from pyqpanda_alg.HHL import HHL
from pyqpanda_alg.QAE import QAE
from pyqpanda_alg.QARM import QuantumAssociationRulesMining
from pyqpanda_alg.QCmp import int_comparator, qft_comparator
from pyqpanda_alg.QKmeans import QuantumKmeans
from pyqpanda_alg.QmRMR.QmRMR_core import Feature_Selection
from pyqpanda_alg.QPCA import qpca
from pyqpanda_alg.QAOA.default_circuits import xy_mixer
from pyqpanda_alg.QAOA.dstate import prepare_dicke_state
from pyqpanda_alg.QAOA.qaoa import QAOA
from pyqpanda_alg.QSEncode import QSpare_Code
from pyqpanda_alg.QSVD import SVD
from pyqpanda_alg.QSVM import QuantumKernel_vqnet
from pyqpanda_alg.QSVR import Quantum_SVR
from pyqpanda_alg.QUBO import QUBO
from pyqpanda_alg.QUBO.QUBO import QUBO_QAOA
from pyqpanda_alg.Shor import Shor, ShorConfig
from pyqpanda_alg.VQE import VQE, VQEConfig
from pyqpanda_alg.execution import ExecutionOptions
from pyqpanda_alg.plugin import QFT, measure_all

# ---------------------------------------------------------------------------
# Fixed statistical values committed before QPU execution.
# ---------------------------------------------------------------------------

#: Shots committed for every sampling/estimation task of every case.
_SHOTS = 1000
#: Timeout for single-run algorithm cases.
_TIMEOUT = 1800.0
#: Timeout for multi-submission cases (Shor attempts, QKmeans iterations,
#: QmRMR SPSA rounds).
_LONG_TIMEOUT = 3600.0

#: Pinned QAOA initial parameters (layer=1): deterministic, committed.
_QAOA_INITIAL_PARA = [0.5, 0.5]
#: Fixed 3-variable QUBO shared by the QUBO_QAOA and QUBO_GAS cases; its
#: unique minimum is -1.0 at state "010".
_QUBO_SYMBOLS = sp.symbols("x0 x1 x2")
_QUBO_FUNCTION = (
    -0.5 * _QUBO_SYMBOLS[0] * _QUBO_SYMBOLS[1]
    - 0.7 * _QUBO_SYMBOLS[0] * _QUBO_SYMBOLS[1]
    + 0.9 * _QUBO_SYMBOLS[1] * _QUBO_SYMBOLS[2]
    + 1.3 * _QUBO_SYMBOLS[0]
    - _QUBO_SYMBOLS[1]
    - 0.5 * _QUBO_SYMBOLS[2]
)
#: The four-point data set shared by the QKmeans/QPCA cases.
_SAMPLE_POINTS = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
#: Two-point training set shared by the QSVM/QSVR cases.
_SAMPLE_X = np.array([[0.0, 0.0], [1.0, 1.0]])
_SAMPLE_Y = np.array([0.0, 1.0])


@dataclass(frozen=True)
class QualificationCase:
    """One fixed qualification case.

    ``builder`` returns the request shape described in the module
    docstring; ``domain`` is the predicate over the parsed result that
    the verdict layer must satisfy; ``mode`` is one of ``"sample"``,
    ``"estimate"``, ``"variational"``, or ``"transpile"``.
    """

    algorithm: str
    builder: Callable[[], Any]
    domain: Callable[[Any], bool]
    mode: str
    required_capabilities: tuple[str, ...]
    shots: int
    threshold: float
    seed: int
    max_attempts: int
    timeout: float
    fake_backend_required: bool = False


# ---------------------------------------------------------------------------
# Builders (import-safe and fast: circuit construction only).
# ---------------------------------------------------------------------------


def _bell_builder():
    """2-qubit Bell state with measurement attached (smoke check)."""
    prog = QProg()
    prog << H(0) << X(1).control(0)
    prog << measure_all([0, 1], [0, 1])
    return prog


def _grover_builder():
    """Grover search for |11> on two qubits, measured."""
    prog = QProg()
    prog << Grover(mark_data="11").cir(q_input=[0, 1], iternum=1)
    prog << measure_all([0, 1], [0, 1])
    return prog


def _qaoa_invoke(backend):
    """One-layer QAOA on a fixed 2-variable QUBO (estimated objective)."""
    x0, x1 = sp.symbols("x0 x1")
    model = QAOA(-0.5 * x0 * x1 + 0.5 * x0 - 0.25 * x1)
    return model.run(
        layer=1,
        initial_para=_QAOA_INITIAL_PARA,
        shots=_SHOTS,
        optimizer_option={"options": {"maxiter": 1}},
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qarm_invoke(backend):
    """QARM association-rule mining over the package's data2.txt fixture."""
    data_file = os.path.join(QARM.__path__[0], "dataset", "data2.txt")
    with open(data_file, "r", encoding="utf8") as handle:
        transactions = [
            [item.strip() for item in line.strip().split(",")]
            for line in handle
            if line
        ]
    miner = QuantumAssociationRulesMining(transactions, 0.2, 0.5)
    return miner.run(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qkmeans_invoke(backend):
    """k=2 quantum k-means over four well-separated 2-D points."""
    return QuantumKmeans(k=2).fit(
        _SAMPLE_POINTS,
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_LONG_TIMEOUT),
    )


def _qpca_invoke(backend):
    """1-component QPCA over the four fixed 2-D points."""
    return qpca(
        _SAMPLE_POINTS,
        1,
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qsvm_invoke(backend):
    """2x2 quantum kernel matrix over the fixed two-point set."""
    return QuantumKernel_vqnet(n_qbits=2).evaluate(
        _SAMPLE_X,
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qsvr_invoke(backend):
    """Quantum SVR regression on the fixed two-point set."""
    return Quantum_SVR(_SAMPLE_X, _SAMPLE_Y).get_res(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qubo_qaoa_invoke(backend):
    """QUBO_QAOA over the fixed 3-variable QUBO (estimated objective)."""
    solver = QUBO_QAOA(_QUBO_FUNCTION)
    return solver.run(
        layer=1,
        optimizer_option={"options": {"maxiter": 1}},
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qubo_gas_invoke(backend):
    """GAS search for the fixed QUBO's unique minimum -1.0 at "010"."""
    solver = QUBO.QUBO_GAS_origin(_QUBO_FUNCTION)
    return solver.run(
        continue_times=2,
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qae_invoke(backend):
    """QAE on a single-qubit amplitude sin(pi/3): p = 0.75."""
    def operator(qlist):
        cir = QCircuit()
        cir << RY(qlist[0], np.pi / 3)
        return cir

    qae = QAE(
        operator_in=operator,
        qnumber=1,
        epsilon=0.01,
        res_index=[0],
        target_state="1",
    )
    return qae.run(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qsvd_invoke(backend):
    """QSVD overlap-cost optimization of diag(1, 0.5) with seeded initial
    parameters (the algorithm's initial parameter draw is unseeded)."""
    solver = SVD(matrix_in=np.array([[1.0, 0.0], [0.0, 0.5]]), depth=4)
    rng = np.random.RandomState(7)
    solver.parameter = 0.5 * np.pi * rng.random((solver.q0 + solver.q1) * solver.iter_depth)
    solver.QSVD_min(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
        maxiter=1,
    )
    return solver.loss_value, solver.loss_uncertainty


def _gas_invoke(backend):
    """GAS search minimizing -(x0*x1) over two bits, deterministic
    rotation policy (the 'random' rotation draws are unseeded)."""
    searcher = GroverAdaptiveSearch(init_value=0, n_index=2)

    def value_function(key):
        x0, x1 = int(key[0]), int(key[1])
        return -(x0 * x1)

    return searcher.run(
        continue_times=3,
        n_value_function=lambda current_min: 1,
        value_function=value_function,
        rotation_change="increase",
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _qmrmr_invoke(backend):
    """QmRMR feature selection over a fixed 2-feature problem; the
    algorithm's internal SPSA seeds itself with np.random.seed(1234)."""
    linear = np.array([0.6, 0.4])
    quadratic = [[0.3, 0.1], [0.1, 0.2]]
    model = Feature_Selection(quadratic, linear, 1)
    return model.get_his_res(
        [0.5, 0.5],
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_LONG_TIMEOUT),
    )


def _qsen_code_invoke(backend):
    """State preparation of |+>|+> reported as a probability list."""
    return QSpare_Code([0.5, 0.5], cut_length=2).Quantum_Res(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _vqe_invoke(backend):
    """VQE ground state of H = Z0 (exact energy -1.0)."""
    solver = VQE(Hamiltonian({"Z0": 1.0}))
    return solver.run(
        initial_parameters=np.array([0.2, 0.0]),
        config=VQEConfig(max_iterations=80, tolerance=1e-6),
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
    )


def _hhl_invoke(backend):
    """HHL on diag(1, 2) with b = [1, 1]: success probability plus one
    data observable -- no tomography on the QPU qualification path."""
    solver = HHL(np.diag([1.0, 2.0]), np.array([1.0, 1.0]))
    return solver.run(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_TIMEOUT),
        observables=["Z0"],
    )


def _shor_invoke(backend):
    """Shor factorization of 15; the base is drawn from the committed
    RNG and the committed attempt budget, so no factor/order is encoded."""
    solver = Shor(
        15,
        rng=random.Random(42),
        config=ShorConfig(max_attempts=6),
    )
    return solver.run(
        backend=backend,
        execution_options=ExecutionOptions(shots=_SHOTS, timeout=_LONG_TIMEOUT),
    )


# ---------------------------------------------------------------------------
# Domain predicates (gates over the parsed result).
# ---------------------------------------------------------------------------


def _binary_distribution(parsed):
    """True for a nonempty dict whose keys are binary strings."""
    return (
        isinstance(parsed, dict)
        and bool(parsed)
        and all(set(key) <= {"0", "1"} for key in parsed)
    )


# ---------------------------------------------------------------------------
# The fixed case inventories.
# ---------------------------------------------------------------------------

#: The 17 fixed QPU qualification cases, one per executable algorithm.
QUALIFICATION_CASES: tuple[QualificationCase, ...] = (
    QualificationCase(
        algorithm="QAOA",
        builder=lambda: _qaoa_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 3
            and _binary_distribution(parsed[0])
            and math.isfinite(parsed[2])
        ),
        mode="estimate",
        required_capabilities=("estimation", "sampling"),
        shots=_SHOTS,
        threshold=0.9,
        seed=12,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QARM",
        builder=lambda: _qarm_invoke,
        domain=lambda parsed: (
            isinstance(parsed, dict)
            and bool(parsed)
            and all(0.0 <= value <= 1.0 for value in parsed.values())
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=11,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QKmeans",
        builder=lambda: _qkmeans_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 2
            and parsed[0].shape == (2, 2)
            and parsed[1].shape == (4,)
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=6,
        max_attempts=1,
        timeout=_LONG_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QPCA",
        builder=lambda: _qpca_invoke,
        domain=lambda parsed: (
            isinstance(parsed, np.ndarray)
            and parsed.shape == (4, 1)
            and np.all(np.isfinite(parsed))
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=8,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QSVM",
        builder=lambda: _qsvm_invoke,
        domain=lambda parsed: (
            isinstance(parsed, np.ndarray)
            and parsed.shape == (2, 2)
            and np.all(np.isfinite(parsed))
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=9,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QUBO_QAOA",
        builder=lambda: _qubo_qaoa_invoke,
        domain=_binary_distribution,
        mode="estimate",
        # Plan 3 gap: the final-distribution path needs statevector, which
        # runtime backends do not advertise.  Declared honestly here.
        required_capabilities=("estimation", "statevector"),
        shots=_SHOTS,
        threshold=0.9,
        seed=13,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QUBO_GAS",
        builder=lambda: _qubo_gas_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 2
            and isinstance(parsed[0], list)
            and bool(parsed[0])
            and isinstance(parsed[0][0], list)
            and parsed[1] <= -0.99  # the fixed QUBO's unique minimum is -1.0
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=3,
        max_attempts=2,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QAE",
        builder=lambda: _qae_invoke,
        domain=lambda parsed: (
            isinstance(parsed, (int, float)) and 0.0 <= parsed <= 1.0
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.05,  # |p_estimated - 0.75| tolerance
        seed=4,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QSVD",
        builder=lambda: _qsvd_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 2
            and 0.0 <= parsed[0] <= 1.0
            and math.isfinite(parsed[1])
        ),
        mode="estimate",
        required_capabilities=("estimation",),
        shots=_SHOTS,
        threshold=0.9,
        seed=7,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QSVR",
        builder=lambda: _qsvr_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 2
            and parsed[1].shape == (2,)
            and np.all(np.isfinite(parsed[0]))
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=10,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="Grover",
        builder=_grover_builder,
        domain=lambda parsed: bool(parsed) and max(parsed, key=parsed.get) == "11",
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,  # P(11) floor; nominal success probability is 1.0
        seed=1,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="GroverAdaptiveSearch",
        builder=lambda: _gas_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 2
            and isinstance(parsed[0], list)
            and math.isfinite(parsed[1])
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,
        seed=2,
        max_attempts=3,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QmRMR",
        builder=lambda: _qmrmr_invoke,
        domain=lambda parsed: (
            isinstance(parsed, tuple)
            and len(parsed) == 3
            and bool(parsed[0])
            and isinstance(parsed[1], list)
            and len(parsed[1]) == 2
            and sum(parsed[1]) == 1
        ),
        mode="estimate",
        # Plan 3 gap: get_his_res unconditionally needs statevector for the
        # theory final distribution; runtime backends do not advertise it.
        required_capabilities=("estimation", "statevector"),
        shots=_SHOTS,
        threshold=0.9,
        seed=1234,
        max_attempts=1,
        timeout=_LONG_TIMEOUT,
    ),
    QualificationCase(
        algorithm="QSEncode",
        builder=lambda: _qsen_code_invoke,
        domain=lambda parsed: (
            isinstance(parsed, list)
            and len(parsed) == 2
            and all(0.0 <= p <= 1.0 for p in parsed)
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.05,  # per-component |p_i - 0.5| tolerance
        seed=5,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="VQE",
        builder=lambda: _vqe_invoke,
        domain=lambda parsed: abs(parsed.energy + 1.0) <= 0.05,
        mode="variational",
        required_capabilities=("variational_session",),
        shots=_SHOTS,
        threshold=0.05,  # |E - E_ground| tolerance, E_ground = -1.0
        seed=14,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="HHL",
        builder=lambda: _hhl_invoke,
        domain=lambda parsed: (
            parsed.success_probability is not None
            and parsed.success_probability >= 0.05
            and bool(parsed.metadata)
            and "observables" in parsed.metadata
            and math.isfinite(float(parsed.metadata["observables"][0]))
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.05,  # success-probability floor
        seed=15,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="Shor",
        builder=lambda: _shor_invoke,
        domain=lambda parsed: (
            parsed["factors"] is not None and set(parsed["factors"]) == {3, 5}
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,  # confidence floor on the {3, 5} factorization
        seed=42,
        max_attempts=6,
        timeout=_LONG_TIMEOUT,
    ),
)

#: Smoke cases: fixed checks that must pass on the FakeBackend before any
#: QPU run.  The bell case lives here -- not in QUALIFICATION_CASES -- so
#: the 17-name completeness test stays exact.
SMOKE_CASES: tuple[QualificationCase, ...] = (
    QualificationCase(
        algorithm="bell",
        builder=_bell_builder,
        domain=lambda parsed: (
            isinstance(parsed, dict) and {"00", "11"} <= set(parsed)
        ),
        mode="sample",
        required_capabilities=("sampling",),
        shots=_SHOTS,
        threshold=0.9,  # P(00) + P(11) floor
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
        fake_backend_required=True,
    ),
)

#: Circuit-only transpilation inventory: backend-free circuits that every
#: candidate device must transpile before QPU execution.
TRANSPILATION_CASES: tuple[QualificationCase, ...] = (
    QualificationCase(
        algorithm="qcmp-int-comparator",
        builder=lambda: int_comparator(2, 1, [0, 1, 2]),
        domain=lambda circuit: isinstance(circuit, QCircuit),
        mode="transpile",
        required_capabilities=(),
        shots=1,
        threshold=1.0,
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="qcmp-qft-comparator",
        builder=lambda: qft_comparator(2, [0, 1], [2], function="g"),
        domain=lambda circuit: isinstance(circuit, QCircuit),
        mode="transpile",
        required_capabilities=(),
        shots=1,
        threshold=1.0,
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="qaoa-xy-mixer",
        builder=lambda: xy_mixer([0, 1], 0.25),
        domain=lambda circuit: isinstance(circuit, QCircuit),
        mode="transpile",
        required_capabilities=(),
        shots=1,
        threshold=1.0,
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="qaoa-dicke-state",
        builder=lambda: prepare_dicke_state([0, 1, 2, 3], 2),
        domain=lambda circuit: isinstance(circuit, QCircuit),
        mode="transpile",
        required_capabilities=(),
        shots=1,
        threshold=1.0,
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
    QualificationCase(
        algorithm="plugin-qft",
        builder=lambda: QFT([0, 1, 2]),
        domain=lambda circuit: isinstance(circuit, QCircuit),
        mode="transpile",
        required_capabilities=(),
        shots=1,
        threshold=1.0,
        seed=0,
        max_attempts=1,
        timeout=_TIMEOUT,
    ),
)


def case_by_name(name: str) -> QualificationCase:
    """Return the case named ``name`` across all inventories.

    Resolves ``QUALIFICATION_CASES``, then ``SMOKE_CASES``, then
    ``TRANSPILATION_CASES``; raises :class:`KeyError` for unknown names.
    """
    for case in (*QUALIFICATION_CASES, *SMOKE_CASES, *TRANSPILATION_CASES):
        if case.algorithm == name:
            return case
    raise KeyError(f"no qualification case named {name!r}")
