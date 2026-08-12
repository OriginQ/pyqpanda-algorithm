"""HHL circuit synthesis: the quantum subroutine behind the solver.

:func:`build_hhl_circuit` validates and reversibly pads a linear system
exactly as :func:`normalize_linear_system` does, then synthesizes the
textbook HHL subroutine on three registers: an amplitude-encoded
``|b>`` data register, a phase-estimation register, and one success
ancilla.  The returned :class:`HHLCircuitBuild` carries the program
together with the register layout and the classical scale metadata
(evolution time, reciprocal scale, eigenvalue bounds) so a solver can
post-select and rescale the outcome without re-deriving them.

Circuit layout
--------------
* ``data_qubits``: ``log2(padded_dimension)`` qubits holding ``|b>``,
  prepared with pyqpanda3's amplitude encoder.
* ``phase_qubits``: ``config.phase_qubits`` qubits for QPE.
* ``success_qubit``: one ancilla post-selected on ``|1>``.

Evolution time and reciprocal scale
-----------------------------------
The unitary evolved is ``U = exp(i A t)`` with ``t = pi / lambda_max``
where ``lambda_max`` is the largest eigenvalue magnitude of the padded
matrix.  Every eigenvalue phase ``lambda t`` then lies in ``[-pi, pi]``,
so the QPE phase fraction ``phi = (lambda t mod 2pi)/(2pi)`` recovers
the signed eigenvalue ``lambda = (2pi/t) * signed(phi)`` with
``signed(phi) = phi`` for ``phi <= 1/2`` and ``phi - 1`` otherwise.
At the boundary fraction ``1/2``, the synthesis uses the signed
eigenvalue of largest magnitude from the validated spectrum, avoiding
the otherwise ambiguous ``+/- lambda_max`` interpretation.
The reciprocal rotation applies angle ``theta_j = 2 arcsin(C/lambda_j)``
for phase-register value ``j``, with the scale ``C = min(|lambda|)``
recorded as ``reciprocal_scale``; the success amplitude is then exactly
``C/lambda_j``, so post-selecting the ancilla on ``|1>`` leaves the
data register proportional to ``A^{-1}|b>``.  A phase estimate at
``lambda ~ 0`` (``j = 0``) cannot be inverted and is left unrotated;
validated systems are invertible, so such a branch carries no weight
of the solution.

Synthesis
---------
State preparation is a pyqpanda3 amplitude encoding of the normalized
padded vector.  QPE applies Hadamards, then one controlled
``Oracle(exp(i A t 2**k))`` per phase qubit ``k``, then an inverse QFT
built from Hadamard and controlled-phase gates.  The reciprocal
rotation is a block-diagonal ``Oracle`` over (phase register, success)
whose ``j``-th block is ``RY(theta_j)``.  The phase-estimation
operations are then reversed explicitly — QFT, adjoint controlled
evolutions in reverse order, Hadamards — so the phase register returns
to ``|0...0>`` before post-selection.  The circuit is measurement-free.
"""

from dataclasses import dataclass

import numpy as np
from pyqpanda3.core import CR, Encode, H, Oracle, QProg, SWAP
from scipy.linalg import expm

from pyqpanda_alg.execution import AlgorithmInputError

from .model import HHLConfig
from .validation import normalize_linear_system

#: Padded eigenvalue magnitude below which a phase estimate is treated
#: as uninvertible (the ``j = 0`` branch of the QPE register).
_ZERO_EIGENVALUE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class HHLCircuitBuild:
    """Immutable HHL circuit build with its register and scale metadata.

    ``program`` is the measurement-free :class:`QProg` implementing the
    full HHL subroutine.  ``data_qubits``, ``phase_qubits``, and
    ``success_qubit`` name the register layout inside the program.
    ``evolution_time`` and ``reciprocal_scale`` are the classical
    ``t`` and ``C`` used to build the circuit (see the module
    docstring), and ``eigenvalue_bounds`` is the ``(min, max)``
    spectrum of the padded system matrix; together they let a solver
    reconstruct and rescale the post-selected solution.
    """

    program: QProg
    data_qubits: tuple[int, ...]
    phase_qubits: tuple[int, ...]
    success_qubit: int
    evolution_time: float
    reciprocal_scale: float
    eigenvalue_bounds: tuple[float, float]


def build_hhl_circuit(
    matrix, vector, config: HHLConfig
) -> HHLCircuitBuild:
    """Synthesize the HHL circuit for a validated linear system.

    ``matrix`` and ``vector`` are validated and padded exactly as in
    :func:`normalize_linear_system`; a system whose padded dimension is
    one (a single row) leaves no data qubit and is rejected.  The
    returned build carries the synthesized program, its register
    layout, and the evolution-time/reciprocal-scale metadata chosen
    from the padded spectrum.
    """
    system = normalize_linear_system(matrix, vector, config)
    data_qubits = system.padded_dimension.bit_length() - 1
    if data_qubits < 1:
        raise AlgorithmInputError(
            "a 1-by-1 system leaves no data qubit; "
            "HHL needs at least a 2-by-2 system"
        )
    phase_qubits = config.phase_qubits
    matrix = system.matrix

    data = tuple(range(data_qubits))
    phase = tuple(range(data_qubits, data_qubits + phase_qubits))
    success_qubit = data_qubits + phase_qubits

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalue_bounds = (float(eigenvalues[0]), float(eigenvalues[-1]))
    evolution_time = np.pi / max(abs(eigenvalues))
    reciprocal_scale = float(min(abs(eigenvalues)))
    boundary_eigenvalue = (
        float(eigenvalues[0])
        if abs(eigenvalues[0]) > abs(eigenvalues[-1])
        else float(eigenvalues[-1])
    )

    program = QProg(data_qubits + phase_qubits + 1)
    _prepare_state(program, data, system.vector)
    _phase_estimation(program, data, phase, matrix, evolution_time)
    _reciprocal_rotation(
        program,
        phase,
        success_qubit,
        evolution_time,
        reciprocal_scale,
        boundary_eigenvalue,
    )
    _inverse_phase_estimation(program, data, phase, matrix, evolution_time)

    return HHLCircuitBuild(
        program=program,
        data_qubits=data,
        phase_qubits=phase,
        success_qubit=success_qubit,
        evolution_time=evolution_time,
        reciprocal_scale=reciprocal_scale,
        eigenvalue_bounds=eigenvalue_bounds,
    )


def _prepare_state(program: QProg, data_qubits, vector) -> None:
    """Amplitude-encode the unit-norm padded ``|b>`` into the data register."""
    state = np.asarray(vector, dtype=np.complex128)
    state = state / np.linalg.norm(state)
    encode = Encode()
    encode.amplitude_encode(list(data_qubits), state.tolist())
    program << encode.get_circuit()


def _phase_estimation(program, data_qubits, phase_qubits, matrix, evolution_time) -> None:
    """Apply Hadamards, controlled ``exp(i A t 2**k)``, and the inverse QFT."""
    for qubit in phase_qubits:
        program << H(qubit)
    for index, qubit in enumerate(phase_qubits):
        program << _evolution_gate(data_qubits, qubit, matrix, evolution_time, index)
    _iqft(program, list(phase_qubits))


def _inverse_phase_estimation(
    program, data_qubits, phase_qubits, matrix, evolution_time
) -> None:
    """Reverse the phase-estimation operations explicitly.

    The adjoint of QPE is the QFT, then the adjoint controlled
    evolutions in reverse order, then the Hadamards; this returns the
    phase register to ``|0...0>`` before post-selection.
    """
    _qft(program, list(phase_qubits))
    for index in range(len(phase_qubits) - 1, -1, -1):
        program << _evolution_gate(
            data_qubits, phase_qubits[index], matrix, -evolution_time, index
        )
    for qubit in phase_qubits:
        program << H(qubit)


def _evolution_gate(data_qubits, control, matrix, evolution_time, power):
    """Controlled ``exp(i A t 2**power)`` on the data register."""
    unitary = expm(1j * matrix * evolution_time * (1 << power))
    return Oracle(list(data_qubits), unitary).control(control)


def _reciprocal_rotation(
    program,
    phase_qubits,
    success_qubit,
    evolution_time,
    reciprocal_scale,
    boundary_eigenvalue,
) -> None:
    """Rotate the success qubit by ``2 arcsin(C / lambda_j)`` per phase state.

    The rotation is the block-diagonal unitary whose ``j``-th block is
    ``RY(theta_j)`` on the success qubit, conditioned on the phase
    register reading ``j``; it is applied as one Oracle over the phase
    register and the success qubit.
    """
    dimension = 1 << len(phase_qubits)
    rotation = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    for j in range(dimension):
        theta = _rotation_angle(
            j / dimension,
            evolution_time,
            reciprocal_scale,
            boundary_eigenvalue,
        )
        cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
        rotation[j, j] = cosine
        rotation[j + dimension, j] = sine
        rotation[j, j + dimension] = -sine
        rotation[j + dimension, j + dimension] = cosine
    program << Oracle(list(phase_qubits) + [success_qubit], rotation)


def _rotation_angle(
    phase_fraction, evolution_time, reciprocal_scale, boundary_eigenvalue
) -> float:
    """Inversion angle for the eigenvalue estimated from a phase fraction.

    The fraction is mapped back to the signed eigenvalue with the
    ``t = pi / lambda_max`` convention of the module docstring, and the
    rotation angle is ``2 arcsin(C / lambda)`` clamped to the physical
    range.  A fraction estimating a zero eigenvalue returns no
    rotation.
    """
    if phase_fraction == 0.5:
        eigenvalue = boundary_eigenvalue
    else:
        signed = phase_fraction if phase_fraction < 0.5 else phase_fraction - 1.0
        eigenvalue = (2 * np.pi / evolution_time) * signed
    if abs(eigenvalue) < _ZERO_EIGENVALUE_TOLERANCE:
        return 0.0
    ratio = np.clip(reciprocal_scale / eigenvalue, -1.0, 1.0)
    return 2 * np.arcsin(ratio)


def _qft(program: QProg, qubits) -> None:
    """Standard quantum Fourier transform on ``qubits`` (H + CP gates)."""
    size = len(qubits)
    for i in range(size - 1, -1, -1):
        program << H(qubits[i])
        for j in range(i - 1, -1, -1):
            program << CR(qubits[j], qubits[i], np.pi / 2 ** (i - j))
    for i in range(size // 2):
        program << SWAP(qubits[i], qubits[size - 1 - i])


def _iqft(program: QProg, qubits) -> None:
    """Inverse quantum Fourier transform (adjoint of :func:`_qft`)."""
    size = len(qubits)
    for i in range(size // 2):
        program << SWAP(qubits[i], qubits[size - 1 - i])
    for i in range(size):
        for j in range(i):
            program << CR(qubits[j], qubits[i], -np.pi / 2 ** (i - j))
        program << H(qubits[i])
