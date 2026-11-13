# Licensed under the Apache License, Version 2.0.
# The structured public surface is preserved from tokenjunkielabs PR #81;
# all circuit construction and native execution delegate to the earlier
# 1446170411 PR #34 engine, extended in qpe.py.
"""Explicit adapter for PR #81's incompatible QPE constructor/result API."""

from pyqpanda3.core import CPUQVM, QProg

from .qpe import QPE as CircuitQPE
from .readout import (PhaseEstimationResult, decode_phase, phase_to_eigenvalue,
                      positive_integer)


class StructuredQPE:
    """Structured PR #81 interface using the shared canonical QPE engine.

    ``run()`` now performs the requested measurement shots. ``run_exact()``
    explicitly requests the exact statevector probabilities that PR #81
    returned regardless of shots. Its upper counting-register layout and
    public constructor order are preserved. Import this class explicitly;
    canonical QPE retains PR #34's argument order and float-returning run.
    """

    def __init__(self, unitary, system_qubits, precision_qubits=5,
                 eigenstate_preparation=None, unitary_power=None,
                 shots=2048, machine=None):
        self.unitary = unitary
        self.system_qubits = positive_integer(system_qubits, "system_qubits")
        self.precision_qubits = positive_integer(precision_qubits, "precision_qubits")
        self.eigenstate_preparation = eigenstate_preparation
        self.unitary_power = unitary_power
        self.shots = positive_integer(shots, "shots")
        self.machine = machine if machine is not None else CPUQVM()
        self._engine()  # validate factories without invoking them

    def _engine(self):
        return CircuitQPE(self.unitary, self.precision_qubits, self.system_qubits,
                          self.eigenstate_preparation,
                          unitary_power=self.unitary_power, machine=self.machine)

    def _build(self, engine):
        program = QProg(engine.n_qubits)
        qubits = program.qubits()
        system_register = qubits[:engine.n_target]
        counting_register = qubits[engine.n_target:]
        program << engine._circuit_for_registers(counting_register, system_register)
        return program, counting_register

    def build_program(self):
        """Return the program and upper counting register, as in PR #81."""
        return self._build(self._engine())

    def _run(self, shots):
        engine = self._engine()
        program, counting = self._build(engine)
        probabilities = engine._execute_distribution(program, counting, shots)
        return decode_phase(probabilities, self.precision_qubits,
                            sampling_shots=None if shots == -1 else shots)

    def run(self) -> PhaseEstimationResult:
        """Measure the counting register using the requested positive shots."""
        return self._run(positive_integer(self.shots, "shots"))

    def run_exact(self) -> PhaseEstimationResult:
        """Return exact native statevector probabilities, without sampling."""
        return self._run(-1)


# A module-qualified compatibility import keeps the former class name:
# from pyqpanda_alg.QPE.compat import QPE
QPE = StructuredQPE
