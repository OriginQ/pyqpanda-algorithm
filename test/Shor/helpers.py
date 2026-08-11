"""Shared harness for basis-state permutation testing of Shor circuits.

``run_basis_permutation`` prepares the little-endian basis value on the
circuit's declared value register, sets the declared control qubit (or
every exponent qubit) to the requested value, applies the circuit, and
samples the whole register once on :class:`LocalBackend`.  It decodes
the value register from the measurement and asserts that every declared
ancilla qubit returns to zero.
"""

from pyqpanda3.core import QProg, X, measure

from pyqpanda_alg.execution import ExecutionOptions, LocalBackend


def run_basis_permutation(circuit, control, value):
    """Apply ``circuit`` to the basis state and return the value register.

    ``value`` is prepared little-endian on ``circuit.value_qubits`` and
    the control qubit(s) of ``circuit`` are set by ``control``: the
    single ``control_qubit`` is set when ``control`` is truthy, while an
    exponentiation circuit treats ``control`` as a bit mask over
    ``exponent_qubits`` (bit ``k`` sets exponent qubit ``k``), so every
    exponent value is addressable per-bit.  Every qubit is then measured
    once; the value register is decoded little-endian from the
    measurement and returned, and all ``circuit.ancilla_qubits`` are
    asserted to read zero.
    """
    value_qubits = list(circuit.value_qubits)
    ancilla_qubits = list(circuit.ancilla_qubits)
    control_qubit = getattr(circuit, "control_qubit", None)
    exponent_qubits = list(circuit.exponent_qubits or ())

    measured = sorted(set(value_qubits) | set(ancilla_qubits) | set(exponent_qubits))
    if control_qubit is not None:
        measured.append(control_qubit)
        measured = sorted(set(measured))

    prog = QProg()
    for k, qubit in enumerate(value_qubits):
        if (value >> k) & 1:
            prog << X(qubit)
    if exponent_qubits:
        for k, qubit in enumerate(exponent_qubits):
            if (control >> k) & 1:
                prog << X(qubit)
    elif control_qubit is not None and control:
        prog << X(control_qubit)
    prog << circuit
    prog << measure(measured, measured)

    counts = LocalBackend().submit_sample(
        prog, options=ExecutionOptions(shots=1)
    ).result().single_counts()
    (key,) = counts

    observed = 0
    for k, qubit in enumerate(value_qubits):
        observed |= int(key[len(key) - 1 - measured.index(qubit)]) << k
    for qubit in ancilla_qubits:
        assert key[len(key) - 1 - measured.index(qubit)] == "0", (
            f"ancilla qubit {qubit} not restored to |0>"
        )
    return observed
