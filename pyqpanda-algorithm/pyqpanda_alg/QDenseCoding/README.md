# Superdense Coding

This module implements the Bennett-Wiesner superdense-coding protocol as a reusable PyQPanda3 circuit builder.

Two parties share a Bell pair. Alice applies one of four Pauli encodings to her qubit, transmits that single qubit, and Bob decodes both qubits to recover two classical bits.

## Bit convention

For an ordered qubit pair [alice, bob] and message b0b1:

| Message | Alice operation |
| --- | --- |
| 00 | identity |
| 01 | X |
| 10 | Z |
| 11 | Z then X |

The full protocol is:

1. prepare (|00> + |11>)/sqrt(2) with H + CNOT;
2. encode b0b1 on Alice's qubit;
3. decode with CNOT + H;
4. measure [alice, bob] in that same order to recover b0b1.

## Usage

    from pyqpanda3.core import CPUQVM, QProg
    from pyqpanda_alg.QDenseCoding import SuperdenseCoding

    protocol = SuperdenseCoding("11")
    program = QProg(2)
    qubits = program.qubits()
    program << protocol.cir(qubits)

    machine = CPUQVM()
    machine.run(program, 256)
    probabilities = machine.result().get_prob_dict(qubits)
    print(max(probabilities, key=probabilities.get))

The lower-level helpers prepare_bell_pair, encode_message, decode_message, and superdense_coding_circuit are exported for applications that need to insert transport or noise operations between protocol stages.
