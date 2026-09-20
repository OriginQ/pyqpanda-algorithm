# Bernstein–Vazirani hidden-bitstring algorithm

This module adds the Bernstein–Vazirani algorithm to pyqpanda_alg. It recovers
an unknown bit string s from an oracle f(x) = s · x XOR b with a single quantum
query. A deterministic classical black-box strategy needs one query per input
bit.

## Public API

Import BernsteinVazirani, build_phase_oracle, or evaluate_oracle from
pyqpanda_alg.BernsteinVazirani.

The high-level class owns the hidden-string contract, builds the native
PyQPanda3 circuit, executes it on CPUQVM by default, and returns both the
decoded bit string and the backend probability map.

Example:

    solver = BernsteinVazirani("101101")
    result = solver.run()
    print(result.measured_secret)
    print(result.success_probability)

For integration into a larger circuit:

    circuit = solver.build_circuit(
        input_qubits=[4, 5, 6, 7, 8, 9],
        ancilla=10,
    )

The first character of the secret corresponds to the first logical input
qubit. Result dictionaries are normalised with the repository's
parse_quantum_result_dict helper so decoded strings follow that logical order.

## Oracle contract

evaluate_oracle(secret, input_bits, bias) is a pure-Python reference for
s · x XOR b. build_phase_oracle emits the reversible operation
|x,y> -> |x,y XOR f(x)>. In the Bernstein–Vazirani circuit the target is
prepared in |->, turning the oracle answer into phase kickback.

## Circuit

For an n-bit secret the implementation uses n data qubits plus one ancilla:

1. prepare the ancilla in |->;
2. prepare every data qubit in |+>;
3. query the hidden-string oracle once;
4. apply Hadamard to each data qubit;
5. read the data register as s.

The optional constant bias contributes only a global phase while the ancilla is
in |-> and therefore does not alter the recovered hidden string.

See example/QAlgBase/testeg_bernstein_vazirani.py for a runnable example.
