"""BBHT search example: the algorithm is not told the number of marked states."""

from pyqpanda_alg.Grover import BBHTSearch, mark_data_reflection


MARKED_STATES = {"00101", "11100", "10111"}


def oracle(qubits):
    return mark_data_reflection(qubits[:5], sorted(MARKED_STATES))


def is_marked(bitstring):
    return bitstring in MARKED_STATES


search = BBHTSearch(
    n_index=5,
    oracle_circuit=oracle,
    predicate=is_marked,
    seed=2026,
)
result = search.run(process_show=True)
print(result)
