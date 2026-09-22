from pyqpanda_alg.QAOA.qaoa import _pauli_term_data


class _Pq2OnlyPauliOperator:
    """Minimal stand-in for QPanda3 releases without ``PauliOperator.terms``."""

    def to_hamiltonian_pq2(self):
        return [
            ({0: "Z", 1: "I"}, 2.0 + 0.0j),
            ({0: "I", 1: "Z"}, 3.0 + 0.0j),
        ]


def test_pauli_term_adapter_supports_qpanda3_pq2_export():
    assert _pauli_term_data(_Pq2OnlyPauliOperator()) == [
        (2.0 + 0.0j, ((0, "Z"),)),
        (3.0 + 0.0j, ((1, "Z"),)),
    ]

