import pytest

from pyqpanda_alg import Simon as q


def _secret(index, width):
    return "".join(str(bit) for bit in q.index_to_q0_bits(index, width))


def _xor_bits(left, right):
    return tuple(a ^ b for a, b in zip(left, right))


def test_linear_oracle_has_exact_hidden_xor_fibers_through_five_qubits():
    for width in range(1, 6):
        for secret_index in range(1, 2**width):
            secret = _secret(secret_index, width)
            secret_bits = tuple(int(bit) for bit in secret)
            fibers = {}
            for x_index in range(2**width):
                x = q.index_to_q0_bits(x_index, width)
                output = q.linear_simon_function(x, secret)
                partner = _xor_bits(x, secret_bits)
                assert q.linear_simon_function(partner, secret) == output
                fibers.setdefault(output, []).append(x)
            assert len(fibers) == 2 ** (width - 1)
            assert {len(preimages) for preimages in fibers.values()} == {2}


def test_reference_distribution_and_exact_recovery_all_secrets_through_five_qubits():
    for width in range(1, 6):
        for secret_index in range(1, 2**width):
            secret = _secret(secret_index, width)
            probabilities = q.simon_reference_probabilities(secret)
            assert sum(probabilities) == pytest.approx(1.0)
            support = [
                q.index_to_q0_bits(index, width)
                for index, probability in enumerate(probabilities)
                if probability > 0
            ]
            assert len(support) == 2 ** (width - 1)
            assert all(q.dot_mod2(bits, secret) == 0 for bits in support)
            assert q.recover_secret_from_probabilities(probabilities, width) == secret


def test_gf2_rank_basis_and_fail_closed_undersampling():
    equations = ("010", "101", "111", "000")
    assert q.gf2_rank(equations) == 2
    basis = q.independent_equations(equations)
    assert len(basis) == 2
    assert q.gf2_rank(basis) == 2
    assert q.recover_secret_from_equations(equations) == "101"

    with pytest.raises(ValueError, match="requires rank 2"):
        q.recover_secret_from_equations(("010",), width=3)
    with pytest.raises(ValueError, match="same width"):
        q.recover_secret_from_equations(("01", "001"))


def test_counts_decoder_requires_explicit_display_order():
    # Secret 110 has orthogonal q0-first equations 001 and 110.
    q0_counts = {"001": 7, "110": 5, "000": 100}
    assert q.recover_secret_from_counts(q0_counts, 3, key_order="q0-first") == "110"

    # The same equations as conventional qN-first display strings.
    qn_counts = {"100": 7, "011": 5, "000": 100}
    assert q.recover_secret_from_counts(qn_counts, 3, key_order="qN-first") == "110"

    with pytest.raises(ValueError, match="key_order"):
        q.recover_secret_from_counts(q0_counts, 3, key_order="guess")


def test_validation_rejects_broken_promises_and_shapes():
    for secret in ("", "000", "10x", [1, 2]):
        with pytest.raises(ValueError):
            q.simon_reference_probabilities(secret)
    with pytest.raises(ValueError, match="equal width"):
        q.linear_simon_function("10", "101")
    with pytest.raises(ValueError, match="equal width"):
        q.dot_mod2("10", "1")
    with pytest.raises(ValueError, match="length"):
        q.equations_from_probability_list((0.5, 0.5), 2)


def test_probability_support_is_exact_orthogonal_subspace():
    secret = "10110"
    probabilities = q.simon_reference_probabilities(secret)
    equations = q.equations_from_probability_list(probabilities, len(secret))
    assert q.gf2_rank(equations, len(secret)) == len(secret) - 1
    assert all(q.dot_mod2(row, secret) == 0 for row in equations)
    assert q.recover_secret_from_equations(equations, len(secret)) == secret


def test_cpuqvm_end_to_end_when_pyqpanda3_is_available():
    pytest.importorskip("pyqpanda3")
    for secret in ("1", "10", "101", "0110"):
        recovered, probabilities = q.run_simon(secret)
        assert recovered == secret
        assert probabilities == pytest.approx(q.simon_reference_probabilities(secret), abs=1e-9)
