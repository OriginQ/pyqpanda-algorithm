import pytest

from pyqpanda_alg import QOracleLearning as q


def test_truth_table_validation_and_promise():
    assert q.classify_deutsch_jozsa_promise([0, 0, 0, 0]) == "constant"
    assert q.classify_deutsch_jozsa_promise([1, 1, 1, 1]) == "constant"
    assert q.classify_deutsch_jozsa_promise([0, 1, 1, 0]) == "balanced"
    with pytest.raises(ValueError, match="constant or balanced"):
        q.classify_deutsch_jozsa_promise([0, 0, 0, 1])
    for bad in ([0], [0, 1, 0], [0, 2], []):
        with pytest.raises(ValueError):
            q.validate_truth_table(bad)


def test_bit_order_round_trip_exhaustive():
    for width in range(1, 7):
        for index in range(2**width):
            bits = q.index_to_q0_bits(index, width)
            assert q.q0_bits_to_index(bits) == index


def test_phase_oracle_diagonal_matches_truth_table():
    table = (0, 1, 1, 0, 1, 0, 0, 1)
    assert q.phase_oracle_diagonal(table) == (1, -1, -1, 1, -1, 1, 1, -1)


def test_deutsch_jozsa_walsh_reference():
    for table in ((0,) * 8, (1,) * 8):
        probs = q.walsh_probabilities(table)
        assert probs[0] == pytest.approx(1.0)
        assert sum(probs[1:]) == pytest.approx(0.0)
    balanced = q.affine_truth_table("101", bias=0)
    probs = q.walsh_probabilities(balanced)
    assert probs[0] == pytest.approx(0.0)
    assert sum(probs) == pytest.approx(1.0)


def test_all_bv_secrets_through_five_qubits_both_biases():
    for width in range(1, 6):
        for secret_index in range(2**width):
            secret = "".join(str(bit) for bit in q.index_to_q0_bits(secret_index, width))
            for bias in (0, 1):
                table = q.affine_truth_table(secret, bias)
                assert q.recover_affine_secret(table) == secret
                probs = q.walsh_probabilities(table)
                assert probs[secret_index] == pytest.approx(1.0)
                assert sum(probs) == pytest.approx(1.0)
                assert q.dominant_q0_bitstring(probs, width) == secret


def test_recover_secret_rejects_non_affine_table():
    with pytest.raises(ValueError, match="not affine"):
        q.recover_affine_secret((0, 1, 1, 1))


def test_secret_and_probability_validation():
    for secret in ("", "10x", [1, 2]):
        with pytest.raises(ValueError):
            q.affine_truth_table(secret)
    with pytest.raises(ValueError):
        q.affine_truth_table("10", bias=2)
    with pytest.raises(ValueError):
        q.dominant_q0_bitstring([0.5, 0.5], 1)


def test_cpuqvm_bv_end_to_end():
    for secret in ("0", "1", "1011", "01101"):
        recovered, probs = q.run_bernstein_vazirani(secret)
        assert recovered == secret
        assert max(probs) == pytest.approx(1.0, abs=1e-9)


def test_cpuqvm_deutsch_jozsa_end_to_end():
    for table, expected in (
        ((0,) * 8, "constant"),
        ((1,) * 8, "constant"),
        (q.affine_truth_table("101"), "balanced"),
    ):
        result, probs = q.run_deutsch_jozsa(table)
        assert result == expected
        if expected == "constant":
            assert probs[0] == pytest.approx(1.0, abs=1e-9)
        else:
            assert probs[0] == pytest.approx(0.0, abs=1e-9)
