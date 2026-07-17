#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for shared QKD post-processing and probabilistic Eve controls."""

import sys
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QKD import BB84, QKD


class TestQKDBase:

    @pytest.mark.parametrize('eve_prob', [-0.01, 1.01])
    def test_invalid_eve_probability(self, eve_prob):
        with pytest.raises(AssertionError, match='eve_prob'):
            BB84(eve_prob=eve_prob)

    @pytest.mark.parametrize('privacy_ratio', [0.0, -0.1, 1.01])
    def test_invalid_privacy_ratio(self, privacy_ratio):
        with pytest.raises(AssertionError, match='privacy_ratio'):
            BB84(privacy_ratio=privacy_ratio)

    def test_toeplitz_privacy_amplification(self):
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        final_len = 4
        seed = 17

        # Reproduce the same random Toeplitz diagonals independently and perform
        # the GF(2) matrix-vector product explicitly.
        rng = random.Random(seed)
        diagonals = [rng.randrange(0, 2) for _ in range(len(bits) + final_len - 1)]
        expected = [0] * final_len
        for i in range(final_len):
            start = final_len - 1 - i
            diag = diagonals[start:start + len(bits)]
            res = [a & b for a, b in zip(bits, diag)]
            z = res[0]
            for zz in res[1:]:
                z ^= zz
            expected[i] = z

        actual = QKD(seed=seed)._privacy_amplification(bits, final_len)
        assert all(a == b for a, b in zip(actual, expected))

    def test_keygen_hashes_longer_candidate_key(self):
        qkd = BB84(privacy_ratio=0.5, seed=23)
        call = {}

        def privacy_spy(key_bits, final_len):
            call['candidate_len'] = len(key_bits)
            call['final_len'] = final_len
            return [0] * final_len

        qkd._privacy_amplification = privacy_spy
        key = qkd.keygen(24)

        assert key == '0' * 24
        assert call == {'candidate_len': 48, 'final_len': 24}

    def test_privacy_amplification_is_seeded_for_reproducibility(self):
        bits = [0, 1, 1, 0] * 32
        out1 = QKD(seed=99)._privacy_amplification(bits, 48)
        out2 = QKD(seed=99)._privacy_amplification(bits, 48)
        assert all(a == b for a, b in zip(out1, out2))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
