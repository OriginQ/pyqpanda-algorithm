#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pyqpanda_alg.QKD.BB84 (prepare-and-measure quantum key distribution)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QKD import BB84, QKDInsecureError, mean


class TestBB84:
    """BB84 量子密钥分发测试"""

    def test_keygen_length_and_alphabet(self):
        key = BB84(seed=1).keygen(64)
        assert isinstance(key, str)
        assert len(key) == 64
        assert set(key) <= {'0', '1'}

    def test_clean_channel_is_error_free(self):
        r = BB84(seed=2).distribute(2000)
        assert r.qber == 0.0
        assert r.secure is True
        assert r.alice_key == r.bob_key          # Alice and Bob share the key

    def test_sift_efficiency_about_half(self):
        r = BB84(seed=3).distribute(4000)
        assert abs(r.sift_rate - 0.5) < 0.05     # bases agree ~half the time

    def test_eve_probability_raises_qber(self):
        r = BB84(eve_prob=1.0, seed=4).distribute(4000)
        assert r.qber > 0.12                     # intercept-resend -> ~25 %
        assert r.secure is False
        assert all(r.extra['eve_intercepts'])

    def test_keygen_aborts_under_full_interception(self):
        with pytest.raises(QKDInsecureError):
            BB84(eve_prob=1.0, seed=5).keygen(64)

    def test_partial_eve_probability_is_applied_per_round(self):
        r = BB84(eve_prob=0.5, seed=10).distribute(6000)
        assert abs(mean(r.extra['eve_intercepts']) - 0.5) < 0.03
        assert 0.07 < r.qber < 0.18              # expected QBER ~= eve_prob / 4

    def test_bb84_bits_are_reproducible(self):
        # clean-channel sifted key is fixed by the classical choices -> reproducible
        k1 = BB84(seed=9).keygen(48)
        k2 = BB84(seed=9).keygen(48)
        assert k1 == k2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
