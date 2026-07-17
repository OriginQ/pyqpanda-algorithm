#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pyqpanda_alg.QKD.E91 (entanglement-based QKD with a CHSH test)."""

import sys
import math
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QKD import E91, QKDInsecureError


class TestE91:
    """E91 纠缠式量子密钥分发测试"""

    def test_keygen_length_and_alphabet(self):
        key = E91(seed=1).keygen(48)
        assert isinstance(key, str)
        assert len(key) == 48
        assert set(key) <= {'0', '1'}

    def test_clean_channel_is_error_free(self):
        r = E91(seed=2).distribute(6000)
        assert r.qber == 0.0
        assert r.secure is True
        assert r.alice_key == r.bob_key

    def test_sift_efficiency_two_ninths(self):
        r = E91(seed=3).distribute(6000)
        assert abs(r.sift_rate - 2.0 / 9.0) < 0.05

    def test_chsh_violation_without_eve(self):
        r = E91(seed=3).distribute(6000)
        # genuine entanglement violates the classical bound of 2 (ideal 2*sqrt2)
        assert r.extra['chsh_S'] > 2.4
        assert r.extra['chsh_S'] <= 2 * math.sqrt(2) + 0.15   # within statistics of Tsirelson

    def test_eve_probability_breaks_entanglement(self):
        r = E91(eve_prob=1.0, seed=4).distribute(6000)
        assert r.qber > 0.12
        assert r.secure is False
        assert r.extra['chsh_S'] < 2.0           # CHSH falls back below the classical bound

    def test_keygen_aborts_under_full_interception(self):
        with pytest.raises(QKDInsecureError):
            E91(eve_prob=1.0, seed=5).keygen(48)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
