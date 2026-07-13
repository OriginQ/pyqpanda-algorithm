#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pyqpanda_alg.QKD.SARG04 (basis-encoded prepare-and-measure QKD)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QKD import SARG04, QKDInsecureError


class TestSARG04:
    """SARG04 量子密钥分发测试"""

    def test_keygen_length_and_alphabet(self):
        key = SARG04(seed=1).keygen(64)
        assert isinstance(key, str)
        assert len(key) == 64
        assert set(key) <= {'0', '1'}

    def test_clean_channel_is_error_free(self):
        r = SARG04(seed=2).distribute(3000)
        assert r.qber == 0.0
        assert r.secure is True
        assert r.alice_key == r.bob_key

    def test_sift_efficiency_about_quarter(self):
        r = SARG04(seed=3).distribute(4000)
        assert abs(r.sift_rate - 0.25) < 0.05    # exclusion sifting keeps ~1/4

    def test_eve_probability_raises_qber(self):
        r = SARG04(eve_prob=1.0, seed=4).distribute(4000)
        assert r.qber > 0.12
        assert r.secure is False

    def test_keygen_aborts_under_full_interception(self):
        with pytest.raises(QKDInsecureError):
            SARG04(eve_prob=1.0, seed=5).keygen(64)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
