#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pyqpanda_alg.QJEPG.QJPEG (quantum JPEG down-sampling)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# make `pyqpanda_alg` importable from the nested package directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QJEPG import QJPEG


def _smooth(size: int) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, size)
    xx, yy = np.meshgrid(x, x)
    return (np.sin(xx) * np.cos(yy) * 0.4 + 0.5).astype(np.float64)


def _box_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    p = img.shape[0] // factor
    return img.reshape(p, factor, p, factor).mean(axis=(1, 3))


class TestQJPEG:
    """QJPEG 量子图像下采样测试"""

    def test_output_shape(self):
        img = _smooth(16)
        out = QJPEG(patch_size=16)(img, n_discard=1)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8, 8)

    @pytest.mark.parametrize("n_discard,expected", [(1, 8), (2, 4), (3, 2)])
    def test_downsample_factors(self, n_discard, expected):
        img = _smooth(16)
        out = QJPEG(patch_size=16)(img, n_discard=n_discard)
        assert out.shape == (expected, expected)

    def test_matches_box_average(self):
        """With the Hadamard layers, QJPEG reproduces classical box-averaging."""
        img = _smooth(16)
        out = QJPEG(patch_size=16, wrap_H_layer=True)(img, n_discard=1)
        box = _box_downsample(img, 2)
        corr = np.corrcoef(out.ravel(), box.ravel())[0, 1]
        assert corr > 0.99
        assert np.abs(out - box).mean() < 0.02

    def test_tiling_multiple_patches(self):
        """Image larger than patch_size is tiled and processed per patch."""
        img = _smooth(32)
        out = QJPEG(patch_size=16)(img, n_discard=1)
        assert out.shape == (16, 16)

    @pytest.mark.parametrize("channels", [3, 4])   # 3 = RGB, 4 = RGB+IR / RGBA
    def test_multichannel_image(self, channels):
        img = np.stack([_smooth(16) * (1.0 - 0.1 * c) for c in range(channels)], axis=-1)
        out = QJPEG(patch_size=16)(img, n_discard=1)
        assert out.shape == (8, 8, channels)
        # channels are processed independently, each preserving its own mean
        for c in range(channels):
            assert abs(out[c].mean() - img[c].mean()) < 1e-4

    def test_reject_unsupported_channel_count(self):
        img = np.stack([_smooth(16)] * 5, axis=-1)   # 5-channel is not supported
        with pytest.raises(AssertionError):
            QJPEG(patch_size=16)(img, n_discard=1)

    def test_padding_non_multiple_size(self):
        img = _smooth(20)               # not a multiple of patch_size=16
        out = QJPEG(patch_size=16)(img, n_discard=1)
        # 20 padded to 32, down-sampled by 2, trimmed back to ~20/2
        assert out.shape == (10, 10)

    def test_shots_close_to_exact(self):
        img = _smooth(8)
        exact = QJPEG(patch_size=8)(img, n_discard=1)
        sampled = QJPEG(patch_size=8, shots=100000, seed=0)(img, n_discard=1)
        assert sampled.shape == exact.shape
        assert np.abs(exact - sampled).max() < 0.05

    def test_invalid_patch_size(self):
        with pytest.raises(AssertionError):
            QJPEG(patch_size=15)        # not a power of two

    def test_invalid_n_discard(self):
        img = _smooth(16)               # patch_size 16 -> k = 4, valid n_discard in [1, 3]
        with pytest.raises(AssertionError):
            QJPEG(patch_size=16)(img, n_discard=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
