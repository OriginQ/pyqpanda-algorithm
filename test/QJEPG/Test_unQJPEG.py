#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pyqpanda_alg.QJEPG.unQJPEG (inverse quantum JPEG up-sampling)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# make `pyqpanda_alg` importable from the nested package directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pyqpanda-algorithm'))
from pyqpanda_alg.QJEPG import QJPEG, unQJPEG


def _smooth(size: int) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, size)
    xx, yy = np.meshgrid(x, x)
    return (np.sin(xx) * np.cos(yy) * 0.4 + 0.5).astype(np.float64)


def _box_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    p = img.shape[0] // factor
    return img.reshape(p, factor, p, factor).mean(axis=(1, 3))


class TestUnQJPEG:
    """unQJPEG 量子图像上采样测试"""

    def test_output_shape(self):
        img = _smooth(8)
        out = unQJPEG()(img, n_append=1, patch_size=8)
        assert isinstance(out, np.ndarray)
        assert out.shape == (16, 16)

    @pytest.mark.parametrize("n_append,expected", [(1, 16), (2, 32)])
    def test_upsample_factors(self, n_append, expected):
        img = _smooth(8)
        out = unQJPEG()(img, n_append=n_append, patch_size=8)
        assert out.shape == (expected, expected)

    def test_flat_stays_flat(self):
        """A constant image must up-sample to the same constant, artifact-free."""
        flat = np.full((8, 8), 0.5, dtype=np.float64)
        out = unQJPEG()(flat, n_append=1, patch_size=8)
        assert out.shape == (16, 16)
        assert np.allclose(out, 0.5, atol=1e-6)

    def test_non_negative_and_mean_preserving(self):
        img = _smooth(8)
        out = unQJPEG()(img, n_append=1, patch_size=8)
        assert out.min() >= -1e-9
        # up-sampling preserves the overall mean intensity
        assert abs(out.mean() - img.mean()) < 1e-6

    def test_blockpool_recovers_original(self):
        """Down-pooling the up-sampled image returns (close to) the input."""
        img = _smooth(8)
        out = unQJPEG()(img, n_append=1, patch_size=8)
        rec = _box_downsample(out, 2)
        corr = np.corrcoef(rec.ravel(), img.ravel())[0, 1]
        assert corr > 0.9

    def test_round_trip_with_qjpeg(self):
        """QJPEG down then unQJPEG up returns an image of the original size."""
        orig = _smooth(16)
        low = QJPEG(patch_size=16)(orig, n_discard=1)
        assert low.shape == (8, 8)
        rec = unQJPEG()(low, n_append=1, patch_size=8)
        assert rec.shape == (16, 16)
        corr = np.corrcoef(rec.ravel(), orig.ravel())[0, 1]
        assert corr > 0.9

    @pytest.mark.parametrize("channels", [3, 4])   # 3 = RGB, 4 = RGB+IR / RGBA
    def test_multichannel_image(self, channels):
        img = np.stack([_smooth(8) * (1.0 - 0.1 * c) for c in range(channels)], axis=-1)
        out = unQJPEG()(img, n_append=1, patch_size=8)
        assert out.shape == (16, 16, channels)

    def test_invalid_patch_size(self):
        img = _smooth(8)
        with pytest.raises(AssertionError):
            unQJPEG()(img, n_append=1, patch_size=6)   # not a power of two


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
