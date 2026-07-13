import sys
from pathlib import Path

import numpy as np
from numpy import ndarray
from PIL import Image
from PIL.Image import Resampling
import matplotlib ; matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QJEPG import QJPEG, unQJPEG   # keep

BASE_PATH = Path(__file__).resolve().parent
IMG_DIR = BASE_PATH / 'img'
OUT_DIR = BASE_PATH / 'output'


def load_img(path: Path, size: int) -> ndarray:
    img = Image.open(path).resize((size, size), resample=Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0    # [H, W, C] / [H, W]


def psnr(a: ndarray, b: ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


def nearest_neighbor_upsample(img: ndarray, factor: int = 2) -> ndarray:
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def bilinear_upsample(img: ndarray, factor: int = 2) -> ndarray:
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        H, W = img.shape
    im = np.asarray(np.clip(img, 0, 1) * 255, dtype=np.uint8)
    img = Image.fromarray(im).resize((W * factor, H * factor), resample=Resampling.BILINEAR)
    hires = np.asarray(img, dtype=np.float32) / 255.0
    return hires


def box_downsample(img: ndarray, factor: int = 2) -> ndarray:
    if len(img.shape) == 3:
        H, W, C = img.shape
        p = H // factor
        lowres = img.reshape(p, factor, p, factor, C).mean(axis=(1, 3))
    else:
        H, W = img.shape
        p = H // factor
        lowres = img.reshape(p, factor, p, factor).mean(axis=(1, 3))
    return lowres
