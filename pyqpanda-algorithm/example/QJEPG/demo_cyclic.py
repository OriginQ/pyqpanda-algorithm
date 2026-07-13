#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: QJPEG & unQJPEG forward-inverse cyclic.

Down-samples the given image with QJPEG, then up-samples it back with the unQJPEG
quantum circuit (spectral zero-padding / sinc interpolation) and reports how well
the original resolution is recovered. Also demonstrates plain up-sampling of a
low-resolution image.

Run:
    python demo_cyclic.py
    python demo_cyclic.py -S 512
"""

from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import load_img, psnr, bilinear_upsample, QJPEG, unQJPEG, IMG_DIR, OUT_DIR


def save_filename(args):
    I: Path = args.input
    S = args.size
    sfx = ''
    sfx += f'_{S}'
    return f'{I.stem}-cyclic{sfx}.png'


def run(args):
    size = args.size
    half = size // 2
    orig = load_img(args.input, size)
    print(f'Input img_shape: {orig.shape}')

    # 1) downsample to half resolution with QJPEG
    ts_start = perf_counter()
    lowres = QJPEG(patch_size=size)(orig, n_discard=1)
    ts_end = perf_counter()
    print(f'Downsampled with QJPEG: {orig.shape} -> {lowres.shape} ({ts_end - ts_start:.3f}s)')

    # 2) upsample back to full resolution with unQJPEG
    ts_start = perf_counter()
    recon = unQJPEG()(lowres, n_append=1)
    ts_end = perf_counter()
    psnr_recon = psnr(recon, orig)
    print(f'Upsampled with unQJPEG: {lowres.shape} -> {recon.shape} | PSNR={psnr_recon:.2f} dB ({ts_end - ts_start:.3f}s)')

    # 3) bilinear upscaling of the same lowres image
    ts_start = perf_counter()
    bilinear = bilinear_upsample(lowres, 2)
    ts_end = perf_counter()
    psnr_bilinear = psnr(bilinear, orig)
    print(f'Upsampled with BILINEAR: {lowres.shape} -> {bilinear.shape}  | PSNR={psnr_bilinear:.2f} dB ({ts_end - ts_start:.3f}s)')

    imgs = [orig, np.clip(lowres, 0, 1), np.clip(recon, 0.0, 1.0), bilinear]
    titles = [
        f'original {size}x{size}',
        f'QJPEG {half}x{half}',
        f'unQJPEG {size}x{size}\nPSNR={psnr_recon:.2f}',
        f'bilinear {size}x{size}\nPSNR={psnr_bilinear:.2f}',
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fp = OUT_DIR / save_filename(args)
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    print(f'Saved comparison figure to {fp}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-I', '--input', default=IMG_DIR / 'Border Collie.webp', type=Path)
    parser.add_argument('-S', '--size', default=256, type=int)
    args = parser.parse_args()

    run(args)
