#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Quantum JPEG (QJPEG) image downsampling.

The given image is down-sampled at several compression levels and compared
with classical box-averaging.

Run:
    python demo_QJPEG.py (default -P equals -S, no tiling)
    python demo_QJPEG.py -P 256 -S 512 (large canvas with small patch, force tiling)
"""

from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import load_img, psnr, box_downsample, QJPEG, IMG_DIR, OUT_DIR


def save_filename(args):
    I: Path = args.input
    S = args.img_size
    P = args.patch_size
    sfx = ''
    sfx += f'_{P}' if S == P else f'_P{P}_S{S}'
    if args.no_H_layer is False:
       sfx += '_noH'
    return f'{I.stem}-QJPEG{sfx}.png'


def run(args):
    img = load_img(args.input, args.img_size)
    print(f'Input img_shape: {img.shape}')

    qjpeg = QJPEG(patch_size=args.patch_size, wrap_H_layer=args.no_H_layer)
    print(f'QJPEG patch_size: {qjpeg.patch_size}')

    results: dict[int, np.ndarray] = {}
    for n_discard in (1, 2, 3):
        ts_start = perf_counter()
        out = qjpeg(img, n_discard=n_discard)
        ref = box_downsample(img, 2 ** n_discard)
        ts_end = perf_counter()
        results[n_discard] = out
        print(f'n_discard={n_discard}: {img.shape} -> {out.shape} | PSNR={psnr(out, ref):.2f} dB ({ts_end - ts_start:.3f}s)')

    N = 1 + len(results)
    fig, axes = plt.subplots(1, N, figsize=(3 * N, 3))
    axes[0].imshow(img, vmin=0, vmax=1)
    axes[0].set_title(f'original {img.shape[0]}x{img.shape[1]}')
    axes[0].axis('off')
    for ax, (nd, out) in zip(axes[1:], results.items()):
        ax.imshow(np.clip(out, 0, 1), vmin=0, vmax=1)
        ax.set_title(f'QJPEG {out.shape[0]}x{out.shape[1]}\nn_discard={nd}')
        ax.axis('off')
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fp = OUT_DIR / save_filename(args)
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    print(f'Saved comparison figure to {fp}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-I', '--input', default=IMG_DIR / 'Border Collie.webp', type=Path)
    parser.add_argument('-S', '--img_size', default=256, type=int)
    parser.add_argument('-P', '--patch_size', default=None, type=int)
    parser.add_argument('-nH', '--no_H_layer', action='store_false', default='disable H layer, only experimental use')
    args = parser.parse_args()

    args.patch_size = args.patch_size or args.img_size

    if args.img_size > 256:
        print(f'>> [WARN] the first image size will exceed {args.img_size * 2**2}, might be VERY SLOW!! :(')

    run(args)
