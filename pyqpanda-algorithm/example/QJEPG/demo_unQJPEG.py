#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: inverse Quantum JPEG (unQJPEG) image upsampling / super-resolution.

The given image is up-sampled at several upscale levels and compared
with classical bilinear interpolation.

Run:
    python demo_unQJPEG.py (default -P equals -S, no tiling)
    python demo_unQJPEG.py -P 32 -S 64 (large canvas with small patch, force tiling)
"""

from pathlib import Path
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import load_img, psnr, bilinear_upsample, unQJPEG, IMG_DIR, OUT_DIR



def save_filename(args):
    I: Path = args.input
    S = args.img_size
    P = args.patch_size
    sfx = ''
    sfx += f'_{P}' if S == P else f'_P{P}_S{S}'
    return f'{I.stem}-unQJPEG{sfx}.png'


def run(args):
    img = load_img(args.input, args.img_size)
    print(f'Input img_shape: {img.shape}')

    unqjpeg = unQJPEG(separated_QFT=args.no_separated_QFT)
    print(f'patch_size: {args.patch_size}')

    results: dict[int, np.ndarray] = {}
    for n_append in (1, 2, 3):
        ts_start = perf_counter()
        out = unqjpeg(img, n_append=n_append, patch_size=args.patch_size)
        ref = bilinear_upsample(img, 2 ** n_append)
        ts_end = perf_counter()
        results[n_append] = out
        print(f'n_append={n_append}: {img.shape} -> {out.shape} | PSNR={psnr(out, ref):.2f} dB ({ts_end - ts_start:.3f}s)')

    N = 1 + len(results)
    fig, axes = plt.subplots(1, N, figsize=(3 * N, 3))
    axes[0].imshow(img, vmin=0, vmax=1)
    axes[0].set_title(f'original {img.shape[0]}x{img.shape[1]}')
    axes[0].axis('off')
    for ax, (nd, out) in zip(axes[1:], results.items()):
        ax.imshow(np.clip(out, 0, 1), vmin=0, vmax=1)
        ax.set_title(f'unQJPEG {out.shape[0]}x{out.shape[1]}\nn_append={nd}')
        ax.axis('off')
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fp = OUT_DIR / save_filename(args)
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    print(f'Saved comparison figure to {fp}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-I', '--input', default=IMG_DIR / 'Border Collie.webp', type=Path)
    parser.add_argument('-S', '--img_size', default=64, type=int)
    parser.add_argument('-P', '--patch_size', default=None, type=int)
    parser.add_argument('-nQFT', '--no_separated_QFT', action='store_false', help='disable separated QFT, only experimental use')
    args = parser.parse_args()

    if args.img_size > 64:
        print(f'>> [WARN] the last image size will exceed {args.img_size * 2**3}, could be VERY SLOW!! :(')

    run(args)
