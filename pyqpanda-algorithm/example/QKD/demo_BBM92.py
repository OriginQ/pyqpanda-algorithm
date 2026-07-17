#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: BBM92 entanglement-based quantum key distribution.

BBM92 is the EPR version of BB84: instead of Alice preparing states, a source
emits Bell pairs |Phi+> and both parties measure in a random Z/X basis.  Because
|Phi+> is perfectly correlated in both bases, matching-basis rounds yield the key.
Intercepting one arm breaks the correlation and shows up as QBER, just like BB84.

Run:
    python demo_BBM92.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QKD import BBM92, QKDInsecureError


def report(title: str, qkd: BBM92, n_raw: int):
    r = qkd.distribute(n_raw)
    print(f'\n[{title}]')
    print(f'  entangled pairs      : {r.n_raw}')
    print(f'  sifted (bases match) : {len(r.alice_key)} ({r.sift_rate:.1%})')
    print(f'  QBER                 : {r.qber:.3f} (secure={r.secure})')
    print(f'  Alice sifted key     : {r.alice_key[:48]}...')
    print(f'  Bob   sifted key     : {r.bob_key[:48]}...')
    print(f'  keys identical       : {r.alice_key == r.bob_key}')
    print(f'  timecost             : {r.timecost:.3f}s')


if __name__ == '__main__':
    print('=' * 64)
    print('BBM92 -- entanglement-based twin of BB84 (EPR pairs)')
    print('=' * 64)

    report('clean channel', BBM92(seed=42), 8192)
    report('Eve intercepts every round', BBM92(eve_prob=1.0, seed=42), 8192)

    print('\n[keygen over a clean channel]')
    key = BBM92(seed=7).keygen(128)
    print(f'  128-bit privacy-amplified key: {key}')

    print('\n[keygen with full interception -> aborts]')
    try:
        BBM92(eve_prob=1.0, seed=7).keygen(128)
    except QKDInsecureError as e:
        print(f'  {e}')
