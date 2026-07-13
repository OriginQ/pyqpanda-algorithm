#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: BB84 quantum key distribution.

Alice sends random bits in random bases (Z/X); Bob measures in his own random
bases; they sift the rounds where the bases matched.  The demo runs the protocol
over a clean channel (Alice and Bob obtain an identical key with QBER 0) and then
with an intercept-resend eavesdropper (QBER jumps to ~25 % and the key is
rejected).

Run:
    python demo_BB84.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QKD import BB84, QKDInsecureError


def report(title: str, qkd: BB84, n_raw: int):
    r = qkd.distribute(n_raw)
    print(f'\n[{title}]')
    print(f'  raw qubits           : {r.n_raw}')
    print(f'  sifted (bases match) : {len(r.alice_key)} ({r.sift_rate:.1%})')
    print(f'  QBER                 : {r.qber:.3f} (secure={r.secure})')
    print(f'  Alice sifted key     : {r.alice_key[:48]}...')
    print(f'  Bob   sifted key     : {r.bob_key[:48]}...')
    print(f'  keys identical       : {r.alice_key == r.bob_key}')
    print(f'  timecost             : {r.timecost:.3f}s')


if __name__ == '__main__':
    print('=' * 64)
    print('BB84 -- prepare-and-measure QKD (4 states, 2 bases)')
    print('=' * 64)

    report('clean channel', BB84(seed=42), 8192)
    report('Eve intercepts every round', BB84(eve_prob=1.0, seed=42), 8192)

    print('\n[keygen over a clean channel]')
    key = BB84(seed=7).keygen(128)
    print(f'  128-bit privacy-amplified key: {key}')

    print('\n[keygen with full interception -> aborts]')
    try:
        BB84(eve_prob=1.0, seed=7).keygen(128)
    except QKDInsecureError as e:
        print(f'  {e}')
