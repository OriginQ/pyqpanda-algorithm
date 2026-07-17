#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: E91 (Ekert) entanglement-based quantum key distribution.

A source shares Bell pairs.  Alice and Bob each measure along random settings;
rounds with the same physical angle give perfectly correlated bits (the key),
while the mismatched-angle rounds are spent on the CHSH test.  Genuine
entanglement yields S ~ 2*sqrt(2) = 2.83; an eavesdropper collapses the
entanglement, pushing S back toward the classical bound of 2 and raising the QBER.

Run:
    python demo_E91.py
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QKD import E91, QKDInsecureError


def report(title: str, qkd: E91, n_raw: int):
    r = qkd.distribute(n_raw)
    print(f'\n[{title}]')
    print(f'  entangled pairs     : {r.n_raw}')
    print(f'  key rounds (sifted) : {len(r.alice_key)} ({r.sift_rate:.1%})')
    print(f'  QBER                : {r.qber:.3f} (secure={r.secure})')
    print(f"  CHSH S              : {r.extra['chsh_S']:+.3f}  (|S|>2 => entangled; ideal {2*math.sqrt(2):.3f})")
    print(f'  keys identical      : {r.alice_key == r.bob_key}')
    print(f'  timecost            : {r.timecost:.3f}s')


if __name__ == '__main__':
    print('=' * 64)
    print('E91 -- entanglement-based QKD with a CHSH Bell test')
    print('=' * 64)

    report('clean channel', E91(seed=42), 8192)
    report('Eve intercepts every round', E91(eve_prob=1.0, seed=42), 8192)

    print('\n[keygen over a clean channel]')
    key = E91(seed=7).keygen(128)
    print(f'  128-bit privacy-amplified key: {key}')

    print('\n[keygen with full interception -> aborts]')
    try:
        E91(eve_prob=1.0, seed=7).keygen(128)
    except QKDInsecureError as e:
        print(f'  {e}')
