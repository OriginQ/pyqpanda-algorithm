#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: SARG04 quantum key distribution.

SARG04 uses the same four states as BB84 but encodes the key in the *basis* and
sifts differently: Alice announces the state she sent together with a decoy from
the conjugate basis, and Bob keeps a round only when his measurement excludes one
of the two. This makes it more robust against photon-number-splitting attacks.
An intercept-resend eavesdropper still drives up the QBER.

Run:
    python demo_SARG04.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QKD import SARG04, QKDInsecureError


def report(title: str, qkd: SARG04, n_raw: int):
    r = qkd.distribute(n_raw)
    print(f'\n[{title}]')
    print(f'  raw qubits          : {r.n_raw}')
    print(f'  conclusive (sifted) : {len(r.alice_key)} ({r.sift_rate:.1%})')
    print(f'  QBER                : {r.qber:.3f} (secure={r.secure})')
    print(f'  Alice sifted key    : {r.alice_key[:48]}...')
    print(f'  Bob   sifted key    : {r.bob_key[:48]}...')
    print(f'  keys identical      : {r.alice_key == r.bob_key}')
    print(f'  timecost            : {r.timecost:.3f}s')


if __name__ == '__main__':
    print('=' * 64)
    print('SARG04 -- BB84 states, basis-encoded key, exclusion sifting')
    print('=' * 64)

    report('clean channel', SARG04(seed=42), 8192)
    report('Eve intercepts every round', SARG04(eve_prob=1.0, seed=42), 8192)

    print('\n[keygen over a clean channel]')
    key = SARG04(seed=7).keygen(128)
    print(f'  128-bit privacy-amplified key: {key}')

    print('\n[keygen with full interception -> aborts]')
    try:
        SARG04(eve_prob=1.0, seed=7).keygen(128)
    except QKDInsecureError as e:
        print(f'  {e}')
