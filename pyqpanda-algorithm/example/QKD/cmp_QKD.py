#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the five QKD protocols side by side.

Runs BB84, B92, E91, BBM92 and SARG04 over a clean channel with a low/high-intensity
intercept-resend eavesdropper, then reports sift efficiency and the quantum
bit-error rate (QBER) for each.  In every case the QBER stays at 0 without an
eavesdropper and rises well above the ~11 % security threshold once Eve is on the
line -- the signature that lets the parties detect her.

Outputs (written to ./output):
    cmp_QKD.png  -- sift-efficiency and QBER bar charts

Run:
    python cmp_QKD.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pyqpanda_alg.QKD import QKD, BB84, B92, E91, BBM92, SARG04

import matplotlib ; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).resolve().parent
OUT_DIR = BASE_PATH / 'output'

PROTOCOLS: list[QKD] = [BB84, B92, E91, BBM92, SARG04]
N_RAW = 30000
THRESHOLD = 0.11
EVE_LOW_PROB = 0.2
EVE_HIGH_PROB = 0.5


if __name__ == '__main__':
    names, sift, qber_low, qber_high = [], [], [], []
    print(f'{"protocol":8s} {"sift":>7s} {"QBER(Low)":>12s} {"QBER(High)":>10s}  secure(Low/High)')
    print('-' * 60)
    for P in PROTOCOLS:
        eve_low = P(eve_prob=EVE_LOW_PROB, seed=1).distribute(N_RAW)
        eve_high = P(eve_prob=EVE_HIGH_PROB, seed=1).distribute(N_RAW)
        names.append(P.__name__)
        sift.append(eve_low.sift_rate)
        qber_low.append(eve_low.qber)
        qber_high.append(eve_high.qber)
        chsh = ''
        if 'chsh_S' in eve_low.extra:
            chsh = f"   CHSH S: Low={eve_low.extra['chsh_S']:+.2f} High={eve_high.extra['chsh_S']:+.2f}"
        print(f'{P.__name__:8s} {eve_low.sift_rate:6.1%} {eve_low.qber:12.3f} {eve_high.qber:10.3f}   {eve_low.secure}/{eve_high.secure}{chsh}')

    x = range(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x, sift, color='#4c72b0')
    ax1.set_xticks(list(x)) ; ax1.set_xticklabels(names, rotation=20)
    ax1.set_ylabel('sift efficiency')
    ax1.set_title('Sift efficiency (kept fraction)')
    ax1.set_ylim(0, 0.6)
    for i, v in zip(x, sift):
        ax1.text(i, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=9)
    w = 0.38
    ax2.bar([i - w / 2 for i in x], qber_low, width=w, label=f'Low intercepts {EVE_LOW_PROB:.0%}', color='#55a868')
    ax2.bar([i + w / 2 for i in x], qber_high, width=w, label=f'High intercepts {EVE_HIGH_PROB:.0%}', color='#c44e52')
    ax2.axhline(THRESHOLD, ls='--', color='k', lw=1, label=f'security threshold {THRESHOLD:.0%}')
    ax2.set_xticks(list(x)) ; ax2.set_xticklabels(names, rotation=20)
    ax2.set_ylabel('QBER')
    ax2.set_title('QBER: eavesdropped low vs high intensity')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / 'cmp_QKD.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'\nSaved comparison figure to {path}')
