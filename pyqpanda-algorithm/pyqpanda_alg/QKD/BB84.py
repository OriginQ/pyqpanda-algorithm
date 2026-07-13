"""
BB84 (Bennett & Brassard, 1984) -- the original prepare-and-measure protocol.

For every qubit Alice draws a random bit and a random basis (Z: |0>,|1> or
X: |+>,|->), prepares the corresponding state and sends it.  Bob measures in his
own random basis.  During sifting they keep only the rounds where their bases
match; on those the two bits are identical in an ideal channel.  An
intercept-resend eavesdropper, forced to guess the basis, corrupts 25 % of the
sifted bits and is exposed by the QBER.

~ref: https://en.wikipedia.org/wiki/BB84
~ref: https://arxiv.org/abs/2003.06557
"""

from .QKD import QKD


class BB84(QKD):

    '''最基础的非纠缠QKD协议，双基四态'''

    sift_efficiency = 0.5

    def _exchange(self, n_raw: int) -> dict:
        eve     = self._eve_mask(n_raw)
        a_bits  = self._randbits(n_raw)     # KEY BITS
        a_basis = self._randbits(n_raw)
        b_basis = self._randbits(n_raw)
        e_basis = self._randbits(n_raw) if any(eve) else None

        bob  = n_raw * [-1]
        keep = n_raw * [False]
        for i in range(n_raw):
            # 1) Alice选值v和基B，制备并发送量子态 |phi> = B|v>
            prep_basis, prep_bit = a_basis[i], a_bits[i]
            phi = self._basis_prepare(prep_basis, prep_bit)
            # 2) Eve拦截，选基测量 |phi> 并重放测量结果
            if eve[i]:
                intercept_basis = e_basis[i]
                eve_bit = self._basis_measure(phi, intercept_basis)
                phi = self._basis_prepare(intercept_basis, eve_bit)
            # 3) Bob选基并测量 |phi>
            out = self._basis_measure(phi, b_basis[i])
            # 4) Alice和Bob互相告知自己所选的基，若同基则保留测量值
            if a_basis[i] == b_basis[i]:
                bob[i], keep[i] = out, True

        return {
            'alice': a_bits,
            'bob': bob,
            'keep': keep,
            'extra': {
                'eve_intercepts': eve,
            }
        }
