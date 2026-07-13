"""
SARG04 (Scarani, Acin, Ribordy & Gisin, 2004) -- same four states as BB84 but a
different sifting rule that hardens it against photon-number-splitting attacks.

Alice sends one of the four BB84 states; the *key bit is the basis she used*
(0 = Z, 1 = X), not the state value.  During sifting she does not reveal the
basis; instead she announces an unordered pair made of the state she sent and one
decoy from the conjugate basis.  Bob, who measured in a random basis, can identify
the sent state only when his outcome is orthogonal to -- and therefore excludes --
the decoy:

    conclusive  <=>  Bob measured in the conjugate basis AND his outcome is the
                     opposite value of the decoy.

His own outcome can never be orthogonal to the true state, so the exclusion is
unambiguous and he recovers Alice's basis (the key bit).  The ideal sift rate is
~25 %.

~ref: https://en.wikipedia.org/wiki/SARG04
~ref: https://arxiv.org/abs/quant-ph/0211131
"""

from .QKD import QKD


class SARG04(QKD):

    '''BB84+B92缝合，双基四态；通过基不一致反推'''

    sift_efficiency = 0.25

    def _exchange(self, n_raw: int) -> dict:
        eve     = self._eve_mask(n_raw)
        a_basis = self._randbits(n_raw)  # Alice's basis = KEY BIT
        a_bits  = self._randbits(n_raw)  # Alice's prepare bits
        d_bits  = self._randbits(n_raw)  # Alice's decoy bits (assumed from the conjugate basis)
        b_basis = self._randbits(n_raw)
        e_basis = self._randbits(n_raw) if any(eve) else None

        bob   = n_raw * [-1]
        keep  = n_raw * [False]
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
            # 4) Alice告知Bob刚才制备的 |phi> 来自于集合 {|0>/|1>, |+>/|->}
            #    其中一个是正确答案，另一个是相反基的诱骗答案 (故Alice没有直接公布基)
            #    Bob通过基不一致反推检查自己的测量结果，并告知Alice当前比特是否被保留
            # 若Alice宣告 {|0>, |+>}, 则 Bob 必须测出 1(Z基) 或 -(X基) 才能确定值为v=0
            # 若Alice宣告 {|0>, |->}, 则 Bob 必须测出 1(Z基) 或 +(X基) 才能确定值为v=1/0
            # 若Alice宣告 {|1>, |+>}, 则 Bob 必须测出 0(Z基) 或 -(X基) 才能确定值为v=0/1
            # 若Alice宣告 {|1>, |->}, 则 Bob 必须测出 0(Z基) 或 +(X基) 才能确定值为v=1
            mismatch_sent = b_basis[i] == a_basis[i] and out != a_bits[i]
            mismatch_decoy = b_basis[i] == 1 - a_basis[i] and out != d_bits[i]
            if mismatch_decoy and not mismatch_sent:
                bob[i], keep[i] = a_basis[i], True
            elif mismatch_sent and not mismatch_decoy:
                bob[i], keep[i] = 1 - a_basis[i], True

        return {
            'alice': a_basis,
            'bob': bob,
            'keep': keep,
            'extra': {
                'eve_intercepts': eve,
            }
        }
