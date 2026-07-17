"""
B92 (Bennett, 1992) -- a minimalist prepare-and-measure protocol that needs only
two non-orthogonal states.

Alice encodes bit 0 as |0> and bit 1 as |+>.  Bob measures each qubit in a random
basis (Z or X) and keeps only *conclusive* outcomes:

    * measured in Z and got |1>  =>  the state cannot have been |0>, so Alice sent
      |+>  =>  key bit 1;
    * measured in X and got |->  =>  the state cannot have been |+>, so Alice sent
      |0>  =>  key bit 0.

All other outcomes are inconclusive and discarded, giving a ~25 % sift rate in the
ideal case.  Because the two states are non-orthogonal, no measurement can
distinguish them deterministically -- which is exactly what keeps Eve out.

~ref: https://en.wikipedia.org/wiki/B92_protocol
~ref: http://home.ustc.edu.cn/~gongsiqiu/_book/6-Quantum%20communication/Quantum%20Cryptography%20Using%20Any%20Two%20Nonorthogonal%20States.html
"""

from .QKD import QKD, Z_BASIS, X_BASIS


class B92(QKD):

    '''BB84的简化版，双基双态；通过基不一致反推'''

    sift_efficiency = 0.25

    def _exchange(self, n_raw: int) -> dict:
        eve     = self._eve_mask(n_raw)
        a_bits  = self._randbits(n_raw)     # KEY BITS
        b_basis = self._randbits(n_raw)
        e_basis = self._randbits(n_raw) if any(eve) else None

        bob  = n_raw * [-1]
        keep = n_raw * [False]
        for i in range(n_raw):
            # 1) Alice选值v，依规则使用对应的基B，制备并发送量子态 |phi> = B|0>
            #    v=0: 用Z基制备 |phi> = I|0> = |0>
            #    v=1: 用X基制备 |phi> = H|0> = |+>
            prep_basis = X_BASIS if a_bits[i] == 1 else Z_BASIS
            prep_bit = 0    # 始终是0
            phi = self._basis_prepare(prep_basis, prep_bit)
            # 2) Eve拦截，选基测量 |phi> 并重放测量结果
            if eve[i]:
                intercept_basis = e_basis[i]
                eve_bit = self._basis_measure(phi, intercept_basis)
                phi = self._basis_prepare(intercept_basis, eve_bit)
            # 3) Bob选基并测量 |phi>
            out = self._basis_measure(phi, b_basis[i])
            # 4) 若测量值为1则保留(只可能是制备基-测量基不同导致)，当前比特是否被保留
            if out == 1:    # 与prep_bit相反
                if b_basis[i] == Z_BASIS:
                    bob[i], keep[i] = 1, True   # Bob用Z基测出了1，说明Alice用的X基制备的v=1
                elif b_basis[i] == X_BASIS:
                    bob[i], keep[i] = 0, True   # Bob用X基测出了1，说明Alice用的Z基制备的v=0

        return {
            'alice': a_bits,
            'bob': bob,
            'keep': keep,
            'extra': {
                'eve_intercepts': eve,
            }
        }
