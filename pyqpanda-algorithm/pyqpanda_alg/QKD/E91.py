"""
E91 (Ekert, 1991) -- entanglement-based QKD secured by a Bell inequality.

A source shares Bell pairs |Phi+> = (|00>+|11>)/sqrt2.  Alice measures her qubit
along one of three angles {0, pi/4, pi/2}; Bob along {pi/4, pi/2, 3pi/4}.

    * Key:      rounds where they happen to pick the *same physical angle*
                (Alice pi/4 = Bob pi/4, and Alice pi/2 = Bob pi/2) are perfectly
                correlated and become the shared key.
    * Security: the mismatched-angle rounds feed the CHSH quantity
                S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3).
                Genuine entanglement gives |S| = 2*sqrt2 ~ 2.83; any
                intercept-resend eavesdropper drives S back toward the classical
                bound of 2 (and simultaneously raises the QBER).

~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution#E91_protocol:_Artur_Ekert_.281991.29
~ref: https://cqi.inf.usi.ch/qic/91_Ekert.pdf
"""

from .QKD import QKD, PI

# Alice's three settings
A_ANGLES = {
    0: 0.0,
    1: PI / 4,
    2: PI / 2,
}
# Bob's three settings
B_ANGLES = {
    0: PI / 4,
    1: PI / 2,
    2: 3 * PI / 4,
}
assert len(A_ANGLES) == len(B_ANGLES)
# Eve guesses from the union
E_ANGLES = sorted(set(A_ANGLES.values()) | set(B_ANGLES.values()))
# valid rounds whose physical angles coincide -> raw key: (a_idx, b_idx)
# can configure by yourself according to A_ANGLES and B_ANGLES
_KEY_SETTINGS = {
    (1, 0),     # PI / 4
    (2, 1),     # PI / 2
}
# CHSH uses Alice {0, pi/2} = idx {0, 2} and Bob {pi/4, 3pi/4} = idx {0, 2}
# https://en.wikipedia.org/wiki/CHSH_inequality
_CHSH_SIGN = {
    (0, 0): +1,
    (0, 2): -1,
    (2, 0): +1,
    (2, 2): +1,
}


class E91(QKD):

    '''BBM92的三态版本；使用CHSH检测窃听'''

    sift_efficiency = len(_KEY_SETTINGS) / len(A_ANGLES)**2

    def _exchange(self, n_raw: int) -> dict:
        eve   = self._eve_mask(n_raw)
        a_idx = self._randtrits(n_raw)
        b_idx = self._randtrits(n_raw)
        e_ang = self._rng.choices(E_ANGLES, k=n_raw) if any(eve) else None

        alice = n_raw * [-1]
        bob   = n_raw * [-1]
        keep  = n_raw * [False]

        # accumulate correlations for CHSH: sum of s_A*s_B and counts per setting
        corr_sum = {s: 0.0 for s in _CHSH_SIGN}
        corr_cnt = {s: 0   for s in _CHSH_SIGN}
        for i in range(n_raw):
            # 1) Charlie 分发一个 |Φ+> 纠缠对给Alice和Bob
            phi = self._bell_prepare()
            # 2）Alice和Bob各自选角度、旋转并测量
            ai, bi = a_idx[i], b_idx[i]
            a_rot, b_rot = A_ANGLES[ai], B_ANGLES[bi]
            if eve[i]:
                # Eve在Bob拦截，选基旋转、测量并重放结果给Bob
                # Bob以为自己操作的是纠缠对中的一侧，实际是Eve制备的一个新的单比特系统
                e_rot = e_ang[i]
                self._bell_rot(phi, a_rot, e_rot)
                a_out, e_out = self._bell_measure(phi)
                b_out = self._resend(e_rot, e_out, b_rot)
            else:
                self._bell_rot(phi, a_rot, b_rot)
                a_out, b_out = self._bell_measure(phi)
            alice[i], bob[i] = a_out, b_out
            # 3) Alice和Bob互相告知自己所选的旋转角度，若一致则保留测量值
            if (ai, bi) in _KEY_SETTINGS:
                keep[i] = True
            # 4) 计算CHSH quantity，验证纠缠保持度
            if (ai, bi) in corr_sum:
                s = (-1) ** (a_out == b_out)
                corr_sum[(ai, bi)] += s     # homo=+1, heter=-1
                corr_cnt[(ai, bi)] += 1

        S = 0.0
        for s in _CHSH_SIGN:
            E = corr_sum[s] / corr_cnt[s] if corr_cnt[s] else 0.0
            S += _CHSH_SIGN[s] * E

        return {
            'alice': alice,
            'bob': bob,
            'keep': keep,
            'extra': {
                'eve_intercepts': eve,
                'chsh_S': S,
            }
        }
