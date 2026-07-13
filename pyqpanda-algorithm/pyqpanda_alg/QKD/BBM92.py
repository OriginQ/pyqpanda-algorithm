"""
BBM92 (Bennett, Brassard & Mermin, 1992) -- the entanglement-based twin of BB84.

Instead of Alice preparing states, a shared source emits Bell pairs
|Phi+> = (|00>+|11>)/sqrt2.  Alice and Bob each measure their qubit in a random
basis (Z or X).  Since |Phi+> is perfectly correlated in *both* bases
( = (|++>+|-->)/sqrt2 ), whenever their bases agree their outcomes are identical,
yielding the raw key.  An eavesdropper who intercepts and resends one arm breaks
the entanglement and shows up as errors, exactly as in BB84.

~ref: https://en.wikipedia.org/wiki/BBM92_protocol
~ref: https://www.sci-hub.in/10.1103/physrevlett.68.557
"""

from .QKD import QKD, Z_BASIS, X_BASIS, PI

# Measuring "at angle theta" == apply RY(-theta) then measure Z; the X basis is
# theta = pi/2 (RY(-pi/2) sends |+>->|0>, |->->|1>, matching a Hadamard).
# Z -> 0, X -> pi/2
ROT_ANGLES = {
    Z_BASIS: 0,
    X_BASIS: PI / 2,
}


class BBM92(QKD):

    '''最基础的纠缠QKD协议，BB84的纠缠版'''

    sift_efficiency = 0.5

    def _exchange(self, n_raw: int) -> dict:
        eve     = self._eve_mask(n_raw)
        a_basis = self._randbits(n_raw)
        b_basis = self._randbits(n_raw)
        e_basis = self._randbits(n_raw) if any(eve) else None

        alice = n_raw * [-1]
        bob   = n_raw * [-1]
        keep  = n_raw * [False]
        for i in range(n_raw):
            # 1) Charlie 分发一个 |Φ+> 纠缠对给Alice和Bob
            phi = self._bell_prepare()
            # 2）Alice和Bob各自选基、旋转并测量
            a_rot, b_rot = ROT_ANGLES[a_basis[i]], ROT_ANGLES[b_basis[i]]
            if eve[i]:
                # Eve在Bob拦截，选基旋转、测量并重放结果给Bob
                # Bob以为自己操作的是纠缠对中的一侧，实际是Eve制备的一个新的单比特系统
                e_rot = ROT_ANGLES[e_basis[i]]
                self._bell_rot(phi, a_rot, e_rot)
                a_out, e_out = self._bell_measure(phi)
                b_out = self._resend(e_rot, e_out, b_rot)
            else:
                self._bell_rot(phi, a_rot, b_rot)
                a_out, b_out = self._bell_measure(phi)
            alice[i], bob[i] = a_out, b_out
            # 3) Alice和Bob互相告知自己所选的基，若同基则保留测量值
            if a_basis[i] == b_basis[i]:
                keep[i] = True

        return {
            'alice': alice,
            'bob': bob,
            'keep': keep,
            'extra': {
                'eve_intercepts': eve,
            }
        }
