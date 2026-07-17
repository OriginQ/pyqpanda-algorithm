"""
Common infrastructure shared by every quantum-key-distribution protocol.

A QKD protocol lets two parties (Alice and Bob) grow a shared secret key whose
secrecy is guaranteed by quantum mechanics rather than by computational
hardness: the no-cloning theorem and the measurement-disturbance principle mean
that any eavesdropper (Eve) unavoidably imprints detectable errors on the raw
key.  Every protocol here follows the same three stages

    1. quantum transmission   -- Alice encodes random bits into qubits which Bob
                                 measures (implemented with real ``pyqpanda3``
                                 circuits, one circuit per transmitted qubit);
    2. sifting                -- over a public classical channel the two parties
                                 keep only the rounds that are physically
                                 correlated (e.g. matching bases);
    3. parameter estimation   -- a random subset of the sifted bits is disclosed
                                 to estimate the quantum bit-error rate (QBER); a
                                 QBER above the protocol threshold reveals Eve and
                                 the key is discarded;
    4. privacy amplification -- the undisclosed candidate key is compressed with
                                 a random binary Toeplitz hash, removing partial
                                 information that Eve may have learned.

Sub-classes only implement :meth:`_exchange`, which performs stages 1-2 for a
block of raw qubits and returns Alice's/Bob's raw bits plus the sift mask.  The
base class turns that into a :class:`QKDResult` and exposes :meth:`keygen`.

~ref: https://en.wikipedia.org/wiki/Quantum_key_distribution
"""

import math
import random
from time import perf_counter
from dataclasses import dataclass, field

from pyqpanda3.core import CPUQVM, QProg, H, X, RY, CNOT, measure

# measurement / preparation bases
Z_BASIS = 0   # computational basis     |0>, |1>
X_BASIS = 1   # Hadamard basis          |+>, |->

PI = math.pi

mean = lambda x: sum(x) / len(x) if x else 0.0


def _bits_to_str(bits: list[int]) -> str:
    return ''.join('1' if b else '0' for b in bits)


class QKDInsecureError(RuntimeError):
    """Raised by :meth:`QKD.keygen` when the estimated QBER exceeds the security
    threshold, i.e. an eavesdropper is suspected and the key must be thrown away."""


@dataclass
class QKDResult:
    """Outcome of a single key-distribution round (a block of ``n_raw`` qubits)."""

    protocol: str
    n_raw: int
    alice_key: str          # Alice's sifted bits (before public disclosure), as '0'/'1'
    bob_key: str            # Bob's sifted bits
    key: str                # candidate key after disclosure, before privacy amplification
    sift_rate: float        # kept fraction = len(sifted) / n_raw
    qber: float             # estimated quantum bit-error rate on the disclosed sample
    secure: bool            # qber <= threshold
    timecost: float         # time cost
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f'QKDResult({self.protocol}, n_raw={self.n_raw}, '
                f'sift_rate={self.sift_rate:.2f}, qber={self.qber:.3f}, '
                f'secure={self.secure}, key_len={len(self.key)})')


class QKD:
    """Base class: quantum transmission helpers, sifting, QBER estimation, keygen.

    Args:
        eve_prob: probability that Eve intercepts and resends each transmitted
            qubit. ``0`` disables Eve and ``1`` attacks every round.
        sample_fraction: fraction of the sifted key publicly disclosed to estimate
            the QBER (those bits are removed from the final key).
        min_sample_size: minimum disclosed sample when ``sample_fraction`` is
            non-zero and enough sifted bits exist. A useful lower bound keeps
            short-key QBER estimates from being dominated by sampling noise.
        qber_threshold: QBER above which the key is deemed insecure (BB84's
            asymptotic bound is ~11 %).
        privacy_ratio: final-key length divided by the candidate-key length used
            for Toeplitz privacy amplification. The default 0.5 hashes two
            candidate bits into one final bit.
        seed: seed for the (classical) random-number generator, for reproducibility.
    """

    #: rough kept-fraction after sifting, used only to size the raw block in keygen
    sift_efficiency = 0.5

    def __init__(self,
        eve_prob: float = 0.0,
        sample_fraction: float = 0.2,
        qber_threshold: float = 0.11,
        privacy_ratio: float = 0.5,
        min_sample_size: int = 128,
        seed: int = None,
    ):
        assert 0.0 <= eve_prob <= 1.0, 'eve_prob must be in [0, 1]'
        assert 0.0 <= sample_fraction < 1.0, 'sample_fraction must be in [0, 1)'
        assert 0.0 <= qber_threshold <= 1.0, 'qber_threshold must be in [0, 1]'
        assert 0.0 < privacy_ratio <= 1.0, 'privacy_ratio must be in (0, 1]'
        assert min_sample_size >= 1, 'min_sample_size must be >= 1'
        self.eve_prob = float(eve_prob)
        self.sample_fraction = float(sample_fraction)
        self.qber_threshold = float(qber_threshold)
        self.privacy_ratio = float(privacy_ratio)
        self.min_sample_size = int(min_sample_size)
        self._rng = random.Random(seed)
        self._qvm = CPUQVM()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------ #
    # low-level quantum helpers (one circuit == one transmitted qubit)    #
    # ------------------------------------------------------------------ #
    def _run(self, prog: QProg, n_cbits: int) -> list[int]:
        """Run ``prog`` for a single shot and return the measured bits as a list
        indexed by classical-bit number (``cbit j`` at list position ``j``)."""
        self._qvm.run(prog, 1)
        bitstr = next(iter(self._qvm.result().get_counts()))   # big-endian over cbits
        return [int(bitstr[n_cbits - 1 - j]) for j in range(n_cbits)]

    def _basis_prepare(self, prep_basis: int, prep_bit: int) -> QProg:
        """Prepare primitive: prepare one qubit as ``prep_bit`` in ``prep_basis`` (Z/X)"""
        prog = QProg(1)
        if prep_bit:
            prog << X(0)            # 0/1 -> |0>/|1>
        if prep_basis == X_BASIS:
            prog << H(0)            # |0>/|1> -> |+>/|->
        return prog

    def _basis_measure(self, prog: QProg, meas_basis: int) -> int:
        """Measure primitive: measure a qstate in ``meas_basis`` (Z/X), return the outcome."""
        if meas_basis == X_BASIS:
            prog << H(0)            # rotate X basis back to computational for read-out
        prog << measure(0, 0)
        return self._run(prog, 1)[0]

    def _bell_prepare(self) -> QProg:
        """Create a Bell pair |Phi+> = (|00>+|11>)/sqrt2."""
        prog = QProg(2)
        prog << H(0) << CNOT(0, 1)
        return prog

    def _bell_rot(self, prog: QProg, rot0: float, rot1: float) -> QProg:
        """Rot the qubits at ``rot`` (RY angles) in a bell-state pair"""
        prog << RY(0, -rot0) << RY(1, -rot1)
        return prog

    def _bell_measure(self, prog: QProg) -> list[int, int]:
        """Measure a bell-state pair, returns (out0, out1).
        For |Phi+> the two spin outcomes correlate as ``<s0 s1> = cos(rot0-rot1)``,
        so equal rotations give identical bits -- the raw key of the entanglement
        protocols."""
        prog << measure(0, 0) << measure(1, 1)
        return self._run(prog, 2)

    def _resend(self, prep_rot: float, prep_bit: int, meas_rot: float) -> int:
        """Prepare the eigenstate that Eve collapsed onto (``prep_bit`` at angle
        ``prep_rot``) and measure it at ``meas_rot``. Used to model intercept-resend
        on an entangled arm."""
        prog = QProg(1)
        if prep_bit:
            prog << X(0)
        prog << RY(0, prep_rot)          # rebuild the measured eigenstate
        prog << RY(0, -meas_rot)         # receiver's measurement rotation
        prog << measure(0, 0)
        return self._run(prog, 1)[0]

    def _randbits(self, n_len: int) -> list[int]:
        return [self._rng.randrange(0, 2) for _ in range(n_len)]

    def _randtrits(self, n_len: int) -> list[int]:
        return [self._rng.randrange(0, 3) for _ in range(n_len)]

    def _eve_mask(self, n_raw: int) -> list[bool]:
        """Draw the rounds attacked by an intercept-resend Eve.

        The exact endpoints avoid consuming random numbers when Eve is entirely
        absent or present, while intermediate probabilities make an independent
        Bernoulli decision for every transmitted qubit.
        """
        if self.eve_prob <= 0.0: return [False] * n_raw
        if self.eve_prob >= 1.0: return [True]  * n_raw
        return [self._rng.random() < self.eve_prob for _ in range(n_raw)]

    # ------------------------------------------------------------------ #
    # protocol hook + public API                                          #
    # ------------------------------------------------------------------ #
    def _exchange(self, n_raw: int) -> dict:
        """Run quantum transmission + sifting for ``n_raw`` raw qubits.

        Sub-classes must return a dict with keys ``alice`` (int array of Alice's
        raw bits), ``bob`` (Bob's raw bits), ``keep`` (bool sift mask), and an
        optional ``extra`` dict of protocol-specific diagnostics."""
        raise NotImplementedError

    def distribute(self, n_raw: int) -> QKDResult:
        """Distribute a block of ``n_raw`` raw qubits and return a :class:`QKDResult`."""
        assert n_raw >= 1, 'n_raw must be >= 1'

        ts_start = perf_counter()
        data = self._exchange(n_raw)
        ts_end = perf_counter()

        alice: list[int]  = data['alice']
        bob:   list[int]  = data['bob']
        keep:  list[bool] = data['keep']
        a_sift = [alice[i] for i, k in enumerate(keep) if k]
        b_sift = [bob  [i] for i, k in enumerate(keep) if k]
        n_sift = len(a_sift)

        # publicly disclose a random sample to estimate the QBER
        n_sample = round(self.sample_fraction * n_sift)
        if self.sample_fraction > 0.0 and n_sift > 1:
            n_sample = max(n_sample, min(self.min_sample_size, n_sift - 1))
        if self.eve_prob > 0.0 and n_sift > 0:
            n_sample = max(1, n_sample)
        if n_sift > 1:
            n_sample = min(n_sample, n_sift - 1)   # never disclose the whole key

        if n_sample > 0:
            idx = self._rng.sample(range(n_sift), k=n_sample)
            a_sift_sample = [a_sift[i] for i in idx]
            b_sift_sample = [b_sift[i] for i in idx]
            qber = mean([a != b for a, b in zip(a_sift_sample, b_sift_sample)])
            key_bits = [a_sift[i] for i in range(n_sift) if i not in idx]
        else:
            qber = mean([a != b for a, b in zip(a_sift, b_sift)]) if n_sift else 0.0
            key_bits = a_sift

        return QKDResult(
            protocol=self.name,
            n_raw=n_raw,
            alice_key=_bits_to_str(a_sift),
            bob_key=_bits_to_str(b_sift),
            key=_bits_to_str(key_bits),
            sift_rate=n_sift / n_raw,
            qber=qber,
            secure=qber <= self.qber_threshold,
            timecost=ts_end - ts_start,
            extra=data.get('extra', {}),
        )

    def privacy_amplification(self, key_bits: str | list[int], final_len: int) -> list[int]:
        """Hash ``n`` candidate bits to ``final_len`` bits with a Toeplitz matrix.

        A binary ``m x n`` Toeplitz matrix is fully described by ``m + n - 1``
        random bits (one per diagonal). Multiplication is performed over GF(2).
        This is a universal-hash privacy-amplification step; ``keygen`` controls
        the compression ratio through :attr:`privacy_ratio`.
        """
        if isinstance(key_bits, str):
            assert set(key_bits) <= {'0', '1'}, 'key_bits must be binary'
            bits = [1 if ch == '1' else 0 for ch in key_bits]
        else:
            assert isinstance(key_bits, list)
            bits = key_bits
        assert isinstance(final_len, int) and final_len >= 1, 'final_len must be >= 1'
        assert final_len <= len(bits), f'privacy amplification needs at least {final_len} candidate bits'

        n = len(bits)
        diagonals = self._randbits(n + final_len - 1)
        out = [0] * final_len
        for i in range(final_len):
            start = final_len - 1 - i
            row = diagonals[start:start + n]
            res = [a & b for a, b in zip(bits, row)]
            z = res[0]
            for zz in res[1:]:
                z ^= zz
            out[i] = z
        return out

    def keygen(self, n_len: int) -> str:
        """Generate a shared secret key of ``n_len`` bits (returned as a '0'/'1' string).

        Raw qubits are transmitted in blocks until enough candidate bits accumulate.
        If an eavesdropper is detected (QBER over threshold) a
        :class:`QKDInsecureError` is raised and the key is discarded. The candidate
        key is then compressed to exactly ``n_len`` bits by Toeplitz hashing."""
        assert isinstance(n_len, int) and n_len >= 1, 'n_len must be >= 1'

        # Privacy amplification consumes a longer candidate key and hashes it to
        # the requested final length.
        candidate_len = math.ceil(n_len / self.privacy_ratio)
        candidate = ''
        while len(candidate) < candidate_len:
            # Size every block (including top-ups) to cover both proportional
            # disclosure and the minimum parameter-estimation sample.
            needed = candidate_len - len(candidate)
            proportional_need = needed / max(1.0 - self.sample_fraction, 1e-3)
            minimum_need = needed + (self.min_sample_size if self.sample_fraction > 0.0 else 0)
            sift_needed = max(proportional_need, minimum_need)
            block = math.ceil(sift_needed / max(self.sift_efficiency, 1e-3)) + 32
            res = self.distribute(block)
            if not res.secure:
                raise QKDInsecureError(
                    f'{self.name}: estimated QBER {res.qber:.3f} exceeds threshold '
                    f'{self.qber_threshold:.3f} -- eavesdropping suspected, key discarded')
            candidate += res.key

        final = self.privacy_amplification(candidate[:candidate_len], n_len)
        return _bits_to_str(final)
