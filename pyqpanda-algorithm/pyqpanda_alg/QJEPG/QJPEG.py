# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
from numpy import ndarray

from pyqpanda3.core import CPUQVM, QProg, Encode, H

from ..plugin import QFT


class QJPEG:
    """
    Quantum JPEG (QJPEG) image down-sampling / compression.

    The QJPEG algorithm encodes an image patch into the amplitudes of a quantum
    state (QPIE, Quantum Probability Image Encoding), applies the quantum Fourier
    transform to move to the spatial-frequency domain, discards the high
    spatial-frequency qubits, and transforms back. Reading out the measurement
    probabilities of the remaining qubits yields a lower-resolution image, in the
    same spirit as the classical JPEG that filters high-frequency components [1].

    The image is first split into square ``patch_size`` x ``patch_size`` patches.
    Each patch is processed independently by a ``2 * log2(patch_size)`` qubit
    circuit and down-sampled by a factor of ``2 ** n_discard`` along each axis, so
    a ``P x P`` patch becomes a ``(P / 2**n_discard) x (P / 2**n_discard)`` patch.

    Parameters
        patch_size : ``int``\n
            Side length of the square patches, must be a power of two. The circuit
            uses ``2 * log2(patch_size)`` qubits, which is required to be in
            ``[4, 20]`` (i.e. ``patch_size`` in ``[4, 1024]``).
        wrap_H_layer : ``bool``, optional\n
            Whether to wrap the transform with Hadamard layers, as proposed in [1].
            The Hadamards reduce the statistical fluctuations of each pixel while
            preserving the contrast of the image, making the reconstruction faithful
            to the classical box down-sampling. Strongly recommended (Default: True).
        shots : ``int``, optional\n
            If given, the output probabilities are sampled with this many
            measurement shots (emulating finite-sampling noise on real hardware).
            If ``None`` (Default), the exact probabilities are used.
        seed : ``int``, optional\n
            Random seed used when ``shots`` is set (Default: None).

    References
        [1] Simone Roncallo, Lorenzo Maccone, Chiara Macchiavello. "Quantum JPEG".
        AVS Quantum Science 5, 043803 (2023). arXiv:2306.09323.

    >>> import numpy as np
    >>> from pyqpanda_alg.QJEPG import QJPEG
    >>> img = np.random.rand(16, 16).astype(np.float64)
    >>> qjpeg = QJPEG(patch_size=16)
    >>> out = qjpeg(img, n_discard=1)     # 16x16 -> 8x8
    >>> print(out.shape)
    (8, 8)
    """

    def __init__(self, patch_size: int = 256, wrap_H_layer: bool = True, shots: int = None, seed: int = None):
        assert patch_size & (patch_size - 1) == 0, '"patch_size" must be power of 2'
        self.patch_size = patch_size
        n_qubit = int(math.log2(patch_size)) * 2
        assert 4 <= n_qubit <= 20, f'invalid input "n_qubit": {n_qubit}, should be in range [4, 20]'
        self.n_qubit = n_qubit                  # i.e. k_in/max_qubit

        self.wrap_H_layer = wrap_H_layer
        self.shots = shots
        self.seed = seed
        # per-call state, set inside __call__
        self._grid: tuple[int, int] = None      # (n_patch_H, n_patch_W) tiling of the current channel
        self._scale = 1                         # down-sampling factor per axis (2 ** n_discard)

    def __call__(self, img: ndarray, n_discard: int = 2) -> ndarray:
        assert isinstance(img, ndarray) and img.dtype in [np.float32, np.float64], 'image must be a float ndarray'
        assert 0 <= img.min() and img.max() <= 1.0, 'pixel value must be in [0, 1]'
        orig_dtype = img.dtype
        if orig_dtype != np.float64:    # tackle precision issue
            img = img.astype(np.float64)
        if len(img.shape) == 3:
            assert img.shape[-1] in [3, 4], "colored image must be in shape (H, W, C) with C in [3, 4]"
            is_grey = False
        else:
            assert len(img.shape) == 2, "grey image must be in shape (H, W)"
            is_grey = True
            img = np.expand_dims(img, -1)
        assert self.patch_size <= min(img.shape[:2]), '"patch_size" must be smaller than image size'
        k = self.n_qubit // 2   # qubits per axis == log2(patch_size)
        assert isinstance(n_discard, int) and 1 <= n_discard <= k - 1, f'"n_discard" must be an int in [1, {k - 1}] for patch_size={self.patch_size}'
        self._scale = 2 ** n_discard

        H, W, C = img.shape
        # pad to whole patches, [H, W, C]
        img_pad, pads = self.pad(img)
        # split channels, C * [H, W]
        channels = [img_pad[..., c] for c in range(C)]
        channels_processed: list[ndarray] = []
        # foreach C
        for channel in channels:
            # patchify, [B=(H*W/P**2), h=P, w=P]
            patches = self.pachify(channel)
            # vectorize, [B, D=P**2]
            vectors, norm = self.vectorize(patches)
            # QFT-discard-IQFT, [B, d=p**2]
            qvectors = self.apply_QJPEG(vectors, n_discard)
            # devectorize, [B, p, p]
            devectors = self.devectorize(qvectors, norm)
            # unpatchify, [Ho, Wo]
            channel_unpatchify = self.unpachify(devectors)
            channels_processed.append(channel_unpatchify)
        # merge channel, [Ho, Wo, C]
        img_merged = np.stack(channels_processed, axis=-1)
        # trim pads, [Ho, Wo, C]
        img_unpad = self.unpad(img_merged, pads)
        # dtype back
        img_unpad = img_unpad.astype(orig_dtype)
        return img_unpad[..., 0] if is_grey else img_unpad

    def pad(self, img: ndarray):
        """Zero-pad an ``[H, W, C]`` image so that H and W are whole multiples of
        ``patch_size``, pasting the original content to the center. Returns the
        padded image together with the ``(top, bottom, left, right)`` pad widths
        (in original pixels), or ``None`` when no padding is needed."""
        H, W, C = img.shape
        n_patch_H = math.ceil(H / self.patch_size)
        n_patch_W = math.ceil(W / self.patch_size)
        H_ex = n_patch_H * self.patch_size
        W_ex = n_patch_W * self.patch_size
        if H_ex == H and W_ex == W:
            return img, None
        pad_H = H_ex - H
        pad_W = W_ex - W
        pads = (
            math.floor(pad_H / 2), math.ceil(pad_H / 2),
            math.floor(pad_W / 2), math.ceil(pad_W / 2),
        )
        img_ex = np.zeros(shape=(H_ex, W_ex, C), dtype=img.dtype)
        img_ex[pads[0]:pads[0] + H, pads[2]:pads[2] + W, :] = img
        return img_ex, pads

    def unpad(self, img: ndarray, pads=None) -> ndarray:
        """Trim the padding added by :meth:`pad` from an ``[H, W, C]`` image. The
        pad widths are scaled down by the down-sampling factor to match the
        processed (lower) resolution."""
        if not pads or all(e == 0 for e in pads):
            return img
        top, bottom, left, right = pads
        s = self._scale
        H, W, C = img.shape
        t = round(top / s)
        b = round(bottom / s)
        l = round(left / s)
        r = round(right / s)
        return img[t:(H - b) if b else H, l:(W - r) if r else W, :]

    def pachify(self, img: ndarray) -> ndarray:
        """Split a single-channel ``[H, W]`` image into non-overlapping
        ``patch_size`` x ``patch_size`` patches, returned as ``[B, P, P]`` in
        row-major patch order."""
        P = self.patch_size
        H, W = img.shape
        assert H % P == 0 and W % P == 0
        nH, nW = H // P, W // P
        self._grid = (nH, nW)
        return img.reshape(nH, P, nW, P).swapaxes(1, 2).reshape(nH * nW, P, P)

    def unpachify(self, patches: ndarray) -> ndarray:
        """Re-assemble ``[B, p, p]`` (processed) patches into a single-channel
        ``[Ho, Wo]`` image using the tiling recorded by :meth:`pachify`."""
        nH, nW = self._grid
        B, p, _ = patches.shape
        assert B == nH * nW
        return patches.reshape(nH, nW, p, p).swapaxes(1, 2).reshape(nH * p, nW * p)

    def vectorize(self, patches: ndarray) -> tuple[ndarray, ndarray]:
        """Vectorize ``[B, P, P]`` patches to amplitude vectors ``[B, P**2]``.

        Each patch is flattened (row-major) and encoded as a normalized quantum
        state whose amplitudes are the square-roots of the intensities, i.e.
        ``amplitude = sqrt(pixel / patch_sum)`` (so ``|amplitude|**2`` recovers the
        normalized intensity). The per-patch intensity sum is returned in ``norm``
        so the absolute scale can be restored on decoding. Empty (all-zero) patches
        map to a zero vector.
        """
        B = patches.shape[0]
        vect = patches.reshape(B, -1)
        norm = vect.sum(axis=1)
        amps = np.zeros_like(vect, dtype=patches.dtype)
        nz = norm > 0
        amps[nz] = np.sqrt(vect[nz] / norm[nz, None])
        return amps, norm

    def devectorize(self, vectors: ndarray, norm: ndarray) -> ndarray:
        """Decode output probability vectors ``[B, p**2]`` back into intensity
        patches ``[B, p, p]``. The probabilities are rescaled by the stored patch
        intensity ``norm`` and by the pixel-count ratio so that the mean intensity
        of each patch is preserved by the down-sampling."""
        B, d = vectors.shape
        p = int(round(math.sqrt(d)))
        n_in = self.patch_size ** 2
        n_out = d
        patches = vectors * norm[:, None] * (n_out / n_in)
        return patches.reshape(B, p, p)

    def apply_QJPEG(self, vectors: ndarray, n_discard: int) -> ndarray:
        """Run the QJPEG quantum circuit on each amplitude vector ``[B, P**2]`` and
        return the output measurement-probability vectors ``[B, p**2]``.

        For every patch the circuit (i) amplitude-encodes the state, (ii) optionally
        applies a Hadamard layer, (iii) applies the QFT over all ``n0`` qubits,
        (iv) applies the inverse QFT over the lowest ``n1 = n0 - n_discard`` qubits,
        (v) discards the ``n_discard`` high-frequency qubits at the top of each
        (row / column) half, and (vi) reads the probabilities of the remaining
        ``n2 = n0 - 2 * n_discard`` qubits.
        """

        '''
        LSB                           MSB
        |--------------nq---------------|
        |------ki-------|------ki-------|
        |---ko---| drop |---ko---| drop |
        '''
        nq = self.n_qubit       # all qubits
        ki = nq // 2            # half qubits (aka. k_in)
        ko = ki - n_discard     # half qubits kept (aka. k_out)
        qv = list(range(nq))
        # drop the top(MSB)/last n_discard qubits of each half/axis
        dropped = [i for p in range(2) for i in range(ki * p + ko, ki * p + ki)]
        # qubits kept for measurement
        measured = [i for i in qv if i not in dropped]

        B = vectors.shape[0]
        d = 2 ** (nq - 2 * n_discard)   # dim out
        out = np.zeros((B, d), dtype=vectors.dtype)
        rng = np.random.default_rng(self.seed) if self.shots else None
        for b in range(B):
            vec = vectors[b]
            if not np.any(vec): continue

            prog = QProg(nq)
            # data enc
            enc = Encode()
            enc.amplitude_encode(qv, list(vec))
            prog << enc.get_circuit()
            # QFT
            if self.wrap_H_layer:   # 抵消QFT中的Hlayer
                for qi in qv:
                    prog << H(qi)
            prog << QFT(qv)
            # 舍去列比特 (末尾之前ntilde个)
            # iQFT
            prog << QFT(qv[:nq - n_discard]).dagger()
            if self.wrap_H_layer:   # 抵消iQFT中的Hlayer
                for qi in measured:
                    prog << H(qi)
            # 舍去行比特 (中部之前ntilde个，仅保留measured)

            machine = CPUQVM()
            machine.run(prog, 1)
            prob_dict = machine.result().get_prob_dict(measured)
            probs = self._prob_dict_to_vector(prob_dict, len(measured), d, vectors.dtype)
            if self.shots:
                probs = rng.multinomial(self.shots, probs) / self.shots
            out[b] = probs
        return out

    @staticmethod
    def _prob_dict_to_vector(prob_dict: dict, n: int, dim: int, dtype=np.float64) -> ndarray:
        """Convert a ``get_prob_dict`` result into a dense vector indexed so that
        ``qubits[0]`` is the least-significant bit (matching the QPIE encoding)."""
        vec = np.zeros(dim, dtype=dtype)
        for key, val in prob_dict.items():
            idx = 0
            for k in range(n):
                # key is big-endian over `qubits`: leftmost char == last qubit
                if key[n - 1 - k] == '1':
                    idx |= (1 << k)
            vec[idx] += val
        return vec
