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

from pyqpanda3.core import CPUQVM, QProg, Encode, H, CNOT

from ..plugin import QFT


class unQJPEG:
    """
    Inverse Quantum JPEG (unQJPEG) image up-sampling / super-resolution.

    unQJPEG is the inverse of :class:`~pyqpanda_alg.QJEPG.QJPEG`, originally
    proposed within this library. Where QJPEG *discards* the high spatial-frequency
    qubits to down-sample an image, unQJPEG *re-introduces* them as vacuum (zeros),
    i.e. it zero-pads the spatial-frequency spectrum and transforms back, which is
    the quantum analogue of sinc / Fourier interpolation.

    Each ``patch_size`` x ``patch_size`` patch is encoded into ``2 * log2(patch_size)``
    qubits (QPIE amplitude encoding). For each of the two image axes the quantum
    Fourier transform maps the register to the frequency domain; ``n_append`` fresh
    ``|0>`` qubits are inserted between the low- and high-frequency components (the
    high-frequency band is padded with zeros); an inverse QFT of the enlarged
    register returns to the spatial domain. Reading the probabilities yields an
    up-sampled patch of side ``patch_size * 2 ** n_append``.

    The sign qubit (most-significant frequency bit) of each axis is duplicated onto
    the appended qubits with ``CNOT`` gates so that the negative-frequency band is
    mapped to the top of the enlarged register, preserving the Hermitian frequency
    layout required for a real, interpolating reconstruction.

    Parameters
        separated_QFT : ``bool``, optional\n
            Whether the QFT operation should be axis-wise (Default: True).
        shots : ``int``, optional\n
            If given, the output probabilities are sampled with this many
            measurement shots (emulating finite-sampling noise). If ``None``
            (Default), the exact probabilities are used.
        seed : ``int``, optional\n
            Random seed used when ``shots`` is set (Default: None).

    References
        [1] Simone Roncallo, Lorenzo Maccone, Chiara Macchiavello. "Quantum JPEG".
        AVS Quantum Science 5, 043803 (2023). arXiv:2306.09323.

    >>> import numpy as np
    >>> from pyqpanda_alg.QJEPG import unQJPEG
    >>> img = np.random.rand(8, 8).astype(np.float64)
    >>> unqjpeg = unQJPEG()
    >>> out = unqjpeg(img, n_append=1, patch_size=8)     # 8x8 -> 16x16
    >>> print(out.shape)
    (16, 16)
    """

    def __init__(self, separated_QFT: bool = True, shots: int = None, seed: int = None):
        self.separated_QFT = separated_QFT
        self.shots = shots
        self.seed = seed
        # per-call state, set inside __call__
        self.patch_size: int = None
        self.n_qubit: int = None                # aka. k_out/max_qubit
        self._grid: tuple[int, int] = None      # (n_patch_H, n_patch_W) tiling of the current channel
        self._scale = 1                         # up-sampling factor per axis (2 ** n_append)

    def __call__(self, img: ndarray, n_append: int = 1, patch_size: int = None) -> ndarray:
        assert isinstance(img, ndarray) and img.dtype in [np.float32, np.float64], 'image must be a float array'
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
        assert isinstance(n_append, int) and n_append >= 1, '"n_append" must be a positive int'
        if patch_size is None:
            patch_size = min(img.shape[:2])
        assert patch_size & (patch_size - 1) == 0, '"patch_size" must be power of 2'
        assert patch_size <= min(img.shape[:2]), '"patch_size" must be smaller than image size'
        self.patch_size = patch_size
        nq_init = int(math.log2(patch_size)) * 2
        assert 2 <= nq_init <= 18, f'invalid input "nq_init": {nq_init}, should be in range [2, 18]'
        n_qubit = nq_init + n_append * 2   # qubits of the enlarged (output) register
        assert 4 <= n_qubit <= 20, f'invalid output "n_qubit": {n_qubit}, should be in range [4, 20]; reduce patch_size or n_append'
        self.n_qubit = n_qubit
        self._scale = 2 ** n_append

        H, W, C = img.shape
        # pad to whole patches, [H, W, C]
        img_pad, pads = self.pad(img)
        # split channels, C * [H, W]
        channels = [img_pad[..., c] for c in range(C)]
        channels_processed: list[ndarray] = []
        for channel in channels:
            # patchify, [B, p, p]
            patches = self.pachify(channel)
            # vectorize, [B, d=p**2]
            vectors, norm = self.vectorize(patches)
            # QFT-append-IQFT (spectral zero-pad), [B, D=P**2]
            qvectors = self.apply_unQJPEG(vectors, n_append)
            # devectorize, [B, P, P]
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
        """Zero-pad an ``[H, W, C]`` image so H and W are whole multiples of
        ``patch_size``, pasting the content to the center. Returns the padded image
        and the ``(top, bottom, left, right)`` pad widths, or ``None`` if unneeded."""
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
        """Trim the padding added by :meth:`pad`. The pad widths are scaled *up* by
        the up-sampling factor to match the enlarged resolution."""
        if not pads or all(e == 0 for e in pads):
            return img
        top, bottom, left, right = pads
        s = self._scale
        H, W, C = img.shape
        t = round(top * s)
        b = round(bottom * s)
        l = round(left * s)
        r = round(right * s)
        return img[t:(H - b) if b else H, l:(W - r) if r else W, :]

    def pachify(self, img: ndarray) -> ndarray:
        """Split a single-channel ``[H, W]`` image into non-overlapping
        ``patch_size`` x ``patch_size`` patches, returned as ``[B, p, p]``."""
        P = self.patch_size
        H, W = img.shape
        assert H % P == 0 and W % P == 0
        nH, nW = H // P, W // P
        self._grid = (nH, nW)
        return img.reshape(nH, P, nW, P).swapaxes(1, 2).reshape(nH * nW, P, P)

    def unpachify(self, patches: ndarray) -> ndarray:
        """Re-assemble ``[B, P, P]`` (up-sampled) patches into a ``[Ho, Wo]`` image."""
        nH, nW = self._grid
        B, P, _ = patches.shape
        assert B == nH * nW
        return patches.reshape(nH, nW, P, P).swapaxes(1, 2).reshape(nH * P, nW * P)

    def vectorize(self, patches: ndarray) -> tuple:
        """Vectorize ``[B, p, p]`` patches to normalized amplitude vectors
        ``[B, p**2]`` (``amplitude = sqrt(pixel / patch_sum)``) and return the
        per-patch intensity ``norm``."""
        B = patches.shape[0]
        vect = patches.reshape(B, -1)
        norm = vect.sum(axis=1)
        amps = np.zeros_like(vect, dtype=patches.dtype)
        nz = norm > 0
        amps[nz] = np.sqrt(vect[nz] / norm[nz, None])
        return amps, norm

    def devectorize(self, vectors: ndarray, norm: object) -> ndarray:
        """Decode output probability vectors ``[B, P**2]`` into intensity patches
        ``[B, P, P]``, rescaled by ``norm`` and the pixel-count ratio so the mean
        intensity of each patch is preserved by the up-sampling."""
        B, D = vectors.shape
        P = int(round(math.sqrt(D)))
        n_in = self.patch_size ** 2
        n_out = D
        patches = vectors * norm[:, None] * (n_out / n_in)
        return patches.reshape(B, P, P)

    def apply_unQJPEG(self, vectors: ndarray, n_append: int) -> ndarray:
        """Run the unQJPEG (spectral zero-pad) circuit on each amplitude vector
        ``[B, p**2]`` and return the up-sampled probability vectors ``[B, P**2]``,
        with ``P = p * 2 ** n_append``."""

        '''
        LSB                         MSB
        |-------------nq--------------|
        |------ko------|------ko------|
        |---ki---| add |---ki---| add |
        '''
        nq = self.n_qubit    # all qubits
        ko = nq // 2         # qubits per axis of the (enlarged) output (aka. k_in)
        ki = ko - n_append   # qubits per axis of the (init) input (aka. k_out)
        qv = list(range(nq))
        # per-axis qubit layout of the enlarged register
        col_init = list(range(0, ki))
        col_add = list(range(ki, ko))
        col_msb = col_init[-1]
        row_init = list(range(ko, ko + ki))
        row_add = list(range(ko + ki, 2 * ko))
        row_msb = row_init[-1]
        # data index bit order: col bits (LSB..MSB) then row bits (LSB..MSB)
        enc_qubits = col_init + row_init
        measured = qv

        B = vectors.shape[0]
        D = 2 ** nq     # dim out
        out = np.zeros((B, D), dtype=vectors.dtype)
        rng = np.random.default_rng(self.seed) if self.shots else None
        for b in range(B):
            vec = vectors[b]
            if not np.any(vec): continue

            prog = QProg(nq)
            # data enc
            enc = Encode()
            enc.amplitude_encode(enc_qubits, list(vec))
            prog << enc.get_circuit()
            # QFT
            if self.separated_QFT:
                # per-axis QFT of the init registers
                prog << QFT(col_init)
                prog << QFT(row_init)
            else:
                # QFT over the whole init register
                prog << QFT(enc_qubits)
            # zero-pad the spectrum: copy the sign qubit onto the appended qubits
            for qi in col_add:
                prog << CNOT(col_msb, qi)
            for qi in row_add:
                prog << CNOT(row_msb, qi)
            # iQFT
            if self.separated_QFT:
                # per-axis inverse QFT of the enlarged registers
                prog << QFT(list(range(0, ko))).dagger()
                prog << QFT(list(range(ko, 2 * ko))).dagger()
            else:
                # inverse QFT over the whole enlarged register
                prog << QFT(qv).dagger()

            machine = CPUQVM()
            machine.run(prog, 1)
            prob_dict = machine.result().get_prob_dict(measured)
            probs = self._prob_dict_to_vector(prob_dict, len(measured), D, vectors.dtype)
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
