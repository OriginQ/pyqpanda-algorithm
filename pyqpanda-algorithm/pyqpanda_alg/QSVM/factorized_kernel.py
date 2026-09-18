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

"""Counts-only component sampling for the fixed, ideal QSVM feature map."""

from collections.abc import Mapping
from itertools import combinations, islice, product
from numbers import Integral

import numpy as np
from pyqpanda3.core import CNOT, QProg, U1, U2, measure


__all__ = ["FactorizedQuantumKernel"]


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _as_data(values, name):
    if values is None:
        raise ValueError(f"{name} must not be None")
    try:
        data = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if data.ndim not in (1, 2):
        raise ValueError(f"{name} must be a 1D or 2D array")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{name} must have at least two features")
    if data.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.array(data, dtype=np.float64, copy=True)
        if not np.isfinite(data).all():
            raise ValueError(f"{name} must contain finite float64 values")
        # Finite features can still overflow the fixed map's gate parameters.
        phase = 2.0 * (np.pi - data[:, 0]) * (np.pi - data[:, 1])
        if not np.isfinite(2.0 * data).all() or not np.isfinite(phase).all():
            raise ValueError(f"{name} produces non-finite feature-map angles")
    return data


def _overlap_program(x, y):
    """Restrict the legacy overlap gate order to one connected component."""
    width = len(x)
    program = QProg(width)
    qubits = program.qubits()
    phase_x = phase_y = 0.0
    if width == 2:
        phase_x = 2.0 * (np.pi - x[0]) * (np.pi - x[1])
        phase_y = 2.0 * (np.pi - y[0]) * (np.pi - y[1])
    for layer in range(2):
        for q, value in zip(qubits, x):
            program << U2(q, 0, np.pi) << U1(q, 2.0 * value)
        if width == 2:
            program << CNOT(qubits[0], qubits[1]) << U1(qubits[1], phase_x)
            if layer == 1:
                # Cancel only the adjacent middle CNOT pair, as in the source.
                program << U1(qubits[1], -phase_y)
            program << CNOT(qubits[0], qubits[1])
    for layer in range(2):
        for q, value in zip(qubits, y):
            program << U1(q, -2.0 * value) << U2(q, 0, np.pi)
        if width == 2 and layer == 0:
            program << CNOT(qubits[0], qubits[1])
            program << U1(qubits[1], -phase_y) << CNOT(qubits[0], qubits[1])
    program << measure(qubits, qubits)
    return program


def _zero_frequency(counts, width, shots):
    if not isinstance(counts, Mapping):
        raise ValueError("run_counts must return a counts mapping")
    total = 0
    for bits, count in counts.items():
        if not isinstance(bits, str) or len(bits) != width or any(b not in "01" for b in bits):
            raise ValueError(f"counts keys must be {width}-bit binary strings")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
            raise ValueError("counts values must be non-negative integers")
        total += int(count)
    if total != shots:
        raise ValueError(f"counts total must equal requested shots ({shots}), got {total}")
    return int(counts.get("0" * width, 0)) / shots


class FactorizedQuantumKernel:
    """Sample the existing QSVM map using circuits of at most two qubits.

    This is an opt-in *ideal, fixed-map* estimator, not a general noisy echo
    kernel. The map has two encoding layers, entangles only features 0 and 1,
    and leaves every later feature independent. Each non-identity component
    is prepared and measured by the caller; its all-zero frequency is used
    in a product. No simulator, state, probability oracle or PSD projection
    is used here. ``QuantumKernel_vqnet`` is unchanged.

    Parameters
    ----------
    run_counts : callable
        Required ``run_counts(program, shots) -> Mapping[str, int]``. Execute
        the supplied measured QProg from all-zero initialization and return
        exactly ``shots`` observations. Keys are binary strings of the local
        width (1 or 2), values non-negative integers. Absent zero keys mean
        zero successes. The runner owns execution/reset and must provide
        independent samples across component positions and stationary
        behavior throughout an evaluation. Runner exceptions propagate.
    shots : int, default=1024
        Shots per distinct non-identity component, not per matrix element.
    batch_size : int, default=100
        Maximum number of matrix index pairs buffered at once. Calls to the
        runner remain sequential, each with ``shots``; this does not bound
        the evaluation-local cache or batch device submissions.
    n_qbits : int or None, default=None
        Optional expected feature count (>=2). None infers it per evaluate
        call. This is the original feature width, not the execution width.

    Notes
    -----
    Exactly equal component inputs (after float64 conversion) contribute
    1 without execution. This identity and symmetry require the ideal map;
    they do not calibrate preparation, gate or measurement noise. Likewise,
    local measurements need not reproduce a full noisy device circuit.

    An evaluation reuses a measured frequency only for the same component
    position and exactly equal ordered input pair. Different positions stay
    independent even when their numeric values match. Reuse and symmetric
    mirroring correlate matrix entries; neither creates extra samples. No
    statistics survive the call, including a failed call. Distinct runner
    invocations must not replay one common random sample stream.

    Independent unbiased component frequencies give an unbiased product
    for the ideal kernel, but finite-shot Gram matrices need not be PSD.
    See ``docs/QSVM_factorized_kernel.md`` for assumptions and resource costs.
    """

    def __init__(self, run_counts, *, shots=1024, batch_size=100, n_qbits=None):
        if not callable(run_counts):
            raise TypeError("run_counts must be callable")
        self._run_counts = run_counts
        self._shots = _positive_integer(shots, "shots")
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._n_qbits = None if n_qbits is None else _positive_integer(n_qbits, "n_qbits", 2)

    def evaluate(self, x_vec, y_vec=None):
        """Return an (n_x, n_y) float64 sampled kernel; usable as SVC callable.

        Accept real finite 1D (one sample) or 2D numeric array-likes, with
        matching d>=2. Explicit empty (0, d) arrays return an empty matrix;
        [] has no valid feature dimension and is rejected. x_vec=None,
        ragged/non-numeric arrays, non-finite values/angles and inconsistent
        feature widths raise ValueError before any runner call. y_vec=None
        means x_vec. Equal datasets use one triangle mirrored exactly.
        """
        x = _as_data(x_vec, "x_vec")
        y = x if y_vec is None else _as_data(y_vec, "y_vec")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x_vec and y_vec have incompatible feature dimensions")
        if self._n_qbits is not None and x.shape[1] != self._n_qbits:
            raise ValueError("feature dimension must equal n_qbits")
        symmetric = np.array_equal(x, y)
        kernel = np.empty((len(x), len(y)), dtype=np.float64)
        if symmetric:
            np.fill_diagonal(kernel, 1.0)
            pairs = combinations(range(len(x)), 2)
        else:
            pairs = product(range(len(x)), range(len(y)))
        components = [(0, 2)] + [(j, j + 1) for j in range(2, x.shape[1])]
        cache = {}
        batch_limit = min(self._batch_size, kernel.size)
        while True:
            batch = list(islice(pairs, batch_limit))
            if not batch:
                break
            for i, j in batch:
                value = 1.0
                for start, stop in components:
                    cx = tuple(x[i, start:stop])
                    cy = tuple(y[j, start:stop])
                    if cx == cy:
                        continue  # Ideal fixed-map identity, not a noisy measurement.
                    key = (start, stop, cx, cy)
                    if key not in cache:
                        program = _overlap_program(cx, cy)
                        counts = self._run_counts(program, self._shots)
                        cache[key] = _zero_frequency(counts, stop - start, self._shots)
                    value *= cache[key]
                kernel[i, j] = value
                if symmetric:
                    kernel[j, i] = value
        return kernel
