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
"""Encoding-expressivity and trainability diagnostics for QReupload models.

``fourier_spectrum``   — empirical DFT of a fitted 1-feature model's output over
                          one input period; with *unit* input scaling its harmonics
                          coincide with the truncated Fourier series whose degree is
                          bounded by the number of re-uploads (trainable scaling can
                          shift peaks off the integer grid — see the function doc).
``trainability_scan``  — barren-plateau diagnostic: gradient-variance vs qubit
                          count; an exponential decay (negative log-slope) warns
                          that adding qubits makes the model exponentially harder
                          to train.

These turn two empirical facts about data re-uploading models into reusable
tools so users can size a model to their data (enough frequency reach) without
walking into a barren plateau.
"""
import numpy as np

from pyqpanda3.vqcircuit import DiffMethod
from .qreupload import _QReuploadCore


def fourier_spectrum(model, n_points=512, x_range=(-np.pi, np.pi), max_harmonic=None):
    """Empirical Fourier spectrum of a fitted single-feature QReupload model.

    The model output is sampled on a dense grid over one input period and FFT'd.
    With *unit* input scaling the model is exactly a degree-``n_layers`` truncated
    Fourier series, so magnitudes are (approximately) zero for ``k > n_layers``
    (Schuld, Sweke & Meyer 2021). With a *trainable* input-scaling factor the
    accessible frequencies are learned multiples of the inputs and need not be
    integers, so read this as an empirical DFT over the chosen interval: peaks may
    fall between integer bins (spectral leakage) rather than proving a strict
    integer-degree bound.

    Parameters
    ----------
    model : fitted QReuploadRegressor with ``n_features_in_ == 1``.
    n_points : int, grid resolution (use a power of 2).
    x_range : (low, high), one period of the input.
    max_harmonic : int or None, highest harmonic to return (default n_points//2).

    Returns
    -------
    harmonics : ndarray of int
    magnitudes : ndarray of float   (|c_k|, same length as harmonics)
    """
    if getattr(model, "n_features_in_", None) != 1:
        raise ValueError("fourier_spectrum requires a single-feature (1-qubit) fitted model.")
    lo, hi = x_range
    if not np.isclose(hi - lo, 2 * np.pi):
        raise ValueError("x_range must span exactly 2*pi so the harmonic index equals the frequency.")
    grid = np.linspace(lo, hi, n_points, endpoint=False)
    vals = np.asarray(model.predict(grid), float)
    coef = np.fft.rfft(vals) / n_points          # harmonic index == frequency (period hi-lo)
    mag = np.abs(coef)
    mag[1:] *= 2.0                               # one-sided amplitude
    k = np.arange(mag.size)
    if max_harmonic is not None:
        keep = k <= max_harmonic
        k, mag = k[keep], mag[keep]
    return k, mag


def trainability_scan(qubit_range, n_layers=3, n_samples=200, seed=0,
                      entangle=True, return_grads=False):
    """Barren-plateau diagnostic: Var[d<Z_0>/dtheta] vs number of qubits.

    For each qubit count, random parameters are drawn uniformly in [-pi, pi] and
    the exact adjoint gradient of <Z_0> w.r.t. one block angle is measured; its
    variance across the random draws is the barren-plateau indicator. A
    log-linear fit gives the per-qubit decay rate (slope < 0 => gradients vanish
    exponentially with system size).

    Parameters
    ----------
    qubit_range : iterable of int (e.g. range(2, 9)).
    n_layers : int, circuit depth used for the probe.
    n_samples : int, number of random parameter draws per qubit count.
    seed : int.
    entangle : bool, include the CNOT ring.
    return_grads : if True, also return the raw gradient samples per qubit count.

    Returns
    -------
    qubits : ndarray, variances : ndarray, slope : float
        (slope = d log(Var) / d n_qubits).  If ``return_grads`` also a dict.
    """
    qubits = np.array(sorted(qubit_range))
    if qubits.size < 2 or np.any(qubits < 1):
        raise ValueError("qubit_range must contain at least two positive qubit counts.")
    if n_samples < 2 or n_layers < 1:
        raise ValueError("n_samples must be >= 2 and n_layers must be >= 1.")
    variances = np.empty(qubits.size)
    raw = {}
    for idx, n in enumerate(qubits):
        core = _QReuploadCore(int(n), n_layers, entangle, input_scaling=False)
        circ = core.build(np.zeros(int(n)))        # fixed reference input
        probe = 1                                  # first block's RY angle (index 0 is RZ on |0>)
        rng = np.random.default_rng([seed, int(n)])   # independent stream per qubit count (no overlap)
        thetas = rng.uniform(-np.pi, np.pi, (n_samples, core.nq))
        grads = np.empty(n_samples)
        for s in range(n_samples):
            res = circ.get_gradients_and_expectation(thetas[s], core.observables[0],
                                                     DiffMethod.ADJOINT_DIFF)
            grads[s] = np.asarray(res.gradients())[probe]
        variances[idx] = float(np.var(grads))
        raw[int(n)] = grads
    slope = float(np.polyfit(qubits, np.log(variances + 1e-300), 1)[0])
    if return_grads:
        return qubits, variances, slope, raw
    return qubits, variances, slope
