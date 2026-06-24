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
"""Worked example for the QReupload data re-uploading variational QML module.

Three parts:
  A. 1-feature regression + Fourier-spectrum readout (which frequencies did the
     quantum model learn?), against a degree-matched classical Fourier baseline.
  B. 2-feature regression on the QSVR-style target EXTENDED with a non-separable
     cross term, y = 2 sin(x0) + 1.5 cos(x1) + sin(x0 + x1), which makes
     entanglement load-bearing: an entangle vs no-entangle ablation, with a
     classical full-Fourier model shown as an *oracle upper bound* (it is handed
     the exact cross-frequency basis a priori).
  C. Trainability (barren-plateau) scan.

Honesty note: on band-limited data a Fourier-aware, properly-regularised
classical model is a strong baseline; we report it alongside rather than hiding
it. The quantum model's value here is the trainable encoding (it learns the
cross term without being given a cross-frequency basis), the readable spectrum,
and the built-in trainability warning — not beating that oracle on raw error.
"""
import os
import numpy as np
from sklearn.linear_model import RidgeCV

from pyqpanda_alg.QReupload import QReuploadRegressor, fourier_spectrum, trainability_scan

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_outputs")
os.makedirs(RESULTS, exist_ok=True)


# ---- honest classical baseline: linear regression on Fourier features, CV-ridged ----
def fourier_features(X, degree, cross=False):
    """Per-dimension Fourier features, or the full tensor-product basis (cross=True,
    needed to represent cross-frequency terms like sin(x0 + x1))."""
    if X.ndim == 1:
        X = X[:, None]
    per = []
    for j in range(X.shape[1]):
        cols = [np.ones(len(X))]
        for k in range(1, degree + 1):
            cols += [np.cos(k * X[:, j]), np.sin(k * X[:, j])]
        per.append(np.column_stack(cols))
    if not cross:
        return np.column_stack([np.ones(len(X))] + [p[:, 1:] for p in per])
    feat = per[0]
    for j in range(1, len(per)):
        feat = np.einsum("na,nb->nab", feat, per[j]).reshape(len(X), -1)
    return feat


def fourier_ridge_baseline(Xtr, ytr, Xte, degree, cross=False):
    Ftr, Fte = fourier_features(Xtr, degree, cross), fourier_features(Xte, degree, cross)
    model = RidgeCV(alphas=np.logspace(-6, 2, 25)).fit(Ftr, ytr)
    return model.predict(Fte), Ftr.shape[1]


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def part_a():
    print("\n[A] 1-feature regression + learned Fourier spectrum")
    rng = np.random.default_rng(0)
    f = lambda x: 1.5 * np.sin(2 * x) - np.cos(4 * x)
    Xtr = rng.uniform(-np.pi, np.pi, 200); ytr = f(Xtr) + rng.normal(0, 0.05, 200)
    Xte = np.linspace(-np.pi, np.pi, 1000); yte = f(Xte)

    reg = QReuploadRegressor(n_layers=6, epochs=500, restarts=2, seed=0).fit(Xtr, ytr)
    q_mse = mse(reg.predict(Xte), yte)
    base, bp = fourier_ridge_baseline(Xtr, ytr, Xte, degree=6)
    print(f"  QReupload   : test MSE={q_mse:.4e}  ({reg.n_params_} params)")
    print(f"  Fourier-ridge: test MSE={mse(base, yte):.4e}  ({bp} params, honest baseline)")
    k, mag = fourier_spectrum(reg, max_harmonic=7)
    print("  learned spectrum (target has k=2 @1.5, k=4 @1.0):")
    for kk, mm in zip(k, mag):
        print(f"    k={kk:>2} |c_k|={mm:5.3f} {'#' * int(mm * 30)}")
    return reg, Xte, yte


def part_b():
    # QSVR-style target EXTENDED with a non-separable cross term sin(x0+x1).
    # A no-entangle model reads out sum_q g_q(x_q) (separable) and CANNOT fit the
    # cross term; the entangling ring can. The classical full-Fourier model below
    # is an ORACLE: it is handed the exact cross-frequency basis a priori.
    print("\n[B] 2-feature target  y = 2 sin(x0) + 1.5 cos(x1) + sin(x0 + x1)  (cross term non-separable)")
    rng = np.random.default_rng(1)
    f = lambda X: 2 * np.sin(X[:, 0]) + 1.5 * np.cos(X[:, 1]) + np.sin(X[:, 0] + X[:, 1])
    Xtr = rng.uniform(-np.pi, np.pi, (500, 2)); ytr = f(Xtr) + rng.normal(0, 0.05, 500)
    Xte = rng.uniform(-np.pi, np.pi, (2000, 2)); yte = f(Xte)
    res = {}
    for ent in (True, False):
        reg = QReuploadRegressor(n_layers=4, entangle=ent, epochs=500, restarts=2, seed=0).fit(Xtr, ytr)
        res[ent] = mse(reg.predict(Xte), yte)
        print(f"  QReupload entangle={ent!s:5}: test MSE={res[ent]:.4e}  ({reg.n_params_} params)")
    print(f"  >>> entanglement effect: {res[False] / res[True]:.0f}x lower error WITH the CNOT ring")
    print(f"      (the quantum model learns the cross term WITHOUT being given a cross-frequency basis)")
    base, bp = fourier_ridge_baseline(Xtr, ytr, Xte, degree=2, cross=True)
    print(f"  [oracle] classical Ridge GIVEN the exact 2D Fourier basis: {mse(base, yte):.4e}  ({bp} params)")


def part_c():
    print("\n[C] trainability (barren-plateau) scan")
    q, var, slope = trainability_scan(range(2, 8), n_layers=3, n_samples=150, seed=0)
    for qq, vv in zip(q, var):
        print(f"  n_qubits={qq}  Var[grad]={vv:.4e}")
    print(f"  log-slope = {slope:.3f}/qubit -> "
          f"{'barren plateau: avoid scaling qubits blindly' if slope < -0.3 else 'no strong decay'}")
    return q, var, slope


def make_plot(reg, Xte, yte, q, var, slope):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        order = np.argsort(Xte)
        ax[0].plot(Xte[order], yte[order], "k-", lw=2, label="target")
        ax[0].plot(Xte[order], reg.predict(Xte)[order], "r--", lw=1.5, label="QReupload")
        ax[0].set_title("(A) 1-feature fit"); ax[0].legend(); ax[0].set_xlabel("x")
        k, mag = fourier_spectrum(reg, max_harmonic=7)
        ax[1].bar(k, mag, color="steelblue")
        ax[1].set_title("(A) learned Fourier spectrum"); ax[1].set_xlabel("harmonic k"); ax[1].set_ylabel("|c_k|")
        ax[2].semilogy(q, var, "o-")
        ax[2].set_title(f"(C) barren plateau (slope={slope:.2f}/qubit)")
        ax[2].set_xlabel("# qubits"); ax[2].set_ylabel("Var[grad]"); ax[2].grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(RESULTS, "qreupload_example.png")
        fig.savefig(path, dpi=130)
        print(f"\n[plot] {path}")
    except Exception as e:
        print(f"[plot] skipped ({e})")


if __name__ == "__main__":
    reg, Xte, yte = part_a()
    part_b()
    q, var, slope = part_c()
    make_plot(reg, Xte, yte, q, var, slope)
    print("\nDONE")
