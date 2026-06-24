# -*-coding:utf-8-*-
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
"""Unit tests for the QReupload data re-uploading variational QML module.

Placed under ``test/QAlgBase/`` and named ``Test_*.py`` to match this repo's
pytest config (``test/pytest.ini``) and every existing test; case docstrings use
the CONTRIBUTING ``Test(feature, content)`` form. Runs under pytest or directly
(``python Test_QReupload.py``).
"""
import numpy as np

from pyqpanda_alg.QReupload import (QReuploadRegressor, QReuploadClassifier,
                                    fourier_spectrum, trainability_scan)


class Test_QReupload:

    def test_regressor_fits_band_limited(self):
        """Test(QReupload, regressor reaches low error on a Fourier target)."""
        X = np.linspace(-np.pi, np.pi, 80)
        y = np.sin(2 * X) + 0.5 * np.cos(3 * X)
        reg = QReuploadRegressor(n_layers=5, epochs=250, restarts=1, seed=0).fit(X, y)
        assert reg.predict(X).shape == (X.shape[0],)
        assert np.mean((reg.predict(X) - y) ** 2) < 1e-2
        assert reg.score(X, y) > 0.95

    def test_fourier_spectrum_is_band_limited(self):
        """Test(QReupload, learned spectrum concentrates on target harmonics)."""
        X = np.linspace(-np.pi, np.pi, 80)
        y = np.sin(2 * X) + 0.5 * np.cos(3 * X)            # harmonics k=2 (amp 1), k=3 (amp 0.5)
        reg = QReuploadRegressor(n_layers=5, epochs=250, restarts=1, seed=0).fit(X, y)
        k, mag = fourier_spectrum(reg, max_harmonic=8)
        assert mag[2] > 0.7 and mag[3] > 0.3              # target frequencies present
        assert mag[1] < 0.15 and mag[6] < 0.15           # absent frequencies suppressed

    def test_entanglement_required_for_cross_frequency(self):
        """Test(QReupload, entanglement is load-bearing on a non-separable target)."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, (120, 2))
        y = np.sin(X[:, 0] + X[:, 1])                      # non-separable cross term
        ent = QReuploadRegressor(n_layers=4, entangle=True, epochs=150, restarts=1, seed=0).fit(X, y)
        noent = QReuploadRegressor(n_layers=4, entangle=False, epochs=150, restarts=1, seed=0).fit(X, y)
        e_mse = np.mean((ent.predict(X) - y) ** 2)
        n_mse = np.mean((noent.predict(X) - y) ** 2)
        assert e_mse < 0.5 * n_mse                         # entangling clearly better

    def test_binary_classifier(self):
        """Test(QReupload, binary classifier separates a periodic boundary)."""
        rng = np.random.default_rng(1)
        X = rng.uniform(-np.pi, np.pi, (120, 1))
        y = (np.sin(X[:, 0]) > 0).astype(int)
        clf = QReuploadClassifier(n_layers=4, epochs=250, restarts=1, seed=0).fit(X, y)
        assert clf.score(X, y) > 0.85
        assert clf.predict_proba(X).shape == (120, 2)

    def test_trainability_scan_detects_decay(self):
        """Test(QReupload, gradient variance shrinks with qubit count)."""
        q, var, slope = trainability_scan(range(2, 6), n_layers=3, n_samples=80, seed=0)
        assert slope < 0                                   # barren-plateau signature
        assert var[-1] < 0.5 * var[0]                      # variance clearly shrinks with qubits

    def test_multifeature_smoke(self):
        """Test(QReupload, 3-feature fit runs and predicts the right shape)."""
        rng = np.random.default_rng(2)
        X = rng.uniform(-np.pi, np.pi, (60, 3))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1] + X[:, 2])
        reg = QReuploadRegressor(n_layers=2, epochs=80, restarts=1, seed=0).fit(X, y)
        assert reg.predict(X).shape == (60,)
        assert reg.n_features_in_ == 3

    def test_predict_rejects_wrong_feature_count(self):
        """Test(QReupload, predict rejects X whose feature count != fit)."""
        X = np.linspace(-np.pi, np.pi, 20); y = np.sin(X)
        reg = QReuploadRegressor(n_layers=2, epochs=5, restarts=1, seed=0).fit(X, y)
        try:
            reg.predict(np.ones((4, 2)))                   # fitted with 1 feature
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for wrong feature count")

    def test_fit_rejects_invalid_input(self):
        """Test(QReupload, fit rejects length-mismatched or non-finite data)."""
        X = np.linspace(-np.pi, np.pi, 10)
        cases = [lambda: QReuploadRegressor(epochs=5, restarts=1).fit(X, np.sin(X)[:5]),
                 lambda: QReuploadRegressor(epochs=5, restarts=1).fit(
                     np.array([0.0, np.nan, 1.0]), np.zeros(3))]
        for bad in cases:
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("expected ValueError for invalid fit input")

    def test_get_set_params_protocol(self):
        """Test(QReupload, sklearn-style get_params/set_params round-trip)."""
        reg = QReuploadRegressor(n_layers=3)
        assert reg.get_params()["n_layers"] == 3
        reg.set_params(n_layers=7, epochs=10)
        assert reg.n_layers == 7 and reg.epochs == 10
        try:
            reg.set_params(nope=1)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown parameter")

    def test_param_count_and_input_scaling(self):
        """Test(QReupload, parameter count matches the documented formula)."""
        # d=1, n_layers=5, scaling on: nq = 3 + 5*4 = 23 ; + readout(w,b)=2 -> 25
        X = np.linspace(-np.pi, np.pi, 30); y = np.sin(X)
        on = QReuploadRegressor(n_layers=5, input_scaling=True, epochs=20, restarts=1).fit(X, y)
        off = QReuploadRegressor(n_layers=5, input_scaling=False, epochs=20, restarts=1).fit(X, y)
        assert on.n_params_ == 25
        assert off.n_params_ == 3 + 5 * 3 + 2              # no lambda params: 20
        assert off.n_params_ < on.n_params_


if __name__ == "__main__":
    t = Test_QReupload()
    for name in [m for m in dir(t) if m.startswith("test_")]:
        getattr(t, name)()
        print(f"  PASS  {name}")
    print("ALL TESTS PASSED")
