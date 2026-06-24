QReupload: Data Re-uploading Variational Quantum Learning
==========================================================

``QReupload`` is the first ``pyqpanda_alg`` machine-learning estimator whose
**circuit parameters are trained against a supervised task loss** (sklearn-style
``fit``). ``QSVM``/``QSVR`` fit only a classical SVM over a *parameter-free*
quantum feature map; ``QReupload`` is a **trainable** data re-uploading circuit
with a learnable input-scaling factor, an optional entangling ring, exact
adjoint-gradient training, and two diagnostics that make the model's behaviour
inspectable.

It exposes two estimators and two diagnostics:

.. list-table::

    * - object
      - purpose
    * - ``QReuploadRegressor``
      - sklearn-style ``fit / predict / score`` regressor
    * - ``QReuploadClassifier``
      - binary classifier (``predict_proba``, sigmoid + cross-entropy)
    * - ``fourier_spectrum(model)``
      - recover which Fourier harmonics a fitted 1-feature model learned
    * - ``trainability_scan(qubit_range)``
      - barren-plateau diagnostic: ``Var[d<Z>/dtheta]`` vs qubit count


Background
>>>>>>>>>>>>>>>>>

A data re-uploading model interleaves data-encoding rotations with trainable
single-qubit blocks. With one feature per qubit and ``n_layers`` re-uploads it
realises a **truncated Fourier series of controllable degree** in the inputs
(Schuld, Sweke & Meyer, *PRA* 103, 032430, 2021). A trainable input-scaling
factor lets the model align its accessible frequencies to the data; an
entangling ring lets it represent multi-feature **cross-frequency** terms that a
sum of per-qubit readouts cannot.

Training uses ``pyqpanda3.vqcircuit.VQCircuit`` with ``DiffMethod.ADJOINT_DIFF``
(exact analytic gradients), optimised with a NumPy Adam and a cosine-annealed
learning rate.


Quick start: fit a band-limited target
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: python

    import numpy as np
    from pyqpanda_alg.QReupload import QReuploadRegressor

    X = np.linspace(-np.pi, np.pi, 80)
    y = np.sin(2 * X) + 0.5 * np.cos(3 * X)

    reg = QReuploadRegressor(n_layers=5).fit(X, y)
    print(reg.score(X, y))          # R^2 ~ 0.9999

A single feature maps to a single qubit. ``n_layers`` controls the highest
Fourier harmonic the model can reach, so pick it to cover the frequencies in
your target.


Reading the learned spectrum
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Because a 1-feature model *is* a truncated Fourier series, you can read back
exactly which harmonics it learned:

.. code-block:: python

    from pyqpanda_alg.QReupload import fourier_spectrum

    k, mag = fourier_spectrum(reg, max_harmonic=8)
    for kk, mm in zip(k, mag):
        print(f"k={kk:>2}  |c_k|={mm:5.3f}")

For the target ``1.5*sin(2x) - cos(4x)`` the spectrum concentrates on the two
expected harmonics and suppresses the rest::

    k= 2  |c_k|=1.505
    k= 4  |c_k|=0.998
    (all other |c_k| < 0.01)

With *unit* input scaling these magnitudes coincide with the truncated Fourier
series; with a *trainable* scaling factor read this as an empirical DFT over the
interval (a learned scaling can move peaks off the integer grid).


When does entanglement matter?
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

A no-entangle model reads out a sum of per-qubit functions and therefore
*cannot* represent a non-separable cross term such as ``sin(x0 + x1)``. The CNOT
ring can. On ``y = 2*sin(x0) + 1.5*cos(x1) + sin(x0 + x1)``:

.. code-block:: python

    import numpy as np
    from pyqpanda_alg.QReupload import QReuploadRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, (500, 2))
    y = 2*np.sin(X[:, 0]) + 1.5*np.cos(X[:, 1]) + np.sin(X[:, 0] + X[:, 1])

    ent   = QReuploadRegressor(n_layers=4, entangle=True ).fit(X, y)
    noent = QReuploadRegressor(n_layers=4, entangle=False).fit(X, y)

The entangling model reaches a much lower error (roughly a 40x gap on held-out
data), learning the cross term **without being handed a cross-frequency basis**.


Trainability: a built-in barren-plateau warning
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Local-observable readouts have gradient variance that decays roughly
exponentially in qubit count. ``trainability_scan`` measures the rate for your
configuration so you scale qubits deliberately, not blindly:

.. code-block:: python

    from pyqpanda_alg.QReupload import trainability_scan

    qubits, variances, slope = trainability_scan(range(2, 8), n_layers=3)
    print(slope)        # e.g. ~ -0.70  (log-variance per qubit) -> barren plateau

A clearly negative slope is the barren-plateau signature: each extra qubit makes
the model exponentially harder to train, so prefer a few well-chosen qubits over
many.


A note on honesty
>>>>>>>>>>>>>>>>>>>>>

On band-limited data a Fourier-aware, properly-regularised classical model is a
strong baseline; ``example/QAlgBase/testeg_qreupload.py`` reports it side by side
rather than hiding it. The value of ``QReupload`` is the *trainable encoding*,
the *readable spectrum*, and the *built-in trainability warning* — not beating
that baseline on raw error.


See also
>>>>>>>>>>>>

* ``example/QAlgBase/testeg_qreupload.py`` — runnable three-part demo (spectrum
  readout, entanglement ablation, barren-plateau scan).
* ``test/QAlgBase/Test_QReupload.py`` — unit tests covering behaviour and the
  sklearn-style estimator contract.
* API reference: :doc:`autoapi/pyqpanda_alg/QReupload/index`.
