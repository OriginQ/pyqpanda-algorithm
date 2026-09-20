Quantum State Tomography
========================

The ``pyqpanda_alg.QST`` module reconstructs an n-qubit density matrix from
complete local Pauli measurements.  It is backend agnostic: execute the
measurement circuits with PyQPanda3, a simulator, or hardware, then pass the
returned count dictionaries into the reconstruction routine.

Measurement acquisition
-----------------------

Call ``pauli_measurement_plan(n)`` to obtain the ``3**n`` product-basis
settings.  Each character describes the measurement axis for one logical
qubit:

* ``Z``: measure directly in the computational basis.
* ``X``: apply H, then measure in the computational basis.
* ``Y``: apply S-dagger followed by H, then measure in the computational basis.

Keep the bit-string order aligned with the basis-string order.  Raw shot
counts can be supplied directly; every basis distribution is normalized
independently.

Reconstruction
--------------

.. code-block:: python

    from pyqpanda_alg.QST import pauli_measurement_plan
    from pyqpanda_alg.QST import linear_inversion_tomography

    plan = pauli_measurement_plan(2)
    # Populate one counts dictionary per basis with a PyQPanda backend.
    counts = {
        "XX": {"00": 510, "11": 490},
        # ... XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
    }
    rho = linear_inversion_tomography(counts)

Full linear inversion expands the state in the Pauli basis.  With
``physical=True`` (the default), finite-shot negative eigenvalues are clipped
and the result is renormalized to a positive, trace-one density matrix.

Deterministic reference data
----------------------------

``probabilities_from_density_matrix`` converts a known density matrix into
ideal probabilities for any X/Y/Z basis.  This is useful for simulator
examples and reference datasets.  See
``pyqpanda-algorithm/example/QAlgBase/testeg_QST.py`` for a Bell-state
end-to-end example.
