2.1.0 - development
====================

This entry documents the 2.1.0 development baseline. The package version
is single-sourced from ``pyqpanda_alg.__version__``, and the Sphinx
documentation metadata reports the same version.

Features
---------------------

* The new ``pyqpanda_alg.execution`` package provides a capability-based
  execution layer.  By default, algorithms execute synchronously on the
  local CPU backend (``LocalBackend``) through pyqpanda3's ``CPUQVM``;
  existing valid CPU calls remain compatible and keep this default.
* An optional qpanda3-runtime backend (``QPandaRuntimeBackend``) submits
  work to a remote qpanda3-runtime service.  The dependency is optional
  and installed via the ``runtime`` extra (``pip install
  pyqpanda_alg[runtime]``); a runtime failure never silently falls back
  to the CPU backend or to a classical substitute result.
* Remote runtime qualification and real-hardware support are planned in
  later releases; this baseline does not claim qualified support for
  real quantum hardware.
* ``QPCA.qpca`` default sampling shots changed from 8192 to the
  execution layer's 1000.  Results are normalized by the actual shot
  count, so the output shape is preserved.
* ``QuantumAssociationRulesMining`` (QARM) entry points: the legacy
  ``machine_type='QCloud'`` mode is deprecated.  It warns and now
  requires an explicit ``backend`` argument; ``machine_type='CPU'``
  remains the default.

2.0 - 2025-10-25
===================

Features
---------------------
With the upgrade of the pyqpanda technical architecture, we have decided to migrate the underlying SDK version of the pyqpanda_alg algorithm package from pyqpanda to pyqpanda3. 

This upgrade aims to improve algorithm execution efficiency, enhance functional support capabilities, and optimize compatibility with quantum computing platforms. 

To ensure a smooth transition for users and full utilization of the new version's features, the online documentation for pyqpanda_alg will undergo comprehensive updates to reflect interface adjustments, functional differences, and best practices resulting from the SDK change.


0.2 - 2025-05-20
===================

In our latest update, we have introduced four new algorithms.

Features
---------------------

 **Quantum Singular Value Decomposition (QSVD)** : QSVD is a quantum adaptation of the classical singular value decomposition technique, enabling efficient extraction of principal components from quantum data. It offers exponential speedup for certain matrix operations and plays a foundational role in quantum machine learning and quantum image processing.

 **Quantum Support Vector Regression (QSVR)** : QSVR brings the principles of support vector regression into the quantum domain, enabling efficient fitting of nonlinear functions using quantum-enhanced kernels. It holds promise for accelerated regression tasks in data modeling, finance, and scientific forecasting.

 **Quantum Sparse State Encoding** : This algorithm provides an efficient method for encoding sparse classical data into quantum states with logarithmic resource requirements. It is essential for loading structured datasets into quantum memory, supporting a variety of algorithms such as quantum search, simulation, and optimization.

 **Quantum Minimum Redundancy Maximum Relevance (QmRMR)** : QmRMR is a quantum feature selection algorithm that identifies the most informative and least redundant features in a dataset. Designed for quantum dimensionality reduction, it boosts the performance of quantum classifiers and regression models while reducing noise and computation overhead.


0.1 - 2023-11-16
===================

In our latest update, we have introduced two new algorithms: Shor's algorithm and the HHL algorithm.

Features
---------------------

 **Shor's algorithm** , pioneered by mathematician Peter Shor, is a quantum algorithm that revolutionizes the field of number theory by efficiently factoring large numbers. This breakthrough has profound implications for cryptography, as it challenges the security of widely-used encryption schemes relying on the difficulty of factoring large numbers.

 **The HHL (Harrow-Hassidim-Lloyd) algorithm** , spearheaded by researchers Aram Harrow, Avinatan Hassidim, and Seth Lloyd, is a quantum algorithm designed to solve linear systems of equations. This algorithm offers a quadratic speedup over the best-known classical algorithms for specific types of problems. Its applications extend across diverse fields, including optimization, machine learning, and scientific simulations.





