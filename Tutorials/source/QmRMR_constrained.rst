Constraint-preserving QAOA for QmRMR
====================================

The existing ``pyqpanda_alg.QmRMR.Feature_Selection`` interface is preserved
and extended with ``QmRMRFeatureSelection``.  The new implementation enforces
the exact feature-cardinality constraint by preparing a Dicke state and using
a number-preserving XY mixer.

For a redundancy matrix :math:`Q`, relevance vector :math:`r`, and target
cardinality :math:`k`, the model is

.. math::

   f(x)=x^{\mathsf T}Qx-r^{\mathsf T}x,
   \qquad x_i\in\{0,1\},\qquad \sum_i x_i=k.

The state-vector implementation reports exact feasible probability, selected
state and expectation gaps, and gate/depth metadata.  Exact enumeration,
greedy forward selection, and the original exchange-style ansatz are retained
as transparent comparison baselines.

The runnable example is
``pyqpanda-algorithm/example/QAlgBase/QmRMR/testeg_QmRMR_constrained.py``;
the full benchmark and Notebook are under the adjacent ``constrained``
directory.


