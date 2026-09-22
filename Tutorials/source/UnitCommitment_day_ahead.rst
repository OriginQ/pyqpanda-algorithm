Day-ahead unit commitment on a five-bus benchmark
==================================================

``pyqpanda_alg.UnitCommitment`` provides a compact power-system application
built on the repository's QAOA implementation.  It models a PJM five-bus
benchmark, solves a 24-hour unit-commitment reference with SciPy, and compares
it with one five-qubit QAOA commitment solve per hour.

The public entry point is
``pyqpanda-algorithm/example/QAlgBase/UnitCommitment/run_day_ahead.py``.  The
same workflow is documented cell by cell in
``pyqpanda-algorithm/example/QAlgBase/UnitCommitment/notebooks/case5_day_ahead_workflow.ipynb``.

The dispatch reference uses the daily objective

.. math::

   \min_{u,y,p}\sum_{t=0}^{23}\left[\sum_g(a_gp_{g,t}^2+b_gp_{g,t})
   +F_gu_{g,t}+S_gy_{g,t}\right],

with hourly balance, reserve, generator-limit, start-up and DC line-flow
constraints.  The QAOA stage supplies commitment candidates; the dispatch
and network checks remain classical and auditable.

The archived figures and JSON record are kept in the example directory.  The
optional Runtime FakeBackend adapter accepts a caller-owned API key through an
environment variable, but this example does not submit a real-quantum job.


