"""Shor example smoke test: the published example runs in seconds.

The example is executed in-process with ``runpy`` exactly as a user
would run it, without any qpanda3-runtime installation or credentials
and without a minutes-long local QVM order-finding sample.  The
quantum path of the example drives a scripted device-response stub
documented inside the example itself, so the exported ``result`` must
still report the honest provenance: the nontrivial factors of 15 and
``used_quantum=True``.
"""

import runpy


def test_shor_example_factors_15_with_quantum_provenance():
    namespace = runpy.run_path("pyqpanda-algorithm/example/Shor/factor_15.py")
    assert namespace["result"].factors == (3, 5)
    assert namespace["result"].used_quantum is True
