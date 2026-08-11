"""VQE example smoke test: the published example runs on the CPU.

The example is executed in-process with ``runpy`` exactly as a user
would run it, without any qpanda3-runtime installation or credentials.
The default path must use the local CPU backend and converge below
the ground-state threshold of the one-qubit Z observable.
"""

import runpy


def test_vqe_example_reaches_expected_energy():
    namespace = runpy.run_path("pyqpanda-algorithm/example/VQE/vqe_hamiltonian.py")
    assert namespace["result"].energy < -0.99
