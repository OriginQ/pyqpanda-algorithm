"""HHL example smoke test: the published example runs on the CPU.

The example is executed in-process with ``runpy`` exactly as a user
would run it, without any qpanda3-runtime installation or credentials.
The CPU solve of ``diag([1, 2]) x = [1, 1]`` must reproduce the exact
solution direction with a small residual against the original system.
"""

import runpy


def test_hhl_example_solves_on_local_cpu_backend():
    namespace = runpy.run_path("pyqpanda-algorithm/example/HHL/solve_linear_system.py")
    assert namespace["result"].residual < 0.05
    assert namespace["result"].success_probability > 0.0
