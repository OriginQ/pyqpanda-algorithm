import pytest
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_pkg_root = _project_root / "pyqpanda-algorithm"
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_pkg_root))

import numpy as np
from pyqpanda3.core import QCircuit, Z, X, RZ
from pyqpanda_alg.QPE import QPE


class TestQPE:
    """Tests for the Quantum Phase Estimation algorithm."""

    def test_initialization(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=4, n_target=1)
        assert qpe.n_count == 4
        assert qpe.n_target == 1
        assert qpe.n_qubits == 5

    def test_build_circuit(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=3, n_target=1)
        cir = qpe.build_circuit()
        assert cir is not None

    def test_phase_z_zero(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=4, n_target=1)
        phi = qpe.run()
        assert abs(phi - 0.0) < 0.1

    def test_phase_z_one(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        def prep_one(tq):
            from pyqpanda3.core import X
            cir = QCircuit()
            cir << X(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=4, n_target=1, state_prep=prep_one)
        phi = qpe.run()
        assert abs(phi - 0.5) < 0.1

    def test_phase_rotation(self):
        def u_rz(tq):
            cir = QCircuit()
            cir << RZ(tq[0], np.pi / 2)
            return cir
        qpe = QPE(unitary=u_rz, n_count=4, n_target=1)
        phi = qpe.run()
        assert abs(phi - 0.875) < 0.1

    def test_run_dict(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=2, n_target=1)
        d = qpe.run_dict()
        assert isinstance(d, dict)
        assert len(d) > 0
        assert abs(sum(d.values()) - 1.0) < 1e-10

    def test_shots_mode(self):
        def u_z(tq):
            cir = QCircuit()
            cir << Z(tq[0])
            return cir
        qpe = QPE(unitary=u_z, n_count=4, n_target=1)
        phi = qpe.run(shots=1000)
        assert abs(phi - 0.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
