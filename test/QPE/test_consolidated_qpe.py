"""Native coverage for both preserved APIs and the shared phase engine."""

from math import pi

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, RZ, U1, X

from pyqpanda_alg.QPE import (QPE, StructuredQPE, PhaseEstimationResult,
                             decode_phase, phase_to_eigenvalue)


def prepare_one(qubits):
    return QCircuit() << X(qubits[0])


def unitary(phase):
    return lambda qubits: QCircuit() << U1(qubits[0], 2 * pi * phase)


@pytest.mark.parametrize("phase", [0., 1/8, 1/4, 3/8, 1/2, 3/4, 7/8])
def test_native_distinct_generic_phases_exact_and_sampled(phase):
    canonical = QPE(unitary(phase), 3, 1, prepare_one)
    adapter = StructuredQPE(unitary(phase), 1, 3, prepare_one, shots=37)
    assert canonical.run() == phase
    assert canonical.run(shots=37) == phase
    exact = adapter.run_exact()
    sampled = adapter.run()
    assert exact.phase == sampled.phase == phase
    assert exact.sampling_shots is None
    assert sampled.sampling_shots == 37
    assert exact.probability == pytest.approx(1., abs=1e-14)
    assert sampled.probability == 1.
    assert sampled.bitstring == format(int(phase * 8), "03b")


def test_original_human_rz_convention():
    rotation = lambda qubits: QCircuit() << RZ(qubits[0], pi / 2)
    assert QPE(rotation, n_count=4).run() == 7/8
    assert StructuredQPE(rotation, 1, 4).run_exact().phase == 7/8


def test_actual_shot_histograms_and_upper_register_layout():
    phase = .3  # off-grid: exact distribution cannot equal any 37-shot histogram
    adapter = StructuredQPE(unitary(phase), 2, 4, prepare_one, shots=37)
    program, counting = adapter.build_program()
    assert counting == [2, 3, 4, 5]
    sampled = adapter.run()
    exact = adapter.run_exact()
    assert set(map(len, sampled.probabilities)) == {4}
    assert sum(sampled.probabilities.values()) == pytest.approx(1.)
    for probability in sampled.probabilities.values():
        assert probability * 37 == pytest.approx(round(probability * 37), abs=1e-12)
    assert any(abs(p * 37 - round(p * 37)) > 1e-4
               for p in exact.probabilities.values())
    canonical = QPE(unitary(phase), 4, 2, prepare_one)
    for probability in canonical.run_dict(shots=37).values():
        assert probability * 37 == pytest.approx(round(probability * 37), abs=1e-12)


@pytest.mark.parametrize("precision", [3, 6, 8])
def test_factory_constructed_once_per_build(precision):
    calls = []
    def counted(qubits):
        calls.append(tuple(qubits))
        return unitary(1/4)(qubits)
    QPE(counted, precision, 1).build_circuit()
    assert len(calls) == 1
    calls.clear()
    StructuredQPE(counted, 1, precision).build_program()
    assert len(calls) == 1


def test_efficient_power_hook_matches_repetition_and_skips_base():
    powers = []
    def forbidden(_):
        raise AssertionError("base factory should not be called with power hook")
    def powered(qubits, power):
        powers.append(power)
        return unitary((3/8) * power)(qubits)
    solver = QPE(forbidden, 4, 1, prepare_one, unitary_power=powered)
    assert solver.run() == 3/8
    assert powers == [1, 2, 4, 8]
    powers.clear()
    adapter = StructuredQPE(forbidden, 1, 4, prepare_one, powered, shots=41)
    assert adapter.run().phase == 3/8
    assert powers == [1, 2, 4, 8]


def test_adapter_mutable_configuration_and_original_constructor_shapes():
    from pyqpanda_alg.QPE.compat import QPE as CompatibilityQPE

    solver = CompatibilityQPE(unitary(1/4), 1, 3, prepare_one, None, 31)
    assert solver.run().phase == 1/4
    solver.unitary = unitary(3/4)
    solver.shots = 43
    solver.precision_qubits = 4
    solver.system_qubits = 2
    result = solver.run()
    assert result.phase == 3/4 and result.sampling_shots == 43
    assert len(result.bitstring) == 4
    assert solver.build_program()[1] == [2, 3, 4, 5]
    manual = PhaseEstimationResult(.25, "01", 1., .25, {"01": 1.})
    assert manual.sampling_shots is None
    assert isinstance(QPE(unitary(.25), 2, 1, prepare_one).run(), float)


def test_generic_phase_distribution_does_not_fold_mirror_bins():
    result = decode_phase({"001": .3, "111": .3, "010": .4}, 3)
    assert result.phase == .25 and result.probability == .4
    assert result.probabilities == pytest.approx({"001": .3, "111": .3, "010": .4})
    histogram = decode_phase({"001": 10, "111": 30}, 3, sampling_shots=40)
    assert histogram.phase == 7/8 and histogram.probability == .75
    scaled = decode_phase({"001": 1e308, "111": 1e308}, 3)
    assert scaled.probabilities == {"001": .5, "111": .5}


@pytest.mark.parametrize("weights", [
    {"000": 1, "bad": 0}, {"000": 1, "01": 0},
    {"000": 1, "111": -1}, {"000": 1, "111": np.nan},
    {"000": 1, "111": np.inf}, {"000": 0}, {},
])
def test_every_readout_entry_validated(weights):
    with pytest.raises(ValueError):
        decode_phase(weights, 3)


@pytest.mark.parametrize("time", [np.nan, np.inf, -np.inf, 0., -1.])
def test_nonfinite_or_nonpositive_evolution_time_rejected(time):
    with pytest.raises(ValueError):
        phase_to_eigenvalue(.25, time)


def test_eigenvalue_branch_and_helper_imports():
    from pyqpanda_alg.QPE.qpe import phase_to_eigenvalue as old_import
    assert old_import(.75, centered=True) == -pi / 2
    assert phase_to_eigenvalue(.75) == 3 * pi / 2


def test_eigenvalue_ratio_handles_subnormal_phase_and_time():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    assert phase_to_eigenvalue(tiny, tiny) == 2 * pi
    assert phase_to_eigenvalue(tiny, 2.) == 3 * tiny
    assert phase_to_eigenvalue(0., tiny) == 0.


def test_unrepresentable_eigenvalue_rejected():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    with pytest.raises(ValueError, match="not representable"):
        phase_to_eigenvalue(.25, tiny)


@pytest.mark.parametrize("shots", [True, False, 0, -2, 1.5, -1.0])
def test_invalid_shot_counts_rejected(shots):
    with pytest.raises(ValueError):
        QPE(unitary(.25)).run(shots=shots)


def test_invalid_factories_rejected_before_native_execution():
    with pytest.raises(TypeError):
        QPE(3)
    with pytest.raises(TypeError, match="return QCircuit"):
        QPE(lambda _: None).build_circuit()
    with pytest.raises(TypeError, match="return QCircuit"):
        QPE(unitary(.25), unitary_power=lambda *_: None).build_circuit()
