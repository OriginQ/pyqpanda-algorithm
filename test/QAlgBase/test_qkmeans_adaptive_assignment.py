"""Counts-contract tests; CPUQVM is an independent circuit reference only."""
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest
from pyqpanda3.core import CPUQVM, H, QProg, SWAP, U3, X

from pyqpanda_alg.QKmeans import QuantumKmeans, adaptive_assign, build_assignment_program
from pyqpanda_alg.QKmeans.adaptive_assignment import AssignmentResult, _append_fredkin


POINT = (0.0, -1.0)
CENTERS = ((0.0, -1.0), (0.0, 0.0), (0.0, 1.0))


class CountsRunner:
    """Deterministic contract fixture, not a physical sampling experiment."""
    def __init__(self, point=POINT, centers=CENTERS, fractions=(0.0, 0.5, 1.0)):
        self.templates = [build_assignment_program(point, c).originir() for c in centers]
        self.fractions = fractions
        self.calls = []
        self.programs = []

    def __call__(self, program, shots):
        index = self.templates.index(program.originir())
        self.calls.append((index, shots))
        self.programs.append(program)
        ones = int(shots * self.fractions[index])
        return {"0": shots - ones, "1": ones}


def test_exports_and_single_candidate_no_shots():
    assert callable(QuantumKmeans)
    result = adaptive_assign(POINT, [CENTERS[0]], lambda *_: pytest.fail("no shots"))
    assert isinstance(result, AssignmentResult)
    assert (result.status, result.winner, result.active) == ("single_candidate", 0, (0,))
    assert result.means == (None,)
    assert result.intervals == ((0.0, 1.0),)
    assert result.shots_per_centroid == (0,) and result.total_shots == 0
    with pytest.raises(FrozenInstanceError):
        result.winner = 1


@pytest.mark.parametrize("budget,expected", [(0, (0, 0, 0)), (63, (0, 0, 0)),
    (64, (64, 0, 0)), (191, (64, 64, 0)), (192, (64, 64, 64)),
    (257, (128, 64, 64))])
def test_global_budget_whole_batches_and_unmeasured(budget, expected):
    runner = CountsRunner(fractions=(0.5, 0.5, 0.5))
    result = adaptive_assign(POINT, CENTERS, runner, max_shots=budget)
    assert result.status == "undecided" and result.winner is None
    assert result.shots_per_centroid == expected
    assert result.total_shots == sum(expected) <= budget
    assert all(shots == 64 for _, shots in runner.calls)
    assert result.active == (0, 1, 2)
    for i, shots in enumerate(expected):
        if shots == 0:
            assert result.means[i] is None and result.intervals[i] == (0, 1)


def test_union_bound_radius_and_no_half_probability_clipping():
    runner = CountsRunner(fractions=(0.75, 0.75, 0.75))
    result = adaptive_assign(POINT, CENTERS, runner, batch_shots=64, max_shots=384)
    radius = math.sqrt(math.log(2 * 3 * 6 / 0.05) / (2 * 128))
    assert result.means == (0.75,) * 3
    for interval in result.intervals:
        assert interval == pytest.approx((0.75 - radius, min(1.0, 0.75 + radius)))
        assert interval[1] > 0.5


def test_elimination_retains_cumulative_statistics_and_stops_sampling_loser():
    runner = CountsRunner(fractions=(0.0, 0.0, 1.0))
    result = adaptive_assign(POINT, CENTERS, runner, batch_shots=64, max_shots=1024)
    assert result.status == "undecided" and result.winner is None
    assert result.active == (0, 1)
    assert result.shots_per_centroid == (512, 448, 64)
    assert result.means == (0.0, 0.0, 1.0)
    assert result.total_shots == 1024
    assert [i for i, _ in runner.calls].count(2) == 1


def test_separated_does_not_spend_remaining_budget():
    runner = CountsRunner(centers=CENTERS[:2], fractions=(0.0, 1.0))
    result = adaptive_assign(POINT, CENTERS[:2], runner, max_shots=4096)
    assert (result.status, result.winner, result.active) == ("separated", 0, (0,))
    assert result.shots_per_centroid == (64, 64)
    assert result.total_shots == 128
    assert result.intervals[1][0] > result.intervals[0][1]


def test_identical_centers_remain_unresolved_without_assuming_noiselessness():
    calls = []
    def run(program, shots):
        calls.append(shots)
        return {"0": shots}
    result = adaptive_assign(POINT, [POINT, POINT], run, batch_shots=32, max_shots=129)
    assert result.status == "undecided" and result.winner is None
    assert result.active == (0, 1) and result.total_shots == 128
    assert len(calls) == 4


def test_each_look_uses_all_previous_counts():
    calls = [0, 0]
    templates = [build_assignment_program(POINT, c).originir() for c in CENTERS[:2]]
    def run(program, shots):
        i = templates.index(program.originir())
        calls[i] += 1
        ones = shots * (5 if calls[i] == 1 else 3) // 8
        return {"1": ones, "0": shots - ones}
    result = adaptive_assign(POINT, CENTERS[:2], run, max_shots=256)
    assert result.means == (0.5, 0.5)
    assert result.shots_per_centroid == (128, 128)
    assert result.winner is None


def test_input_and_program_mutation_are_isolated_and_calls_are_fresh():
    point = list(POINT)
    centers = [list(c) for c in CENTERS[:2]]
    expected = [build_assignment_program(POINT, c).originir() for c in CENTERS[:2]]
    programs = []
    seen = []
    def run(program, shots):
        seen.append(program.originir())
        programs.append(program)
        point[:] = [float("nan"), 9]
        if centers:
            centers[1][:] = [9, 9]
            centers.clear()
        program.clear()
        return {"0": shots}
    result = adaptive_assign(point, centers, run, max_shots=256)
    assert seen == expected * 2
    assert len({id(p) for p in programs}) == 4
    assert result.total_shots == 256 and result.means == (0.0, 0.0)


def test_no_cross_call_statistics_or_program_cache():
    first = CountsRunner(centers=CENTERS[:2], fractions=(0.0, 0.0))
    second = CountsRunner(centers=CENTERS[:2], fractions=(1.0, 1.0))
    a = adaptive_assign(POINT, CENTERS[:2], first, max_shots=128)
    b = adaptive_assign(POINT, CENTERS[:2], second, max_shots=128)
    assert a.means == (0.0, 0.0) and b.means == (1.0, 1.0)
    assert a.total_shots == b.total_shots == 128
    assert not {id(p) for p in first.programs} & {id(p) for p in second.programs}


@pytest.mark.parametrize("bad", [None, [], {}, {"0": 63}, {"0": 65}, {"0": -1, "1": 65},
    {"0": True, "1": 63}, {"0": 64.0}, {"0": "64"}, {0: 64}, {False: 64},
    {"00": 64}, {"c0": 64}, {"0": 64, "other": 0}, {"1": float("nan")},
    {"1": np.bool_(True)}])
def test_invalid_counts_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        adaptive_assign(POINT, CENTERS, lambda *_: bad, max_shots=64)


@pytest.mark.parametrize("counts,mean", [({"0": 64}, 0.0), ({"1": 64}, 1.0),
    ({"0": np.int64(32), "1": np.int64(32)}, 0.5)])
def test_valid_single_bit_integer_counts(counts, mean):
    result = adaptive_assign(POINT, CENTERS, lambda *_: counts, max_shots=64)
    assert result.means == (mean, None, None)


@pytest.mark.parametrize("bad", [None, [], [0], [0, 0, 0], "00", [True, 0],
    [np.bool_(False), 0], ["0", 0], [complex(0), 0], [float("nan"), 0],
    [float("inf"), 0], [1.001, 0], [0, -1.001]])
def test_invalid_coordinates_before_sampling(bad):
    def no_run(*_):
        pytest.fail("invalid input reached runner")
    with pytest.raises((TypeError, ValueError)):
        adaptive_assign(bad, CENTERS, no_run)
    with pytest.raises((TypeError, ValueError)):
        adaptive_assign(POINT, [CENTERS[0], bad], no_run)
    with pytest.raises((TypeError, ValueError)):
        build_assignment_program(POINT, bad)


@pytest.mark.parametrize("centers", [None, [], "00", [0, 1]])
def test_invalid_candidate_collection(centers):
    with pytest.raises((TypeError, ValueError)):
        adaptive_assign(POINT, centers, lambda *_: pytest.fail("runner"))


@pytest.mark.parametrize("kwargs", [{"batch_shots": 0}, {"batch_shots": -1},
    {"batch_shots": True}, {"batch_shots": 1.0}, {"max_shots": -1},
    {"max_shots": False}, {"max_shots": 64.0}, {"alpha": 0}, {"alpha": 1},
    {"alpha": True}, {"alpha": float("nan")}, {"alpha": float("inf")}, {"alpha": "0.05"}])
def test_invalid_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        adaptive_assign(POINT, CENTERS, lambda *_: pytest.fail("runner"), **kwargs)


def test_runner_validation_and_failure_propagation():
    with pytest.raises(TypeError):
        adaptive_assign(POINT, CENTERS, None)
    def broken(*_):
        raise RuntimeError("external failure")
    with pytest.raises(RuntimeError, match="external failure"):
        adaptive_assign(POINT, CENTERS, broken)


def test_numpy_input_and_candidate_generator():
    result = adaptive_assign(np.array(POINT), (np.array(c) for c in CENTERS),
        lambda *_: pytest.fail("runner"), max_shots=np.int64(0), batch_shots=np.int64(64))
    assert result.means == (None,) * 3


def _state(program):
    machine = CPUQVM()
    machine.run(program, 1)
    return np.array(machine.result().get_state_vector())


def test_fredkin_decomposition_full_matrix_against_native_gate():
    actual, reference = [], []
    for basis in range(8):
        a, b = QProg(3), QProg(3)
        for qubit in range(3):
            if basis & (1 << qubit):
                a << X(qubit)
                b << X(qubit)
        _append_fredkin(a, 2, 0, 1)
        b << SWAP(0, 1).control(2)
        actual.append(_state(a))
        reference.append(_state(b))
    # All amplitudes, not just probabilities: catches relative-phase errors.
    np.testing.assert_allclose(np.array(actual), np.array(reference), atol=2e-14, rtol=0)


@pytest.mark.parametrize("point,center", [((0, -1), (0, 1)), ((0, 0), (0, 0)),
    ((-1, -1), (1, 1)), ((0.31, -0.72), (-0.47, 0.23)), ((1, 0), (-1, 0))])
def test_assignment_program_matches_legacy_mapping_and_native_swap(point, center):
    program = build_assignment_program(point, center)
    assert program.qubits() == [0, 1, 2]
    assert program.cbits() == [0]
    assert len(program.get_measure_nodes()) == 1
    assert "MEASURE q[2],c[0]" in program.originir()
    assert set(program.count_ops()) <= {"H", "T", "CNOT", "U3"}
    assert "CONTROL" not in program.originir()
    actual = QProg(3)
    for gate in program.gate_operations():
        actual << gate
    reference = QProg(3)
    reference << H(2)
    for qubit, data in enumerate((point, center)):
        reference << U3(qubit, (data[1] + 1) * math.pi / 2,
                        (data[0] + 1) * math.pi / 2, 0)
    reference << SWAP(0, 1).control(2) << H(2)
    np.testing.assert_allclose(_state(actual), _state(reference), atol=2e-14, rtol=0)


def test_explicit_cpuqvm_counts_runner_smoke():
    machine = CPUQVM()
    calls = []
    def run(program, shots):
        machine.run(program, shots)
        counts = machine.result().get_counts()
        calls.append(counts)
        return counts
    result = adaptive_assign(POINT, [POINT, POINT], run, max_shots=128)
    assert result.status == "undecided" and result.total_shots == 128
    assert sum(sum(counts.values()) for counts in calls) == 128


def test_smallest_positive_float_alpha_does_not_overflow_ratio():
    result = adaptive_assign(POINT, CENTERS, lambda p, n: {"0": n},
                             max_shots=192, alpha=float.fromhex("0x0.0000000000001p-1022"))
    assert result.status == "undecided"
    assert result.intervals == ((0, 1),) * 3


def test_undecided_does_not_report_empirical_argmin():
    runner = CountsRunner(centers=CENTERS[:2], fractions=(0.25, 0.375))
    result = adaptive_assign(POINT, CENTERS[:2], runner, max_shots=128)
    assert result.means[0] < result.means[1]
    assert result.winner is None and result.status == "undecided"


def test_production_module_has_no_backend_or_probability_access():
    import ast
    import inspect
    from pyqpanda_alg.QKmeans import adaptive_assignment as module
    tree = ast.parse(inspect.getsource(module))
    forbidden = {"CPUQVM", "get_prob_dict", "get_prob_list", "get_state_vector"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden


def test_touching_intervals_are_not_eliminated():
    # K=2, L=2, n=64 gives radius 1/4 with this alpha.
    runner = CountsRunner(centers=CENTERS[:2], fractions=(0.25, 0.75))
    result = adaptive_assign(POINT, CENTERS[:2], runner, max_shots=128,
                             alpha=8 * math.exp(-8))
    assert result.intervals[0][1] == result.intervals[1][0] == 0.5
    assert result.status == "undecided" and result.active == (0, 1)


@pytest.mark.parametrize("near_one", [False, True])
def test_alpha_conversion_must_remain_inside_float_probability_domain(near_one):
    from fractions import Fraction
    tiny = Fraction(1, 10 ** 400)
    alpha = 1 - tiny if near_one else tiny
    with pytest.raises(ValueError, match="representable"):
        adaptive_assign([0, 0], [[0, 0], [1, 1]],
                        lambda *args: pytest.fail("invalid alpha reached runner"),
                        alpha=alpha, max_shots=0)
