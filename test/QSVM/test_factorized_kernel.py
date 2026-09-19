"""独立小系统参考与测量计数接口约定测试。

CPUQVM的概率只在这里作为测试参考。脚本生成的计数用于检查协议和记账，
不是器件精度证据。抽样测试实际执行CPUQVM测量；设备及Port接口另行验收。
"""

from collections import Counter

import numpy as np
import pytest
from pyqpanda3.core import CNOT, CPUQVM, H, QCircuit, QProg, U1

from pyqpanda_alg.QSVM import FactorizedQuantumKernel, QuantumKernel_vqnet
from pyqpanda_alg.QSVM.quantum_kernel_svm import _build_circuit


def feature(values):
    """独立构造正向特征映射，不复用待测的重叠线路构造器。"""
    circuit = QCircuit()
    for _ in range(2):
        for q, value in enumerate(values):
            circuit << H(q) << U1(q, 2 * value)
        if len(values) >= 2:
            circuit << CNOT(0, 1)
            circuit << U1(1, 2 * (np.pi - values[0]) * (np.pi - values[1]))
            circuit << CNOT(0, 1)
    return circuit


def probability(program):
    machine = CPUQVM()
    machine.run(program, 1)
    return machine.result().get_prob_dict().get("0" * program.qubits_num(), 0.0)


def reference(x, y):
    program = QProg(len(x))
    program << feature(x) << feature(y).dagger()
    return probability(program)


def without_measurements(program):
    bare = QProg(program.qubits_num())
    for gate in program.gate_operations():
        bare << gate
    return bare


class RecordingCounts:
    """脚本驱动的单元测试替身，绝不当作实际测量数据。"""

    def __init__(self, zero_counts=None):
        self.calls = []
        self.zero_counts = iter(zero_counts) if zero_counts is not None else None

    def __call__(self, program, shots):
        self.calls.append((program, shots))
        width = program.qubits_num()
        assert len(program.get_measure_nodes()) == width
        assert program.cbits_num() == width
        zeros = next(self.zero_counts) if self.zero_counts is not None else shots // 2
        return {"0" * width: zeros, "1" * width: shots - zeros}


class SamplingCounts(RecordingCounts):
    """执行真实测量，不向evaluate传入精确概率。"""

    def __call__(self, program, shots):
        self.calls.append((program, shots))
        machine = CPUQVM()
        machine.run(program, shots)
        return machine.result().get_counts()


def forbidden_runner(*args):
    pytest.fail("runner must not be called")


def test_public_export_preserves_legacy():
    from pyqpanda_alg.QSVM.factorized_kernel import FactorizedQuantumKernel as direct
    assert FactorizedQuantumKernel is direct
    assert QuantumKernel_vqnet.__module__.endswith("quantum_kernel_svm")


@pytest.mark.parametrize("dimension", [2, 3, 6])
@pytest.mark.parametrize("seed", [2, 19, 31])
def test_components_match_independent_and_legacy_oracles(dimension, seed):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-1.2, 2.3, size=(2, dimension))
    runner = RecordingCounts()
    kernel = FactorizedQuantumKernel(runner, shots=100)
    kernel.evaluate(x, y)
    assert [p.qubits_num() for p, _ in runner.calls] == [2] + [1] * (dimension - 2)
    components = [(x[:2], y[:2])] + [(x[j:j + 1], y[j:j + 1]) for j in range(2, dimension)]
    component_probabilities = []
    for (program, _), (cx, cy) in zip(runner.calls, components):
        actual = probability(without_measurements(program))
        assert actual == pytest.approx(reference(cx, cy), abs=2e-12)
        component_probabilities.append(actual)
    full = QProg(dimension)
    full << _build_circuit(full.qubits(), dimension, x, y)
    assert np.prod(component_probabilities) == pytest.approx(reference(x, y), abs=2e-12)
    assert probability(full) == pytest.approx(reference(x, y), abs=2e-12)


def test_nontrivial_gate_sequence_and_measurement_layout():
    x, y = [0.23, -0.61, 0.47], [1.19, 0.32, -0.83]
    runner = RecordingCounts()
    FactorizedQuantumKernel(runner).evaluate(x, y)
    core, single = [p for p, _ in runner.calls]
    expected = QProg(2)
    expected << _build_circuit(expected.qubits(), 2, x[:2], y[:2])
    assert without_measurements(core).originir() == expected.originir()
    assert [g.name() for g in single.gate_operations()] == [
        "U2", "U1", "U2", "U1", "U1", "U2", "U1", "U2"
    ]
    assert single.count_ops() == {"U1": 4, "U2": 4}
    assert core.count_ops() == {"CNOT": 6, "U1": 12, "U2": 8}
    for program in [core, single]:
        p = probability(without_measurements(program))
        assert 0.01 < p < 0.99
        for q in range(program.qubits_num()):
            assert f"MEASURE q[{q}],c[{q}]" in program.originir()


@pytest.mark.parametrize("dimension", [2, 3, 6])
def test_actual_measurements_rectangular_and_swap(dimension):
    rng = np.random.default_rng(20 + dimension)
    x = rng.uniform(-0.7, 1.4, size=(2, dimension))
    y = rng.uniform(-0.7, 1.4, size=(3, dimension))
    runner = SamplingCounts()
    kernel = FactorizedQuantumKernel(runner, shots=16384, batch_size=2)
    xy = kernel.evaluate(x, y)
    yx = kernel.evaluate(y, x)
    expected = np.array([[reference(a, b) for b in y] for a in x])
    np.testing.assert_allclose(xy, expected, atol=0.035, rtol=0)
    np.testing.assert_allclose(yx.T, expected, atol=0.035, rtol=0)
    assert xy.shape == (2, 3)
    assert all(p.qubits_num() <= 2 and shots == 16384 for p, shots in runner.calls)


def test_gram_symmetry_diagonal_and_non_diagonal_duplicates():
    runner = RecordingCounts()
    x = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
    matrix = FactorizedQuantumKernel(runner, shots=8).evaluate(x)
    np.testing.assert_array_equal(matrix, [[1, 0.25, 1], [0.25, 1, 0.25], [1, 0.25, 1]])
    np.testing.assert_array_equal(matrix, matrix.T)


def test_equal_explicit_y_uses_symmetric_path():
    x = np.array([[0.1, 0.2], [0.7, 0.8]])
    runner = RecordingCounts()
    result = FactorizedQuantumKernel(runner, shots=8).evaluate(x, x.copy())
    assert len(runner.calls) == 1
    np.testing.assert_array_equal(result, [[1, 0.5], [0.5, 1]])


def test_rectangular_duplicates_and_identity_components():
    runner = RecordingCounts()
    kernel = FactorizedQuantumKernel(runner, shots=8)
    x = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.9]]
    y = [[0.1, 0.2, 0.3]]
    np.testing.assert_array_equal(kernel.evaluate(x, y), [[1], [0.5]])
    assert [p.qubits_num() for p, _ in runner.calls] == [1]
    runner.calls.clear()
    np.testing.assert_array_equal(kernel.evaluate([0.1, 0.2, 0.3], [0.7, 0.8, 0.3]), [[0.5]])
    assert [p.qubits_num() for p, _ in runner.calls] == [2]


def test_identity_is_exact_equality_not_allclose():
    runner = RecordingCounts()
    kernel = FactorizedQuantumKernel(runner, shots=8)
    assert kernel.evaluate([0.1, 0.2], [0.1 + 1e-12, 0.2])[0, 0] == 0.5
    assert len(runner.calls) == 1
    np.testing.assert_array_equal(FactorizedQuantumKernel(forbidden_runner).evaluate([0.1, 0.2]), [[1]])


def test_cache_exact_keys_within_call_not_across_calls_or_batches():
    x = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [0.1, 0.2, 0.3]]
    y = [[1.1, 1.2, 1.3], [1.1, 1.2, 1.4]]
    runner = RecordingCounts([4] * 5 + [2] * 5)
    kernel = FactorizedQuantumKernel(runner, shots=8, batch_size=1)
    np.testing.assert_array_equal(kernel.evaluate(x, y), np.full((3, 2), 0.25))
    assert len(runner.calls) == 5  # 一个二位核心分量，加上四组有序的单位分量输入对。
    assert Counter(p.qubits_num() for p, _ in runner.calls) == {2: 1, 1: 4}
    np.testing.assert_array_equal(kernel.evaluate(x, y), np.full((3, 2), 0.0625))
    assert len(runner.calls) == 10


def test_cache_does_not_merge_nearby_inputs():
    runner = RecordingCounts()
    FactorizedQuantumKernel(runner).evaluate([[0.1, 0.2], [0.1 + 1e-12, 0.2]], [[0.6, 0.7]])
    assert len(runner.calls) == 2


def test_repeated_values_at_different_positions_have_independent_runs():
    runner = RecordingCounts([2, 1, 3])
    result = FactorizedQuantumKernel(runner, shots=4).evaluate(
        [0.1, 0.2, 0.3, 0.3], [0.7, 0.8, 0.9, 0.9]
    )
    assert len(runner.calls) == 3
    assert result[0, 0] == 0.5 * 0.25 * 0.75


def test_cache_keeps_ordered_component_pairs():
    runner = RecordingCounts()
    FactorizedQuantumKernel(runner).evaluate(
        [[0.1, 0.2], [0.6, 0.7]], [[0.6, 0.7], [0.1, 0.2]]
    )
    assert len(runner.calls) == 2


@pytest.mark.parametrize("batch_size", [1, 2, 4, 100])
def test_batching_preserves_all_pairs_and_last_partial_batch(batch_size):
    runner = RecordingCounts()
    x = np.arange(8).reshape(4, 2) / 10
    y = np.arange(6).reshape(3, 2) / 10 + 2
    matrix = FactorizedQuantumKernel(runner, shots=10, batch_size=batch_size).evaluate(x, y)
    np.testing.assert_array_equal(matrix, np.full((4, 3), 0.5))
    assert len(runner.calls) == 12


def test_no_psd_projection():
    runner = RecordingCounts([4, 4, 0])
    matrix = FactorizedQuantumKernel(runner, shots=4).evaluate([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_equal(matrix, [[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    assert np.linalg.eigvalsh(matrix).min() < 0


@pytest.mark.parametrize("bad", [None, 0, [], "backend"])
def test_runner_required_callable(bad):
    with pytest.raises(TypeError, match="run_counts"):
        FactorizedQuantumKernel(bad)


def test_no_default_runner():
    with pytest.raises(TypeError):
        FactorizedQuantumKernel()


@pytest.mark.parametrize("name", ["shots", "batch_size"])
@pytest.mark.parametrize("bad", [None, 0, -1, 1.5, True, np.bool_(False), "10"])
def test_invalid_positive_integer_parameters(name, bad):
    with pytest.raises(ValueError, match=name):
        FactorizedQuantumKernel(forbidden_runner, **{name: bad})


@pytest.mark.parametrize("bad", [0, 1, -2, 2.5, True, "2"])
def test_invalid_n_qbits(bad):
    with pytest.raises(ValueError, match="n_qbits"):
        FactorizedQuantumKernel(forbidden_runner, n_qbits=bad)


def test_numpy_integer_configuration_and_dimension_inference_is_per_call():
    runner = RecordingCounts()
    kernel = FactorizedQuantumKernel(runner, shots=np.int64(8), batch_size=np.int64(2))
    assert kernel.evaluate([0, 1], [1, 2]).shape == (1, 1)
    assert kernel.evaluate([0, 1, 2], [1, 2, 3]).shape == (1, 1)
    np.testing.assert_array_equal(FactorizedQuantumKernel(forbidden_runner, n_qbits=np.int64(2)).evaluate([0, 1]), [[1]])


@pytest.mark.parametrize("bad", [
    None, [], 2.0, [0.1], [[0.1]], [[]], np.zeros((2, 2, 2)),
    [[0, 1], [2]], [0, np.nan], [0, np.inf], [0, -np.inf],
    [0, 1j], ["0", "1"], [True, False], [None, 1],
])
def test_invalid_x_rejected_before_running(bad):
    with pytest.raises(ValueError, match="x_vec"):
        FactorizedQuantumKernel(forbidden_runner).evaluate(bad)


@pytest.mark.parametrize("bad", [[], 2.0, [0.1], [[0, 1], [2]], [0, np.nan], [1j, 0]])
def test_invalid_y_rejected_before_running(bad):
    with pytest.raises(ValueError, match="y_vec"):
        FactorizedQuantumKernel(forbidden_runner).evaluate([0, 1], bad)


def test_feature_dimensions_must_agree():
    with pytest.raises(ValueError, match="dimensions"):
        FactorizedQuantumKernel(forbidden_runner).evaluate([0, 1], [0, 1, 2])
    with pytest.raises(ValueError, match="n_qbits"):
        FactorizedQuantumKernel(forbidden_runner, n_qbits=3).evaluate([0, 1])


@pytest.mark.parametrize("xshape,yshape,expected", [
    ((0, 3), None, (0, 0)), ((0, 3), (2, 3), (0, 2)),
    ((2, 3), (0, 3), (2, 0)), ((0, 2), (0, 2), (0, 0)),
])
def test_explicit_empty_2d_arrays(xshape, yshape, expected):
    x = np.zeros(xshape)
    y = None if yshape is None else np.zeros(yshape)
    result = FactorizedQuantumKernel(forbidden_runner).evaluate(x, y)
    assert result.shape == expected
    assert result.dtype == np.float64


def test_overflowing_angles_rejected_before_running():
    for x in [[1e308, 1e308], [1e308, 0]]:
        with pytest.raises(ValueError, match="angles"):
            FactorizedQuantumKernel(forbidden_runner).evaluate(x, [0, 1])


@pytest.mark.parametrize("counts", [
    None, [], {}, {"00": 0}, {"00": 7}, {"00": 9},
    {"00": -1, "11": 9}, {"00": 8.0}, {"00": True, "11": 7},
    {"00": np.nan}, {0: 8}, {"0": 8}, {"000": 8}, {"02": 8},
    {"0 0": 8}, {"0x0": 8}, {"": 8}, {"00": 8, "bad": 0},
])
def test_invalid_counts_fail_clearly(counts):
    with pytest.raises(ValueError, match="counts"):
        FactorizedQuantumKernel(lambda p, s: counts, shots=8).evaluate([0.1, 0.2], [0.3, 0.4])


def test_zero_event_absent_is_zero_not_epsilon_and_numpy_counts_are_valid():
    result = FactorizedQuantumKernel(lambda p, s: {"11": np.int64(s)}, shots=8).evaluate([0.1, 0.2], [0.3, 0.4])
    np.testing.assert_array_equal(result, [[0]])
    result = FactorizedQuantumKernel(lambda p, s: {"00": np.int64(s)}, shots=8).evaluate([0.1, 0.2], [0.3, 0.4])
    np.testing.assert_array_equal(result, [[1]])


def test_runner_exceptions_propagate_and_failed_calls_do_not_cache():
    calls = []

    def runner(program, shots):
        calls.append(program.qubits_num())
        if len(calls) == 2:
            raise RuntimeError("device unavailable")
        return {"0" * program.qubits_num(): shots}

    kernel = FactorizedQuantumKernel(runner, shots=8)
    with pytest.raises(RuntimeError, match="device unavailable"):
        kernel.evaluate([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    kernel.evaluate([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    assert calls == [2, 1, 2, 1]


def test_sklearn_callable_shape_without_changing_trainer():
    from sklearn.svm import SVC
    x = [[0.1, 0.2], [0.3, 0.4], [0.7, 0.8], [0.9, 1.0]]
    kernel = FactorizedQuantumKernel(SamplingCounts(), shots=4096)
    classifier = SVC(kernel=kernel.evaluate).fit(x, [0, 0, 1, 1])
    assert classifier.predict(x[:2]).shape == (2,)


def test_very_large_batch_size_is_only_a_buffer_limit():
    runner = RecordingCounts()
    result = FactorizedQuantumKernel(runner, batch_size=10**30).evaluate([0.1, 0.2], [0.3, 0.4])
    np.testing.assert_array_equal(result, [[0.5]])
    assert len(runner.calls) == 1


def test_counts_only_path_never_invokes_a_simulator(monkeypatch):
    import pyqpanda3.core as core
    import pyqpanda_alg.QSVM.quantum_kernel_svm as legacy
    import pyqpanda_alg.QSVM.factorized_kernel as factorized

    monkeypatch.setattr(core, "CPUQVM", forbidden_runner)
    monkeypatch.setattr(legacy, "CPUQVM", forbidden_runner)
    monkeypatch.setattr(legacy, "_run_circuit", forbidden_runner)
    monkeypatch.setattr(factorized, "CPUQVM", forbidden_runner, raising=False)
    runner = RecordingCounts([2, 3])
    result = FactorizedQuantumKernel(runner, shots=4).evaluate(
        [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]
    )
    np.testing.assert_array_equal(result, [[0.375]])
    assert len(runner.calls) == 2


def test_inputs_are_snapshotted_before_first_runner_callback():
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    y = np.array([[0.7, 0.8, 0.9]])
    widths = []
    def mutate_inputs(program, shots):
        width = program.qubits_num()
        widths.append(width)
        x[:] = 0
        y[:] = 0
        return {"0" * width: shots // 2, "1" * width: shots // 2}
    kernel = FactorizedQuantumKernel(mutate_inputs, shots=8)
    np.testing.assert_array_equal(kernel.evaluate(x, y), [[0.25], [0.25]])
    assert widths == [2, 1, 2, 1]
