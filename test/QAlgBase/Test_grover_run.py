"""Deterministic CPU regressions for adaptive search result retention."""

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, X
from pyqpanda_alg.Grover import GroverAdaptiveSearch, mark_data_reflection


@pytest.mark.parametrize("rotation_change", ["increase", "random"])
@pytest.mark.parametrize(
    "values, initial, samples, expected_keys, expected_value",
    [
        ([2, 3, 4, 5], 2, [0], {0}, 2),
        ([5, 2, 2, 9], 2, [1, 2], {1, 2}, 2),
        ([2, 2, 3, 4], 9, [0, 1], {0, 1}, 2),
        ([5, 5, 2, 2], 5, [0, 1, 2, 3], {2, 3}, 2),
        ([-3, -3, 1, 2], -3, [0, 1], {0, 1}, -3),
        ([1, 4, 5, 6], 1, [1], set(), 1),
        ([2, 4, 5, 6], 9, [0, 1], {0}, 2),
        ([0, 0, 0, 0], 0, [0, 1, 2, 3], {0, 1, 2, 3}, 0),
    ],
    ids=[
        "initial-already-optimal",
        "equal-initial-minima",
        "improve-then-tie",
        "discard-old-minima-after-improvement",
        "negative-initial-minima",
        "no-sampled-minimum",
        "worse-samples-excluded",
        "constant-objective",
    ],
)
def test_search_retains_measured_minima(
    rotation_change: str,
    values: list[int],
    initial: int,
    samples: list[int],
    expected_keys: set[int],
    expected_value: int,
) -> None:
    """Only measured states attaining the incumbent belong in the result.

    Prepare a chosen computational basis state using the public init callback.
    A diagonal threshold oracle and Grover reflection preserve that basis state
    up to phase. Thus CPUQVM's real measurements are deterministic; no simulator
    or result is mocked. Advance preparation only after the value callback sees
    a measurement, so forward and inverse preparations within an iteration agree.
    """
    observed: list[int] = []

    def prepare(qubits: list[int], threshold: float) -> QCircuit:
        key = samples[min(len(observed), len(samples) - 1)]
        circuit = QCircuit()
        for bit in range(2):
            if key & (1 << bit):
                circuit << X(qubits[bit])
        return circuit

    def oracle(qubits: list[int], threshold: float) -> QCircuit:
        marked = [
            format(key, "02b") for key, value in enumerate(values) if value < threshold
        ]
        if not marked:
            return QCircuit()
        return mark_data_reflection(qubits[:2], marked)

    def measured_value(bits: str) -> int:
        key = int(bits, 2)
        observed.append(key)
        return values[key]

    search = GroverAdaptiveSearch(
        init_value=initial,
        n_index=2,
        init_circuit=prepare,
        oracle_circuit=oracle,
    )
    random_state = np.random.get_state()
    try:
        np.random.seed(7)
        solutions, value = search.run(
            continue_times=4,
            n_value_function=lambda threshold: 1,
            value_function=measured_value,
            rotation_change=rotation_change,
        )
    finally:
        np.random.set_state(random_state)

    assert observed[: len(samples)] == samples
    assert value == expected_value
    # Public solutions are in variable order (least significant bit first).
    keys = [sum(bit << index for index, bit in enumerate(bits)) for bits in solutions]
    assert set(keys) == expected_keys
    assert len(keys) == len(set(keys)), "Repeated samples must not duplicate solutions"
    assert all(
        len(bits) == 2 and all(bit in (0, 1) for bit in bits) for bits in solutions
    )
    assert all(key in observed and values[key] == value for key in keys)
