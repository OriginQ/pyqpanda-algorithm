"""Check signed QUBO widths by decoding CPU arithmetic and phase oracles."""

from itertools import product

import numpy as np
import pytest
import sympy as sp
from pyqpanda3.core import CPUQVM, H, QProg, X
from pyqpanda_alg.QUBO.QUBO import QuadraticBinary, QUBO_GAS_origin


def _linear(coefficient: int | float) -> dict:
    """Build an explicit one-variable input, including the zero polynomial."""
    return {"quadratic": [[0]], "linear": [coefficient], "constant": 0}


def _value(data: dict, bits: tuple[int, ...]) -> int | float:
    """Compute the original polynomial independently of QuadraticBinary."""
    return (
        data["constant"]
        + sum(a * x for a, x in zip(data["linear"], bits, strict=True))
        + sum(
            data["quadratic"][i][j] * bits[i] * bits[j]
            for i in range(len(bits))
            for j in range(len(bits))
        )
    )


def _assert_encoded_values(data: dict) -> None:
    """Verify every basis input has the correct deterministic signed result."""
    problem = QuadraticBinary(data)
    key_width, value_width = map(int, problem.query_qnumber())
    assert value_width >= 1
    for bits in product((0, 1), repeat=key_width):
        value = int(_value(data, bits))
        assert -(1 << (value_width - 1)) <= value < (1 << (value_width - 1))
        program = QProg(key_width + value_width)
        for qubit, bit in enumerate(bits):
            if bit:
                program << X(qubit)
        program << problem.cir(
            list(range(key_width)), list(range(key_width, key_width + value_width))
        )
        machine = CPUQVM()
        machine.run(program, shots=1)
        state = np.asarray(machine.result().get_state_vector())
        key = sum(bit << i for i, bit in enumerate(bits))
        encoded = value % (1 << value_width)
        target = key | (encoded << key_width)
        assert abs(state[target]) ** 2 == pytest.approx(1.0, abs=1e-12)
        decoded = (
            encoded - (1 << value_width) if encoded >> (value_width - 1) else encoded
        )
        assert decoded == value


@pytest.mark.parametrize(
    ("coefficient", "expected_width"),
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
        (4, 4),
        (7, 4),
        (-1, 1),
        (-2, 2),
        (-3, 3),
        (-4, 3),
    ],
)
def test_integer_boundaries_encode_with_a_sign_bit(
    coefficient: int, expected_width: int
) -> None:
    """Exercise zero, both signs, powers of two and their adjacent values."""
    data = _linear(coefficient)
    assert QuadraticBinary(data).query_qnumber() == [1, expected_width]
    _assert_encoded_values(data)


@pytest.mark.parametrize(
    ("coefficient", "expected_width"),
    [
        (0.25, 2),
        (-0.25, 1),
        (1.25, 3),
        (-2.25, 3),
        (2**53 - 1, 54),
        (2**53, 55),
        (2**53 + 1, 55),
        (-(2**53), 54),
        (-(2**53) - 1, 55),
    ],
)
def test_fractional_envelopes_and_large_integer_widths(
    coefficient: int | float, expected_width: int
) -> None:
    """Check allocation only, without fractional simulation or 55-qubit claims."""
    assert QuadraticBinary(_linear(coefficient)).query_qnumber() == [1, expected_width]


@pytest.mark.parametrize(
    "data",
    [
        {"quadratic": [[0, 3], [0, 0]], "linear": [0, 0], "constant": 0},
        {"quadratic": [[1, 2], [-1, 0]], "linear": [-3, 4], "constant": 1},
        {"quadratic": [[0]], "linear": [0], "constant": 3},
    ],
)
def test_quadratic_diagonal_cross_terms_and_constant_encode(data: dict) -> None:
    """Cover the full supported matrix form, not only linear expressions."""
    _assert_encoded_values(data)


@pytest.mark.parametrize(
    "data",
    [
        _linear(0),
        _linear(4),
        {"quadratic": [[0, 2], [1, 0]], "linear": [3, -2], "constant": -1},
    ],
)
@pytest.mark.parametrize("threshold", [0, 1, 5])
def test_gas_oracle_marks_only_values_below_threshold(
    data: dict, threshold: int
) -> None:
    """Compare coherent signs and ancilla uncomputation to the classical predicate."""
    problem = QUBO_GAS_origin(data)
    key_width = len(data["linear"])
    value_width = int(problem._n_value_function(threshold))
    assert value_width >= 1
    program = QProg(key_width + value_width)
    for qubit in range(key_width):
        program << H(qubit)
    program << problem._flip_oracle_function(
        list(range(key_width + value_width)), threshold
    )
    machine = CPUQVM()
    machine.run(program, shots=1)
    actual = np.asarray(machine.result().get_state_vector())
    expected = np.zeros(1 << (key_width + value_width), dtype=complex)
    for key in range(1 << key_width):
        bits = tuple((key >> i) & 1 for i in range(key_width))
        expected[key] = (-1 if _value(data, bits) < threshold else 1) / np.sqrt(
            1 << key_width
        )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_original_fractional_example_retains_its_width() -> None:
    """Preserve the documented example without mirroring the allocation formula."""
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    expression = -1.2 * x0 * x1 + 0.9 * x1 * x2 + 1.3 * x0 - x1 - 0.5 * x2
    assert QuadraticBinary(expression).query_qnumber() == [3, 3]


@pytest.mark.parametrize(
    "quadratic, linear, constant, expected_width",
    [
        (None, np.array([-(2**63)], dtype=np.int64), 0, 64),
        (None, np.array([2**62] * 3, dtype=np.int64), 0, 65),
        (None, np.array([2**61] * 8, dtype=np.int64), 0, 66),
        (None, np.array([2**63] * 2, dtype=np.uint64), 0, 66),
        (None, np.array([2**30] * 4, dtype=np.int32), 0, 34),
        (None, [0], np.int64(2**63 - 1), 64),
        (None, [0], np.uint64(2**64 - 1), 65),
        (np.diag(np.array([2**62] * 3, dtype=np.int64)), None, 0, 65),
    ],
    ids=[
        "int64-minimum-absolute-value",
        "int64-positive-wraparound",
        "int64-wraparound-to-zero",
        "uint64-wraparound",
        "int32-wraparound",
        "int64-constant-exact-boundary",
        "uint64-constant-exact-boundary",
        "quadratic-int64-wraparound",
    ],
)
def test_numpy_integer_bounds_do_not_overflow(
    quadratic: np.ndarray | None,
    linear: np.ndarray | list[int] | None,
    constant: int | np.integer,
    expected_width: int,
) -> None:
    """Check allocation only: these widths must never trigger a huge simulation."""
    problem = QuadraticBinary(
        {"quadratic": quadratic, "linear": linear, "constant": constant}
    )
    with np.errstate(over="raise"):
        assert problem.query_qnumber()[1] == expected_width
