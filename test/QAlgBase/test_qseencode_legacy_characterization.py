"""Characterization tests for the frozen QSpare_Code legacy API.

Expected constants were extracted read-only from
upstream/develop@5f973efccb84bc193157d1ccebe32137e307293b on 2026-08-21.
These tests intentionally preserve observable quirks; they are not a preferred
contract for the new QSEncodeInsight facade.
"""

import inspect

import numpy as np
import pytest
from pyqpanda3.core import QCircuit, QProg

from pyqpanda_alg.QSEncode import QSpare_Code


def test_legacy_import_and_constructor_signature_are_unchanged():
    assert QSpare_Code.__name__ == "QSpare_Code"
    assert str(inspect.signature(QSpare_Code)) == (
        "(prob_list=None, cut_length=None, mode='walsh')"
    )


def test_constructor_positional_keyword_defaults_and_observable_attributes():
    defaulted = QSpare_Code([0.25, 0.75])
    positional = QSpare_Code([0.25, 0.75], 1, "fourier")
    keyword = QSpare_Code(prob_list=[0.25, 0.75], cut_length=1, mode="fourier")

    assert defaulted.cut == 2 * defaulted.qubits_num == 2
    assert defaulted.mode == "walsh"
    assert positional.cut == keyword.cut == 1
    assert positional.mode == keyword.mode == "fourier"
    np.testing.assert_allclose(defaulted.prob, [0.25, 0.75])
    np.testing.assert_allclose(defaulted.amp, [0.5, np.sqrt(0.75)])


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.25, 0.75],
        np.array([0.25, 0.75], dtype=np.float64),
        [np.float64(0.25), np.float64(0.75)],
    ],
)
def test_constructor_accepts_current_float_containers(probabilities):
    assert isinstance(QSpare_Code(probabilities), QSpare_Code)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (None, "prob list should be supported"),
        ((0.25, 0.75), "prob_list should be np.ndarray or list"),
        ([], "at least one number in the prob_list "),
        ([0, 1], "elements of prob_list should be float type"),
        (
            np.array([np.float32(0.25), np.float32(0.75)]),
            "elements of prob_list should be float type",
        ),
        ([-0.1, 1.1], "prob must > 0"),
    ],
)
def test_constructor_rejects_inputs_with_exact_legacy_errors(probabilities, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        QSpare_Code(probabilities)


def test_sum_tolerance_raises_warning_object_instead_of_emitting_warning():
    accepted = QSpare_Code([0.5, 0.5009])
    np.testing.assert_allclose(accepted.prob.sum(), 1.0)

    with pytest.raises(Warning, match="sum of prob list should be 1"):
        QSpare_Code([0.5, 0.5011])


def test_nan_currently_passes_constructor_and_remains_nan():
    result = QSpare_Code([float("nan"), 0.0])
    assert np.isnan(result.prob).all()
    assert np.isnan(result.amp).all()


def test_power_of_two_padding_qubits_and_default_cut_are_preserved():
    result = QSpare_Code([0.2, 0.3, 0.5])
    assert result.qubits_num == 2
    assert len(result.prob) == 4
    assert result.cut == 4
    np.testing.assert_allclose(result.prob, [0.2, 0.3, 0.5, 0.0])
    np.testing.assert_allclose(result.amp, np.sqrt([0.2, 0.3, 0.5, 0.0]))


def test_single_value_and_unvalidated_constructor_fields_are_preserved():
    single = QSpare_Code([1.0])
    assert single.qubits_num == 0
    assert single.cut == 0

    assert QSpare_Code([0.5, 0.5], cut_length=0).cut == 0
    assert QSpare_Code([0.5, 0.5], cut_length=1.0).cut == 1.0
    assert QSpare_Code([0.5, 0.5], cut_length=99).cut == 99
    assert QSpare_Code([0.5, 0.5], mode="invalid").mode == "invalid"


def test_select_top_n_return_type_ties_and_boundary_validation():
    engine = QSpare_Code([0.25, 0.75])
    values = np.array([1 + 1j, -3j, 2 + 0j])

    selected = engine.select_top_n_complex_numbers(values, 2)
    assert isinstance(selected, np.ndarray)
    assert selected.dtype == np.complex128
    np.testing.assert_array_equal(selected, [0j, -3j, 2 + 0j])
    np.testing.assert_array_equal(
        engine.select_top_n_complex_numbers(values, 99), values
    )

    for invalid in (0, -1, 1.0, np.int64(1)):
        with pytest.raises(ValueError, match="n must > 0 and with class int"):
            engine.select_top_n_complex_numbers(values, invalid)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "walsh",
            [
                0.9718097255278189,
                -0.1078594020058149,
                -0.208368364011023,
                -0.02312642747730509,
            ],
        ),
        (
            "fourier",
            [
                1.943619451055638 + 0j,
                -0.23149479148832813 + 0.18524193653371795j,
                -0.2157188040116298 + 0j,
                -0.23149479148832813 - 0.18524193653371795j,
            ],
        ),
    ],
)
def test_transform_small_n_convention_ordering_and_complex_behavior(mode, expected):
    engine = QSpare_Code([0.1, 0.2, 0.3, 0.4], mode=mode)
    transformed = engine.Transform(engine.amp)
    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(
        np.asarray(transformed, dtype=np.complex128), expected, atol=1e-14
    )
    if mode == "walsh":
        assert transformed.dtype == object
    else:
        assert transformed.dtype == np.complex128


def test_invalid_transform_mode_is_rejected_when_transform_is_called():
    engine = QSpare_Code([0.25, 0.75], mode="invalid")
    with pytest.raises(ValueError, match="mode only support walsh or fourier"):
        engine.Transform(engine.amp)


@pytest.mark.parametrize("mode", ["walsh", "fourier"])
def test_quantum_circuit_and_minimal_quantum_result_contract(mode):
    engine = QSpare_Code([0.25, 0.75], cut_length=1, mode=mode)
    qubits = QProg(engine.qubits_num).qubits()
    circuit = engine.quantum_cir(qubits)
    result = engine.Quantum_Res()

    assert isinstance(circuit, QCircuit)
    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-12)
