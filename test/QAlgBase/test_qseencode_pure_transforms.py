import numpy as np
import pytest
from scipy.fft import fft

import pyqpanda_alg.QSEncode._transforms as transforms_module
from pyqpanda_alg.QSEncode import InputValidationError, InternalInvariantError
from pyqpanda_alg.QSEncode._error_budget import (
    find_k_star,
    stable_magnitude_order,
    top_k_coefficients,
)
from pyqpanda_alg.QSEncode._transforms import (
    analyze_transform,
    explicit_walsh_oracle,
    normalized_fourier,
    normalized_fwht,
)


ATOL = 1e-12
RTOL = 1e-10


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("kind", ["basis", "random_real", "random_complex"])
def test_fwht_matches_explicit_hadamard_oracle(size, kind):
    rng = np.random.default_rng(20260821 + size)
    if kind == "basis":
        values = np.eye(size, dtype=np.float64)[size // 2]
    elif kind == "random_real":
        values = rng.normal(size=size)
    else:
        values = rng.normal(size=size) + 1j * rng.normal(size=size)
    np.testing.assert_allclose(
        normalized_fwht(values),
        explicit_walsh_oracle(values),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("complex_input", [False, True])
def test_fwht_involution_parseval_input_immutability_and_dtype(complex_input):
    rng = np.random.default_rng(7)
    values = rng.normal(size=32)
    if complex_input:
        values = values + 1j * rng.normal(size=32)
    before = values.copy()
    transformed = normalized_fwht(values)

    np.testing.assert_array_equal(values, before)
    np.testing.assert_allclose(normalized_fwht(transformed), values, atol=ATOL, rtol=RTOL)
    assert np.sum(np.abs(transformed) ** 2) == pytest.approx(
        np.sum(np.abs(values) ** 2), rel=RTOL, abs=ATOL
    )
    assert np.iscomplexobj(transformed) is complex_input


@pytest.mark.parametrize("values", [[], [1.0, 2.0, 3.0]])
def test_fwht_rejects_non_positive_power_of_two_lengths(values):
    with pytest.raises(InputValidationError) as exc_info:
        normalized_fwht(values)
    assert exc_info.value.code == "transform_non_power_of_two"


@pytest.mark.parametrize("values", [[1.0, np.nan], [1.0, np.inf]])
def test_transforms_reject_nonfinite_values(values):
    with pytest.raises(InputValidationError) as exc_info:
        normalized_fwht(values)
    assert exc_info.value.code == "transform_nonfinite"
    with pytest.raises(InputValidationError) as exc_info:
        normalized_fourier(values)
    assert exc_info.value.code == "transform_nonfinite"


def test_fourier_orthonormal_scale_and_scientific_invariants():
    rng = np.random.default_rng(20260821)
    values = rng.normal(size=16) + 1j * rng.normal(size=16)
    old = fft(values)
    new = normalized_fourier(values)

    np.testing.assert_allclose(new * np.sqrt(len(values)), old, atol=ATOL, rtol=RTOL)
    np.testing.assert_array_equal(
        stable_magnitude_order(new), stable_magnitude_order(old)
    )
    assert find_k_star(new, 0.99).k_star == find_k_star(old, 0.99).k_star
    k = find_k_star(new, 0.99).k_star
    np.testing.assert_allclose(
        top_k_coefficients(new, k, normalize=True),
        top_k_coefficients(old, k, normalize=True),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("basis", ["walsh", "fourier"])
def test_transform_diagnostics_report_orthonormal_parseval_contract(basis):
    values = np.sqrt(np.array([0.1, 0.2, 0.3, 0.4]))
    coefficients, diagnostics = analyze_transform(
        values, basis=basis, oracle_check=(basis == "walsh")
    )
    assert len(coefficients) == 4
    assert diagnostics.basis == basis
    assert diagnostics.coefficient_count == 4
    assert diagnostics.normalization == "orthonormal"
    assert diagnostics.parseval_error <= ATOL
    if basis == "walsh":
        assert diagnostics.implementation == "iterative_fwht_v1"
        assert diagnostics.oracle_checked is True
        assert diagnostics.oracle_max_abs_error is not None
        assert diagnostics.oracle_max_abs_error <= ATOL
    else:
        assert diagnostics.implementation == "scipy_fft_ortho_v1"
        assert diagnostics.oracle_checked is False
        assert diagnostics.oracle_max_abs_error is None


def test_transform_rejects_unknown_basis_without_auto_selection():
    with pytest.raises(InputValidationError) as exc_info:
        analyze_transform(np.ones(4), basis="auto")
    assert exc_info.value.code == "invalid_transform_basis"


def test_transform_invariant_tolerances_are_frozen():
    assert transforms_module.TRANSFORM_ATOL == 1e-12
    assert transforms_module.TRANSFORM_RTOL == 1e-10


@pytest.mark.parametrize("basis", ["walsh", "fourier"])
def test_finite_overflow_prone_source_rejects_nonfinite_input_energy(basis):
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(InternalInvariantError) as exc_info:
            analyze_transform(np.full(4, 1e308), basis=basis)
    assert exc_info.value.code == "nonfinite_transform_energy"


@pytest.mark.parametrize("basis", ["walsh", "fourier"])
def test_zero_source_rejects_zero_transform_energy(basis):
    with pytest.raises(InternalInvariantError) as exc_info:
        analyze_transform(np.zeros(4), basis=basis)
    assert exc_info.value.code == "zero_transform_energy"


def test_nonfinite_transformed_coefficients_are_rejected(monkeypatch):
    monkeypatch.setattr(
        transforms_module,
        "normalized_fwht",
        lambda values: np.full(4, np.inf),
    )
    with pytest.raises(InternalInvariantError) as exc_info:
        analyze_transform([1.0, 0.0, 0.0, 0.0], basis="walsh")
    assert exc_info.value.code == "nonfinite_transform_coefficients"


def test_finite_coefficients_with_nonfinite_output_energy_are_rejected(monkeypatch):
    monkeypatch.setattr(
        transforms_module,
        "normalized_fwht",
        lambda values: np.array([1e308, 0.0, 0.0, 0.0]),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(InternalInvariantError) as exc_info:
            analyze_transform([1.0, 0.0, 0.0, 0.0], basis="walsh")
    assert exc_info.value.code == "nonfinite_transform_energy"


def test_zero_transformed_energy_is_rejected(monkeypatch):
    monkeypatch.setattr(
        transforms_module,
        "normalized_fwht",
        lambda values: np.zeros(4),
    )
    with pytest.raises(InternalInvariantError) as exc_info:
        analyze_transform([1.0, 0.0, 0.0, 0.0], basis="walsh")
    assert exc_info.value.code == "zero_transform_energy"


def test_parseval_guard_rejects_controlled_energy_violation(monkeypatch):
    original_fwht = transforms_module.normalized_fwht
    monkeypatch.setattr(
        transforms_module,
        "normalized_fwht",
        lambda values: original_fwht(values) * 2.0,
    )
    with pytest.raises(InternalInvariantError) as exc_info:
        analyze_transform([1.0, 0.0, 0.0, 0.0], basis="walsh")
    assert exc_info.value.code == "parseval_invariant_failed"
