import pytest

from pyqpanda_alg.QSEncode import (
    ConfigurationError,
    QSEncodeInsightError,
    ResultBindingError,
)


@pytest.mark.parametrize("code", ["input_mismatch", "configuration_mismatch"])
def test_result_binding_error_has_stable_code_and_message(code):
    error = ResultBindingError(code=code)
    assert isinstance(error, QSEncodeInsightError)
    assert error.code == code
    assert str(error) == code


def test_base_exception_supports_explicit_message_without_losing_code():
    error = ConfigurationError(code="invalid_basis", message="basis is invalid")
    assert error.code == "invalid_basis"
    assert str(error) == "basis is invalid"
