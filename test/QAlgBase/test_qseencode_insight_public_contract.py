import inspect

import pytest

import pyqpanda_alg.QSEncode as public_module
from pyqpanda_alg.QSEncode import (
    ConfigurationError,
    DEFAULT_METHODS,
    InsightResult,
    PreparationMethod,
    QSEncodeInsight,
)


def test_basis_is_a_required_keyword_only_argument():
    signature = inspect.signature(QSEncodeInsight)
    assert tuple(signature.parameters) == (
        "basis",
        "fidelity_target",
        "verification",
        "methods",
        "compiler",
        "input_policy",
    )
    assert signature.parameters["basis"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["basis"].default is inspect.Parameter.empty

    with pytest.raises(TypeError, match="basis"):
        QSEncodeInsight()


@pytest.mark.parametrize("basis", ["walsh", "fourier"])
def test_explicit_valid_basis_constructs_contract_facade(basis):
    engine = QSEncodeInsight(basis=basis)
    assert engine.basis == basis
    assert engine.fidelity_target == 0.99
    assert engine.methods == DEFAULT_METHODS


@pytest.mark.parametrize("basis", [None, "auto", "WALSH", ""])
def test_invalid_basis_is_rejected_without_fallback(basis):
    with pytest.raises(ConfigurationError) as exc_info:
        QSEncodeInsight(basis=basis)
    assert getattr(exc_info.value, "code", None) == "invalid_basis"


def test_default_method_enum_order_and_values_are_stable():
    assert DEFAULT_METHODS == (
        PreparationMethod.AMPLITUDE_ENCODE,
        PreparationMethod.SPARSE_ISOMETRY,
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
    )
    assert tuple(method.value for method in DEFAULT_METHODS) == (
        "amplitude_encode",
        "sparse_isometry",
        "ds_quantum_state_preparation",
    )


def test_phase7_analyze_and_prepare_are_activated_product_apis():
    engine = QSEncodeInsight(basis="walsh")
    result = engine.analyze([0.5, 0.5])
    artifact = engine.prepare([0.5, 0.5], result=result)
    assert result.selection is not None
    assert artifact.program is not None


def test_public_result_type_is_exported_without_constructing_fake_result():
    assert InsightResult.__name__ == "InsightResult"


def test_public_all_keeps_legacy_and_adds_phase1_contract_names():
    required = {
        "QSpare_Code",
        "QSEncodeInsight",
        "InsightResult",
        "PreparationMethod",
        "ResultBindingError",
        "analysis_config_fingerprint",
    }
    assert required <= set(public_module.__all__)
    assert all(hasattr(public_module, name) for name in public_module.__all__)
