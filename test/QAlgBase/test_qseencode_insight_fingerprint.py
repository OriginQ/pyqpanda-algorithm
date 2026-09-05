from dataclasses import replace

from pyqpanda_alg.QSEncode import (
    AnalysisConfig,
    CompilerConfig,
    EvidenceScopeStatus,
    InputPolicy,
    PreparationMethod,
    SelectionDecision,
    VerificationLevel,
    analysis_config_fingerprint,
    canonical_analysis_config_json,
)


DEFAULT_WALSH_CANONICAL_JSON = (
    '{"basis":"walsh","compiler":{"basis_gates":["U3","CNOT"],'
    '"initial_mapping":"identity","optimization_level":2,'
    '"physical_capacity_multiplier":2,"pyqpanda_version":"0.3.5",'
    '"resource_aggregation":"median_with_range","technical_repetitions":5,'
    '"topology":"linear"},"fidelity_target":"0x1.fae147ae147aep-1",'
    '"input_policy":{"normalization":"normalize",'
    '"normalization_tolerance":"0x1.19799812dea11p-40",'
    '"padding":"next_power_of_two"},"methods":["amplitude_encode",'
    '"sparse_isometry","ds_quantum_state_preparation"],'
    '"schema_version":"qseencode-insight-v1",'
    '"selection_policy":"frozen_lexicographic_v1",'
    '"verification":"standard"}'
)
DEFAULT_WALSH_FINGERPRINT = (
    "6530A50A213BB2C74FF71D0C9A0D36BEA84F4A3CE1EB0E6A94CDDE00C0B9FD39"
)


def make_config(**changes):
    base = AnalysisConfig(
        basis="walsh",
        fidelity_target=0.99,
        verification=VerificationLevel.STANDARD,
    )
    return replace(base, **changes)


def test_mapping_key_insertion_order_does_not_change_fingerprint():
    first = {"basis": "walsh", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "basis": "walsh"}
    assert analysis_config_fingerprint(first) == analysis_config_fingerprint(second)


def test_default_walsh_config_has_absolute_canonical_json_and_hash_golden():
    config = make_config()
    assert canonical_analysis_config_json(config) == DEFAULT_WALSH_CANONICAL_JSON
    assert analysis_config_fingerprint(config).upper() == DEFAULT_WALSH_FINGERPRINT


def test_each_frozen_selection_configuration_dimension_changes_fingerprint():
    baseline = make_config()
    baseline_hash = analysis_config_fingerprint(baseline)
    variants = (
        replace(
            baseline,
            methods=(
                PreparationMethod.SPARSE_ISOMETRY,
                PreparationMethod.AMPLITUDE_ENCODE,
                PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
            ),
        ),
        replace(baseline, basis="fourier"),
        replace(baseline, fidelity_target=0.98),
        replace(
            baseline,
            compiler=replace(CompilerConfig(), optimization_level=1),
        ),
        replace(
            baseline,
            input_policy=replace(InputPolicy(), padding="reject"),
        ),
        replace(baseline, verification=VerificationLevel.AUDIT),
    )
    assert all(
        analysis_config_fingerprint(variant) != baseline_hash for variant in variants
    )


def test_fingerprint_is_deterministic_and_uses_float_hex_canonicalization():
    config = make_config()
    assert analysis_config_fingerprint(config) == analysis_config_fingerprint(config)
    assert config.canonical_payload()["fidelity_target"] == float(0.99).hex()


def test_public_enum_values_and_compiler_profile_defaults_are_frozen():
    assert tuple(item.value for item in VerificationLevel) == ("standard", "audit")
    assert tuple(item.value for item in EvidenceScopeStatus) == (
        "validated_default",
        "outside_validated_scope",
    )
    assert tuple(item.value for item in SelectionDecision) == (
        "compress",
        "do_not_compress",
    )
    compiler = CompilerConfig()
    assert (
        compiler.pyqpanda_version,
        compiler.topology,
        compiler.physical_capacity_multiplier,
        compiler.initial_mapping,
        compiler.optimization_level,
        compiler.basis_gates,
        compiler.technical_repetitions,
        compiler.resource_aggregation,
    ) == (
        "0.3.5",
        "linear",
        2,
        "identity",
        2,
        ("U3", "CNOT"),
        5,
        "median_with_range",
    )
