import pytest

from pyqpanda3.transpilation import Transpiler

from pyqpanda_alg.QSEncode import CompilerConfig, PreparationMethod
from pyqpanda_alg.QSEncode._capability import assess_capability
from pyqpanda_alg.QSEncode._compiler import (
    FROZEN_COMPILER_PROFILE,
    TECHNICAL_REPETITIONS,
    compile_five_repetitions,
    compose_end_to_end_program,
)
from pyqpanda_alg.QSEncode._preparation import adapt_preparation_input
from pyqpanda_alg.QSEncode._resources import audit_build_resources
from pyqpanda_alg.QSEncode.exceptions import InternalInvariantError


def _compatible_build(method=PreparationMethod.AMPLITUDE_ENCODE):
    prepared = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    assessment = assess_capability(method, prepared)
    assert assessment.report.compatible and assessment.build is not None
    return assessment.build


def test_frozen_compiler_profile_is_exact():
    assert FROZEN_COMPILER_PROFILE == CompilerConfig()
    assert FROZEN_COMPILER_PROFILE.pyqpanda_version == "0.3.5"
    assert FROZEN_COMPILER_PROFILE.topology == "linear"
    assert FROZEN_COMPILER_PROFILE.physical_capacity_multiplier == 2
    assert FROZEN_COMPILER_PROFILE.initial_mapping == "identity"
    assert FROZEN_COMPILER_PROFILE.optimization_level == 2
    assert FROZEN_COMPILER_PROFILE.basis_gates == ("U3", "CNOT")
    assert FROZEN_COMPILER_PROFILE.technical_repetitions == 5
    assert FROZEN_COMPILER_PROFILE.resource_aggregation == "median_with_range"


@pytest.mark.parametrize("method", list(PreparationMethod))
def test_end_to_end_program_separates_required_and_allocated_width(method):
    compilation = compose_end_to_end_program(_compatible_build(method), basis="walsh")

    assert compilation.allocated_qubits == 4
    assert compilation.output_qubits == (2, 3)
    assert compilation.required_qubits == (4 if method.value.startswith("ds_") else 2)
    assert len(compilation.ancillas) == (2 if method.value.startswith("ds_") else 0)
    assert compilation.program is not None


def test_five_real_compilations_record_complete_u3_cnot_resources():
    compilation = compose_end_to_end_program(_compatible_build(), basis="walsh")
    attempts = compile_five_repetitions(compilation)

    assert TECHNICAL_REPETITIONS == 5
    assert len(attempts) == 5
    assert tuple(attempt.attempt_index for attempt in attempts) == (0, 1, 2, 3, 4)
    assert all(attempt.success for attempt in attempts)
    assert all(attempt.compiled_program is not None for attempt in attempts)
    assert all(attempt.compiled_originir for attempt in attempts)
    assert all(len(attempt.originir_sha256) == 64 for attempt in attempts)
    assert all(attempt.compiled_two_qubit_gates == attempt.compiled_cnot_gates for attempt in attempts)
    assert all(attempt.compiled_total_gates >= attempt.compiled_two_qubit_gates >= 0 for attempt in attempts)
    assert all(attempt.compiled_depth >= 0 for attempt in attempts)
    assert len({attempt.compiler_profile_fingerprint for attempt in attempts}) == 1


def test_one_failed_repeat_is_retained_and_no_sixth_compile_is_attempted():
    compilation = compose_end_to_end_program(_compatible_build(), basis="walsh")

    class FlakyTranspiler:
        calls = 0

        def transpile(self, *args, **kwargs):
            type(self).calls += 1
            if type(self).calls == 3:
                raise RuntimeError("synthetic repeat failure")
            return Transpiler().transpile(*args, **kwargs)

    attempts = compile_five_repetitions(
        compilation, transpiler_factory=FlakyTranspiler
    )
    audit = audit_build_resources(
        _compatible_build(),
        basis="walsh",
        attempts=attempts,
    )

    assert FlakyTranspiler.calls == 5
    assert len(attempts) == 5
    assert sum(attempt.success for attempt in attempts) == 4
    assert attempts[2].status == "compile_failure"
    assert attempts[2].exception_type == "RuntimeError"
    assert audit.valid is False
    assert audit.status == "compile_failure"
    assert audit.successful_attempts == 4
    assert audit.failed_attempts == 1
    assert len(audit.compilation_attempts) == 5


def test_compiler_internal_invariant_propagates():
    compilation = compose_end_to_end_program(_compatible_build(), basis="walsh")

    class BrokenInternalTranspiler:
        def transpile(self, *args, **kwargs):
            raise InternalInvariantError(code="synthetic_compiler_invariant")

    with pytest.raises(InternalInvariantError) as exc_info:
        compile_five_repetitions(
            compilation, transpiler_factory=BrokenInternalTranspiler
        )
    assert exc_info.value.code == "synthetic_compiler_invariant"
