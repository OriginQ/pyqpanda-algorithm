import numpy as np
import pytest

from pyqpanda_alg.QSEncode import PreparationMethod
from pyqpanda_alg.QSEncode._preparation import (
    _normalize_backend_output_register,
    adapt_preparation_input,
    build_preparation,
)


@pytest.mark.parametrize(
    "method",
    [
        PreparationMethod.AMPLITUDE_ENCODE,
        PreparationMethod.SPARSE_ISOMETRY,
        PreparationMethod.DS_QUANTUM_STATE_PREPARATION,
    ],
)
def test_all_three_adapters_construct_program_and_metadata(method):
    selected = np.array([1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2)])
    prepared_input = adapt_preparation_input(selected)

    build = build_preparation(method, prepared_input)

    assert build.status == "success"
    assert build.program is not None
    assert build.circuit is not None
    assert len(build.output_qubits) == 2
    assert len(set(build.output_qubits)) == 2
    assert set(build.output_qubits).isdisjoint(build.ancillas)
    assert build.required_qubits == len(build.output_qubits) + len(build.ancillas)
    assert build.logical_output_qubits == 2


def test_method_specific_qubit_and_representation_contracts():
    selected = adapt_preparation_input([1.0, 0.0, 0.0, 0.0])
    amplitude = build_preparation(PreparationMethod.AMPLITUDE_ENCODE, selected)
    sparse = build_preparation(PreparationMethod.SPARSE_ISOMETRY, selected)
    ds = build_preparation(PreparationMethod.DS_QUANTUM_STATE_PREPARATION, selected)

    assert (amplitude.required_qubits, amplitude.ancillas, amplitude.input_representation) == (2, (), "dense_list")
    assert (sparse.required_qubits, sparse.ancillas, sparse.input_representation) == (2, (), "sparse_binary_map")
    assert ds.required_qubits == 4
    assert len(ds.ancillas) == 2
    assert ds.input_representation == "sparse_binary_map"


def test_complex_amplitude_backend_duplicate_output_metadata_is_normalized_and_recorded():
    selected = adapt_preparation_input([1 / np.sqrt(2), 0, 1j / np.sqrt(2), 0])
    build = build_preparation(PreparationMethod.AMPLITUDE_ENCODE, selected)

    assert build.output_qubits == (0, 1)
    assert build.diagnostics["backend_reported_output_qubits"] == (0, 1, 0, 1)
    assert build.diagnostics["output_register_normalization"] == "deduplicated_exact_repetition"


def test_real_amplitude_duplicate_output_metadata_is_not_normalized():
    output, normalization = _normalize_backend_output_register(
        PreparationMethod.AMPLITUDE_ENCODE,
        (0, 1),
        (0, 1, 0, 1),
        2,
        is_complex=False,
    )

    assert output == (0, 1, 0, 1)
    assert normalization is None
