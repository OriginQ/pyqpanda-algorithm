import numpy as np

from pyqpanda_alg.QSEncode._validation import canonical_probability_sha256


def test_original_input_hash_known_vector():
    assert canonical_probability_sha256(
        np.array([1.0, 3.0, 0.0]),
        domain=b"qseencode-original-input-v1\0",
    ) == "e729a6fb50de27cc3f1868f7a7ccd7c322e735dcb64680ca127852e61e11c797"


def test_effective_probability_hash_known_vector():
    assert canonical_probability_sha256(
        np.array([0.25, 0.75, 0.0, 0.0]),
        domain=b"qseencode-effective-probability-v1\0",
    ) == "2b120f880bd1947febe7ca47d2023eb587a39478584caadae547e43ecbacbf44"


def test_hash_is_independent_of_native_dtype_and_memory_layout():
    contiguous = np.array([0.2, 0.3, 0.5], dtype="<f8")
    backing = np.array([0.2, 9.0, 0.3, 9.0, 0.5], dtype=">f8")
    strided = backing[::2]
    domain = b"qseencode-original-input-v1\0"
    expected = "f483716e621e017e6dcada99cacde0641e7de34b978ae2077bd5228999cd53bd"
    assert canonical_probability_sha256(contiguous, domain=domain) == expected
    assert canonical_probability_sha256(strided, domain=domain) == expected


def test_hash_domains_and_vector_length_are_separated():
    values = np.array([0.25, 0.75])
    original = canonical_probability_sha256(
        values, domain=b"qseencode-original-input-v1\0"
    )
    effective = canonical_probability_sha256(
        values, domain=b"qseencode-effective-probability-v1\0"
    )
    extended = canonical_probability_sha256(
        np.array([0.25, 0.75, 0.0]), domain=b"qseencode-original-input-v1\0"
    )
    assert len({original, effective, extended}) == 3
