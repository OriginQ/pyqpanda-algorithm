"""Superdense-coding protocol helpers."""

from .dense_coding import (
    SuperdenseCoding,
    decode_message,
    encode_message,
    normalize_message,
    prepare_bell_pair,
    superdense_coding_circuit,
)

__all__ = [
    "SuperdenseCoding",
    "decode_message",
    "encode_message",
    "normalize_message",
    "prepare_bell_pair",
    "superdense_coding_circuit",
]
