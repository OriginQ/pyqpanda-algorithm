# Licensed under the Apache License, Version 2.0.
"""Minimal Pauli-string algebra used by the counterdiabatic construction."""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np

PauliString = Tuple[str, ...]
PauliOp = Dict[PauliString, complex]

_MUL = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("Y", "I"): (1, "Y"), ("Z", "I"): (1, "Z"),
    ("X", "X"): (1, "I"), ("Y", "Y"): (1, "I"), ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y"),
}


def identity(n: int) -> PauliString:
    """Return the n-qubit identity Pauli string."""
    return tuple("I" for _ in range(n))


def single(n: int, q: int, p: str) -> PauliString:
    """Return an n-qubit string with Pauli ``p`` acting on qubit ``q``."""
    s = ["I"] * n
    s[q] = p
    return tuple(s)


def pair(n: int, q0: int, p0: str, q1: int, p1: str) -> PauliString:
    """Return an n-qubit string with two requested non-identity Paulis."""
    s = ["I"] * n
    s[q0] = p0
    s[q1] = p1
    return tuple(s)




def pauli_string(n: int, operators: dict[int, str]) -> PauliString:
    """Construct an arbitrary Pauli string from {qubit: letter}."""
    s = ["I"] * int(n)
    for q, op in operators.items():
        q = int(q); op = str(op).upper()
        if q < 0 or q >= n:
            raise ValueError(f"qubit {q} outside [0,{n})")
        if op not in {"I","X","Y","Z"}:
            raise ValueError(f"invalid Pauli {op}")
        s[q] = op
    return tuple(s)


def z_string(n: int, qubits) -> PauliString:
    """Construct Z on every listed qubit and identity elsewhere."""
    return pauli_string(n, {int(q): "Z" for q in qubits})


def clean(op: PauliOp, tol: float = 1e-13) -> PauliOp:
    """Drop numerically negligible Pauli coefficients."""
    return {p: complex(c) for p, c in op.items() if abs(c) > tol}


def add(a: PauliOp, b: PauliOp, scale_b: complex = 1.0) -> PauliOp:
    """Return ``a + scale_b * b`` in sparse Pauli representation."""
    out: defaultdict[PauliString, complex] = defaultdict(complex)
    for p, c in a.items(): out[p] += c
    for p, c in b.items(): out[p] += scale_b * c
    return clean(dict(out))


def scale(a: PauliOp, s: complex) -> PauliOp:
    """Scale all Pauli coefficients by a scalar."""
    return clean({p: s * c for p, c in a.items()})


def multiply_strings(a: PauliString, b: PauliString) -> tuple[complex, PauliString]:
    """Multiply two Pauli strings and return phase and resulting string."""
    if len(a) != len(b): raise ValueError("Pauli strings must have equal length")
    phase = 1.0 + 0.0j
    out = []
    for x, y in zip(a, b):
        ph, p = _MUL[(x, y)]
        phase *= ph
        out.append(p)
    return phase, tuple(out)


def multiply(a: PauliOp, b: PauliOp) -> PauliOp:
    """Multiply two sparse Pauli operators exactly."""
    out: defaultdict[PauliString, complex] = defaultdict(complex)
    for pa, ca in a.items():
        for pb, cb in b.items():
            ph, p = multiply_strings(pa, pb)
            out[p] += ca * cb * ph
    return clean(dict(out))


def commutator(a: PauliOp, b: PauliOp) -> PauliOp:
    """Return the operator commutator ``[a,b]``."""
    return add(multiply(a, b), multiply(b, a), scale_b=-1.0)


def hs_norm_sq(op: PauliOp, *, include_dimension: bool = False) -> float:
    """Return the Hilbert-Schmidt norm squared of a sparse Pauli operator."""
    val = float(sum(abs(c) ** 2 for c in op.values()))
    if include_dimension and op:
        val *= 2 ** len(next(iter(op)))
    return val


def matrix(op: PauliOp) -> np.ndarray:
    """Materialize a sparse Pauli operator as a dense reference matrix."""
    if not op: return np.zeros((1, 1), dtype=complex)
    n = len(next(iter(op)))
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mats = {"I": I, "X": X, "Y": Y, "Z": Z}
    out = np.zeros((1 << n, 1 << n), dtype=complex)
    for p, c in op.items():
        m = np.array([[1.0 + 0j]])
        for q in reversed(range(n)):
            m = np.kron(m, mats[p[q]])
        out += c * m
    return out
