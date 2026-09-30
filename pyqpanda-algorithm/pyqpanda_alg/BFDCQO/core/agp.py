# Licensed under the Apache License, Version 2.0.
"""First-order nested-commutator adiabatic gauge potential."""
from __future__ import annotations

from dataclasses import dataclass

from .hamiltonian import interpolate
from .pauli import PauliOp, commutator, hs_norm_sq, scale


@dataclass(frozen=True)
class GaugePotential:
    """First-order adiabatic-gauge-potential data for one schedule point."""

    alpha1: float
    o1_norm_sq: float
    o2_norm_sq: float
    A1: PauliOp


def first_order_gauge_potential(
    hi: PauliOp,
    hf: PauliOp,
    lam: float,
) -> GaugePotential:
    """Construct the first-order nested-commutator gauge potential.

    The variational coefficient follows ``alpha_1 = -Gamma_1 / Gamma_2`` with
    ``O1=[Hi,Hf]`` and ``O2=[Had,O1]``.  Pauli strings are orthogonal under the
    Hilbert-Schmidt inner product, so the common matrix-dimension factor cancels.
    """
    o1 = commutator(hi, hf)
    had = interpolate(hi, hf, lam)
    o2 = commutator(had, o1)
    gamma1 = hs_norm_sq(o1)
    gamma2 = hs_norm_sq(o2)
    alpha = 0.0 if gamma2 <= 1e-18 else -gamma1 / gamma2
    gauge = scale(o1, 1j * alpha)
    for coefficient in gauge.values():
        if abs(coefficient.imag) > 1e-10:
            raise RuntimeError("A1 is unexpectedly non-Hermitian")
    gauge = {pauli: float(coefficient.real) for pauli, coefficient in gauge.items()}
    return GaugePotential(float(alpha), gamma1, gamma2, gauge)
