# Licensed under the Apache License, Version 2.0 (the "License");

"""Quantum-counting public API."""

from .QuantumCounting import QuantumCounting, QuantumCountingResult

__all__ = ["QuantumCounting", "QuantumCountingResult", "PhaseEstimate",
           "decode_phase_distribution", "phase_to_probability"]

from .decoding import PhaseEstimate, decode_phase_distribution, phase_to_probability
