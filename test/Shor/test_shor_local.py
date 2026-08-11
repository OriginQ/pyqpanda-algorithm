"""Seeded local order-finding and factor-recovery tests for the Shor solver.

Covers the bounded attempt state machine end to end on the CPU backend:
a seeded run factors 15 through quantum order finding and records honest
provenance, and the order-reduction rule that makes recovered orders
that are proper multiples of the true order usable is pinned directly.
"""

import random

from pyqpanda_alg.Shor import Shor, ShorConfig
from pyqpanda_alg.Shor.shor import _reduce_order_multiples


def test_shor_factors_fifteen_with_quantum_order_finding():
    solver = Shor(15, rng=random.Random(7), config=ShorConfig(max_attempts=6))
    result = solver.run()
    assert set(result.factors) == {3, 5}
    assert result.used_quantum is True
    assert result.order is not None


def test_recovered_order_multiples_are_halved_before_factor_derivation():
    # sample=256/1024 recovers order 12 for base 2 mod 21 — a proper
    # multiple of the true order 6 — and factor recovery fails both ways
    # with r=12; the halving rule must reduce it to 6 before use.
    assert _reduce_order_multiples(order=12, base=2, modulus=21) == 6
    # An already-exact order is untouched.
    assert _reduce_order_multiples(order=4, base=2, modulus=15) == 4
