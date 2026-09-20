# QWalk: discrete-time coined quantum walk on a cycle

`pyqpanda_alg.QWalk` adds a reusable circuit implementation of a discrete-time
coined quantum walk on a cyclic position register of size `N = 2^n`.

## Step operator

The state space is a coin qubit times an `n`-qubit position register. One step
is

```text
U = S (H_coin ⊗ I_position)
```

by default. The conditional shift is

```text
S |0, x> = |0, x-1 mod N>
S |1, x> = |1, x+1 mod N>.
```

The circuit does not allocate shift ancillas. Modular increment/decrement is
synthesized from multi-controlled `X` gates. `q_position[0]` is explicitly the
least-significant position bit, so an integer position has one unambiguous
mapping to the register.

A custom coin gate/circuit can be supplied to `walk_step()` or
`coined_walk_cycle()`. The exact NumPy reference accepts a 2x2 unitary
`coin_matrix` for independent small-instance verification.

## Quick start

```python
from pyqpanda_alg.QWalk import CoinedQuantumWalkCycle

walk = CoinedQuantumWalkCycle(
    num_position_qubits=3,
    steps=4,
    initial_position=0,
    initial_coin=0,
)

expected = walk.reference()   # NumPy-only independent evolution
observed = walk.run_exact()   # exact PyQPanda3 StateVector evolution
```

For one Hadamard step from `|coin=0, position=0>` on an 8-cycle, the position
probability is exactly `1/2` at position 7 and `1/2` at position 1. After two
steps it is `1/4` at 6, `1/2` at 0, and `1/4` at 2, exposing the interference
that distinguishes the walk from a classical random walk.

## Lower-level composition

The module also exports:

- `controlled_increment_cycle` / `controlled_decrement_cycle`: reversible
  controlled modular arithmetic with no ancilla;
- `conditional_cycle_shift`: the coin-conditioned left/right shift;
- `walk_step`: one customizable coin-and-shift step;
- `coined_walk_cycle`: complete basis-state preparation plus repeated steps;
- `reference_walk_cycle`: independent exact NumPy evolution;
- `principal_displacement_moments`: explicit principal-branch mean/variance for
  wrapped position distributions.

## Complexity and scope

For `n` position qubits, each shift uses `2n` controlled `X` operations plus two
coin flips for selecting the coin-0 decrement branch. A `t`-step walk therefore
uses `O(t n)` logical controlled operations. A multi-controlled `X` may itself
be decomposed by the backend, so hardware-native gate cost depends on the target
backend and transpilation strategy.

This contribution implements the canonical one-dimensional coined walk on a
power-of-two cycle. It does not claim arbitrary graph compilation, decoherence
models, or asymptotic quantum speedup for a particular application.
