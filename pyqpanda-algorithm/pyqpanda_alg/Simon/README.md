# Simon hidden-XOR algorithm

This module adds Simon's algorithm as a small, testable PyQPanda3 building block. It separates the quantum circuit from the exact GF(2) post-processing so the hidden-period contract can be validated independently of a simulator.

## Problem contract

For a non-zero `n`-bit secret `s`, Simon's promise is a two-to-one function with

```text
f(x) = f(x XOR s)
```

and no other collisions. Public bit strings in this module are **q0-first**. For example, `"101"` means `q0=1, q1=0, q2=1`; integer basis-state indices use q0 as the least-significant bit.

The supplied oracle is a canonical linear member of Simon's promise family. Choose the first pivot `p` for which `s[p] = 1`, then

```text
f(x)[p] = 0
f(x)[j] = x[j] XOR s[j] * x[p]   for j != p
```

Its kernel is exactly `{0, s}`, so every output has exactly the preimages `x` and `x XOR s`.

## Quantum circuit

`linear_simon_oracle(secret, input_qubits, output_qubits)` implements the reversible map

```text
|x, y> -> |x, y XOR f(x)>
```

using only CNOT gates. The exact CNOT count is

```text
n - 1 + wt(s) - 1
```

which is at most `2n - 3` for `n > 1` (and zero for the valid one-bit secret `1`). `simon_circuit(...)` adds the initial and final Hadamards on the input register.

After the final Hadamards, the input register is supported exactly on strings `y` satisfying

```text
y · s = 0  (mod 2)
```

with uniform ideal probability `1 / 2^(n-1)`.

## Exact classical recovery

`recover_secret_from_equations(...)` performs Gaussian elimination over GF(2). It returns a secret only when the homogeneous system has rank exactly `n-1`, which makes the non-zero nullspace vector unique. Undersampled inputs fail with `ValueError` rather than returning a guess.

Useful pure-Python helpers include:

- `linear_simon_function` — evaluate the canonical promise function.
- `simon_reference_probabilities` — exact ideal measurement distribution.
- `gf2_rank` / `independent_equations` — inspect sampled equation quality.
- `recover_secret_from_probabilities` — recover from a marginal probability list.
- `recover_secret_from_counts` — recover from measured string counts with explicit `q0-first` or `qN-first` key order.

Circuit helpers import `pyqpanda3` lazily; the GF(2), promise, and reference-distribution helpers do not require PyQPanda.

## Example

```python
from pyqpanda_alg import Simon

secret = "10110"
probs = Simon.simon_reference_probabilities(secret)
assert Simon.recover_secret_from_probabilities(probs, len(secret)) == secret

# With PyQPanda3 installed:
recovered, probs = Simon.run_simon(secret)
assert recovered == secret
```

`run_simon` uses the simulator's input-register marginal probability list. Real sampled hardware-style counts may not span rank `n-1` on the first batch; collect more valid outcomes and call `recover_secret_from_counts` again rather than guessing.
