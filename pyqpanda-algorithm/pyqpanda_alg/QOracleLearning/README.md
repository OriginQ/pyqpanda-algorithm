# QOracleLearning

`QOracleLearning` adds two textbook oracle-learning algorithms to `pyqpanda-algorithm`:

- **Deutsch-Jozsa**: one quantum oracle query distinguishes a promised constant Boolean function from a balanced one.
- **Bernstein-Vazirani**: one quantum oracle query recovers the complete hidden affine bit string `s` in `f(x)=s·x XOR b`.

## Bit-order contract

All public secret strings are **q0-first**. `"101"` means `q0=1`, `q1=0`, `q2=1`. Truth-table index `x` uses ordinary integer bits, so q0 is the least-significant bit. Execution uses `get_prob_list()` and decodes the winning integer basis index explicitly, avoiding display-string endianness ambiguity.

## Oracle construction

`truth_table_phase_oracle()` accepts an explicit truth table and directly synthesizes the diagonal phase oracle `(-1)^f(x)`. This generic teaching path is worst-case `O(2^n)` gates and is not claimed to be scalable.

`affine_phase_oracle()` uses the structure of Bernstein-Vazirani: each `s_i=1` contributes one Z gate, so the oracle is `O(n)`. The affine bias is a global phase and is intentionally omitted.

## Example

```python
from pyqpanda_alg import QOracleLearning

secret, probs = QOracleLearning.run_bernstein_vazirani("1011")
assert secret == "1011"

oracle = QOracleLearning.affine_truth_table("101")
kind, probs = QOracleLearning.run_deutsch_jozsa(oracle)
assert kind == "balanced"
```

## Independent correctness reference

`walsh_probabilities()` computes the exact classical Walsh-Hadamard spectrum. It is deliberately independent of PyQPanda3 and is used to test every hidden string through five qubits under both affine biases. The test suite also covers promise rejection, non-affine rejection, bit-order round trips, phase-oracle signs, and CPUQVM end-to-end execution.

The module imports PyQPanda3 lazily: pure truth-table / reference helpers remain usable for validation and documentation tooling without initializing a quantum backend.
