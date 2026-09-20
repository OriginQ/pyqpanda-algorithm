# QEC: three-qubit repetition codes

`pyqpanda_alg.QEC` adds the canonical three-qubit repetition code in two
complementary forms:

- **bit-flip code**: corrects one Pauli-X error;
- **phase-flip code**: applies the same repetition logic in the Hadamard basis
  and corrects one Pauli-Z error.

These are intentionally small, inspectable quantum error-correction building
blocks. They demonstrate encoding, syndrome formation and coherent recovery
without claiming a general fault-tolerant stack.

## Qubit contract

The public circuit functions receive exactly three distinct qubits:

```text
q_code = [q0, q1, q2]
```

Before encoding, `q0` contains the arbitrary logical state
`alpha|0> + beta|1>` and `q1=q2=|0>`.

After recovery, the decoded logical state is again on `q0`. The two former
ancillas retain a deterministic syndrome in `(q1, q2)`:

| error | syndrome `q1 q2` | syndrome integer `q1 + 2*q2` |
| --- | --- | ---: |
| none | `00` | 0 |
| on `q0` | `11` | 3 |
| on `q1` | `10` | 1 |
| on `q2` | `01` | 2 |

No measurement or reset is assumed by the recovery circuit; the syndrome stays
coherently encoded in the ancillas.

## Bit-flip code

Encoding uses two CNOTs:

```text
alpha|000> + beta|100>  --encode-->  alpha|000> + beta|111>
```

where the ket ordering in the equation is conceptual `(q0,q1,q2)` ordering.
The circuit API itself uses the supplied qubit objects directly and does not
rely on a state-vector bitstring display convention.

Recovery applies:

```text
CNOT(q0, q1)
CNOT(q0, q2)
TOFFOLI(q1, q2, q0)
```

For any one X error, this returns the original logical state to `q0` and leaves
the corresponding syndrome in `q1,q2`.

```python
from pyqpanda_alg.QEC import bit_flip_encode, bit_flip_recover

q = [0, 1, 2]
encode = bit_flip_encode(q)
recover = bit_flip_recover(q)
```

## Phase-flip code

The phase-flip encoder first creates the ordinary repetition code and then
applies `H` to all three code qubits. A Z error in that basis is mapped to an X
error by another layer of Hadamards during recovery, after which the bit-flip
recovery circuit is reused.

```python
from pyqpanda_alg.QEC import phase_flip_encode, phase_flip_recover

q = [0, 1, 2]
encode = phase_flip_encode(q)
recover = phase_flip_recover(q)
```

## Independent NumPy reference

`reference_repetition_recovery()` performs exact three-qubit state-vector
evolution without importing PyQPanda3. It is intended as an independent small
problem oracle for tests and examples.

```python
import numpy as np
from pyqpanda_alg.QEC import reference_repetition_recovery

logical = np.array([1, 1j], dtype=complex) / np.sqrt(2)
result = reference_repetition_recovery(
    "phase_flip",
    logical_state=logical,
    error_qubit=2,
)

print(result["logical_fidelity"])        # ~1.0
print(result["syndrome_probabilities"]) # [0, 0, 1, 0]
```

The reference exposes:

- the reduced density matrix of decoded `q0`;
- pure-state logical fidelity;
- syndrome probabilities;
- optional final three-qubit state vector;
- explicit `X`/`Z` override so callers can demonstrate the error-model
  boundary rather than accidentally imply broader correction.

## Capability boundary

A three-qubit repetition code has distance three only for the selected error
basis:

- the bit-flip code corrects one **X** error but does not generally correct Z;
- the phase-flip code corrects one **Z** error but does not generally correct X.

This module does **not** claim to correct an arbitrary single-qubit Pauli error.
A construction such as the nine-qubit Shor code combines both mechanisms for a
broader error model; that construction is outside this module's scope.

## Gate cost

Ignoring error injection itself:

| operation | CNOT | TOFFOLI | H |
| --- | ---: | ---: | ---: |
| bit-flip encode | 2 | 0 | 0 |
| bit-flip recover | 2 | 1 | 0 |
| phase-flip encode | 2 | 0 | 3 |
| phase-flip recover | 2 | 1 | 3 |

Backend-native cost for the Toffoli depends on backend decomposition and
transpilation.
