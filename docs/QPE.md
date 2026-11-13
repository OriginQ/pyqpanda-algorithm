# Quantum phase estimation: one engine and two explicit interfaces

The canonical `QPE` preserves the earlier contribution by `1446170411` in
[OriginQ PR #34](https://github.com/OriginQ/pyqpanda-algorithm/pull/34):

```python
from pyqpanda_alg.QPE import QPE

solver = QPE(unitary, n_count=4, n_target=1, state_prep=prepare)
phase = solver.run()                  # float, exact simulator probabilities
phase = solver.run(shots=1000)        # float, actual measured counts
probabilities = solver.run_dict()     # dictionary, exact probabilities
observed = solver.run_dict(shots=1000)
```

The constructor's original positional order, `build_circuit()` return type,
`run()` float result, `run_dict()` dictionary and default `shots=-1` behavior
remain available. Positive shots append native measurement operations and
read `get_counts`; they are not a label attached to exact probabilities.

`StructuredQPE` preserves the structured result and constructor surface from
[tokenjunkielabs PR #81](https://github.com/OriginQ/pyqpanda-algorithm/pull/81)
through a thin adapter over the same engine:

```python
from pyqpanda_alg.QPE import StructuredQPE

solver = StructuredQPE(unitary, system_qubits=1, precision_qubits=5,
                       eigenstate_preparation=prepare, shots=2048)
result = solver.run()                 # actual 2048-shot measured histogram
exact = solver.run_exact()            # explicit exact statevector mode
program, counting = solver.build_program()
```

Alternatively, `from pyqpanda_alg.QPE.compat import QPE` keeps the older local
class name for #81 callers. This import change is necessary: the two original
classes both used the name `QPE`, but the same positional numbers describe
opposite register roles, and `run()` returns different types. A single
transparent constructor cannot infer which API a positional caller intends.

The adapter retains #81's upper counting-register layout; canonical QPE keeps
#34's lower counting register. Both use the same circuit builder and native
execution routine. The adapter reads its current public configuration fields
on each build or run, including changes to `shots` and `machine`.

Correcting #81's sampling behavior is intentional: its original `run()` called
the simulator with a shots argument but always read exact `get_prob_dict`
values. Use `run_exact()` to retain that numerical mode. Structured results
keep the original five fields and add optional `sampling_shots`: `None` for
exact probabilities, a positive number for actual shots. Original five-field
dataclass construction remains valid. The reported `probability` is the
selected bin's probability or observed frequency, not a confidence interval.

## Circuit construction

The `unitary` factory must describe a fixed deterministic U, with its exact
global phase, and contain no measurements. The engine constructs U once per
build and each controlled version once per counting qubit. The unoptimized
logical query count remains `2**n_count - 1`; this change avoids repeated host
factory construction and is not a quantum query-complexity improvement.

Both interfaces accept optional `unitary_power(qubits, power)`, preserved from
#81. It must return a circuit for the exact `U**power`, including global phase;
the engine asks for powers `1, 2, 4, ...` and does not invoke the base factory
in this mode. In canonical QPE this and optional `machine` are new keyword-only
arguments. Return values from all circuit factories are checked as QCircuit.

## Eigenphase readout

`decode_phase(weights, precision_qubits)` validates every key and weight,
including losing bins, and normalizes finite nonnegative probabilities or
counts. It rejects zero total weight. It preserves the original structured
API's insertion-order tie rule. Canonical `run()` preserves #34's exact-mode
lexical tie rule and sampled-mode dictionary order.

Generic eigenphases `phi` and `1-phi` are distinct. They are never folded or
pooled; the mirror-bin rule used by Quantum Counting does not apply here.
For an input state with several eigencomponents, the result is the most likely
phase bin, not a claim that the input is a single eigenstate.

`phase_to_eigenvalue(phase, evolution_time=1, centered=False)` uses
`U=exp(i*H*t)` and chooses either `[0,2*pi/t)` or the centered phase branch.
Eigenvalues remain ambiguous modulo `2*pi/t`; the helper does not determine
an unknown spectral branch. Phase and evolution time must be finite, the time
positive, and the output representable as a finite float.
