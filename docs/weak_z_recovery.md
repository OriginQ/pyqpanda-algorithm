# Heralded reversal of a weak Z measurement

`weak_z_recovery_block` adds a fixed weak measurement and its probabilistic
reversal to an existing program. It never prepares the data, selects a backend
or hides failed shots. The data may be unknown and entangled with a reference.
This is an implementation of standard measurement reversal, not a new quantum
error-correction code.

## Usage

```python
from pyqpanda_alg.QRecovery import weak_z_recovery_block

program << weak_z_recovery_block(
    data=0, first_auxiliary=2, recovery_auxiliary=3,
    first_cbit=0, recovery_cbit=1, strength=0.8,
)
# Append further computation on the same data register.
# recovery_cbit == 0 heralds success; retain all failed shots as well.
```

The three qubit indices must differ. Both auxiliary qubits must initially be
|0>; the two classical bits must be distinct. The block uses RY, CNOT and Z
measurements. After the first auxiliary is measured, its retained basis state
controls the recovery; **do not reset it between that measurement and its
controlled gates**. The caller owns register allocation and the initial state.

Only finite `0.5 < strength < 1` is supported. A backend must support mid-circuit
measurement followed by gates. Terminal-only sampling services may not support
this program; no noisy-hardware equivalence or calibration guarantee is implied.

## Principle

For `s = strength`, the first instrument is

```text
K0 = diag(sqrt(s), sqrt(1-s))
K1 = diag(sqrt(1-s), sqrt(s)).
```

Set `r = sqrt((1-s)/s)`. The successful filters selected by the measured bit are
`A0 = diag(r,1)` and `A1 = diag(1,r)`. Consequently

```text
A0 K0 = A1 K1 = sqrt(1-s) I.
```

Each successful history has probability `1-s` from the start of the program;
total success probability is `2(1-s)` for every input. Tensoring the identity
with any untouched reference proves ideal correlation preservation. The
probability conditioned on a particular first result generally depends on the
input and must not be confused with this total probability.

Failure histories have cumulative operators `diag(sqrt(2s-1),0)` and
`diag(0,sqrt(2s-1))`. They are not fresh copies of the input. Do not blindly
repeat recovery on failure or interpret a failure as a deterministic reversal
of a projective measurement. This primitive is relevant when the initial weak
measurement is required; doing nothing to undisturbed data is a different task.

## Tests and sample observations

```sh
PYTHONPATH=pyqpanda-algorithm python -m pytest -q -o addopts='' test/QAlgBase/test_weak_z_recovery.py
PYTHONPATH=pyqpanda-algorithm python examples/weak_z_recovery.py
```

The tests check all four branch maps, including failure, at three strengths,
plus register and parameter validation. CPUQVM statevectors are used only as
independent test references. The example explicitly selects CPUQVM, prepares a
Bell pair, appends the block, then applies H and S to its data. An inverse/joint
readout checks that the heralded output remains usable and correlated.

At `s=0.8`, the exact success probability is 0.4. In a 512-shot run on an external
classical-control quantum simulator, 186 shots heralded success and all 186
passed this specified joint readout. All failed shots were retained. Full counts
and the readout convention are in
[`weak_z_counts_observed.json`](../benchmarks/weak_z_counts_observed.json).
These are finite software-simulation observations, not a QPU experiment or a
fidelity theorem; ideal correctness follows from the branch identities above.

See [Korotkov and Jordan, PRL 97, 166805](https://doi.org/10.1103/PhysRevLett.97.166805)
and [Jordan and Korotkov's review](https://arxiv.org/abs/0906.3468).
The module deliberately supports only this instrument, not arbitrary Kraus
operators or general recovery-policy synthesis.
