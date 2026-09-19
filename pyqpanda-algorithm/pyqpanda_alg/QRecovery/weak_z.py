"""A narrow PyQPanda weak-Z measurement/reversal block, with heralded output.

Append to an existing program whose data may be unknown/entangled. The two
auxiliaries must be fresh |0>. Both outcome bits remain explicit. This block
never reinitializes data and does not select a simulator or read probabilities.
"""
from math import acos, isfinite, sqrt
from pyqpanda3.core import CNOT, QProg, RY, measure


def _diagonal(program,data,auxiliary,d0,d1):
    t0,t1=2*acos(d0),2*acos(d1)
    program << RY(auxiliary,(t0+t1)/2)
    program << CNOT(data,auxiliary)
    program << RY(auxiliary,(t0-t1)/2)
    program << CNOT(data,auxiliary)


def weak_z_recovery_block(data,first_auxiliary,recovery_auxiliary,*,
                          first_cbit,recovery_cbit,strength=.8):
    """Return a measured QProg block; recovery_cbit==0 heralds restoration.

    Output data remains usable. No label is a promise under hardware noise.
    The first measured auxiliary is kept and used as a computational-basis
    control, equivalent to its recorded classical feedback in the ideal model.
    It is NOT reset before its two controlled-NOT uses. The returned block has
    no preparation instruction, backend execution, or implicit postselection.
    Both auxiliaries must begin in |0>; cbits must be distinct and caller-owned.
    Only 0.5 < strength < 1 is supported. Full probability and failure behavior
    are described in docs/weak_z_recovery.md; do not repeat on failure as if data were fresh.
    """
    qubits=(data,first_auxiliary,recovery_auxiliary)
    if any(type(q) is not int or q<0 for q in qubits) or len(set(qubits))!=3:
        raise ValueError('three distinct nonnegative qubit indices required')
    if any(type(c) is not int or c<0 for c in (first_cbit,recovery_cbit)) or first_cbit==recovery_cbit:
        raise ValueError('two distinct nonnegative classical bits required')
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not isfinite(strength) or not .5<strength<1:
        raise ValueError('strength must be finite and strictly between 0.5 and 1')
    program=QProg(max(qubits)+1)
    _diagonal(program,data,first_auxiliary,sqrt(strength),sqrt(1-strength))
    program << measure(first_auxiliary,first_cbit)
    # The measured flag controls which component of the actual poststate is
    # attenuated; both CXs restore the original data basis before returning.
    program << CNOT(first_auxiliary,data)
    _diagonal(program,data,recovery_auxiliary,sqrt((1-strength)/strength),1.)
    program << CNOT(first_auxiliary,data)
    program << measure(recovery_auxiliary,recovery_cbit)
    return program
