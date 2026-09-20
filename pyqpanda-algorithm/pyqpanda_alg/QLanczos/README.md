# QLanczos

QLanczos implements a quantum Krylov / Lanczos spectral solver for noisy or
exact matrix elements. It solves the generalized Hermitian eigenproblem
H c = E S c for a non-orthogonal sequence of quantum states.

The execution layer is callback based, so the same algorithm can consume
PyQPanda circuits, hardware estimators, remote batched jobs, or exact simulator
state vectors.

## Numerical strategy

Krylov states often become nearly linearly dependent as the order grows.
Directly applying inv(S) H is fragile. QLanczos instead uses rank-revealing
symmetric orthogonalization, rejects non-physical overlap matrices, supports a
relative overlap cutoff and optional diagonal regularization, and reports the
raw/effective rank, discarded directions, condition number and residual norms.

Only unique upper-triangle matrix elements are requested. The helper
krylov_measurement_pairs(order) exposes those pairs for remote batching.

## Backend contract

A backend supplies two callbacks:

overlap(left_state, right_state) -> complex

hamiltonian(left_state, right_state) -> complex

State objects are opaque to QLanczos. They can be PyQPanda circuit factories,
prepared programs, remote state identifiers, or custom objects understood by
those callbacks.

The QuantumLanczos class can also construct a repeated Krylov sequence through
an evolve(previous_state, step) callback. Exact-state helpers are included for
reference and simulator use: statevector_overlap, dense_hamiltonian_element,
and imaginary_time_evolver.
