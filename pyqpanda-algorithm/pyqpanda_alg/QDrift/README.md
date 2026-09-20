# QDrift

QDrift implements randomized Hamiltonian simulation for Hamiltonians written as
real-coefficient Pauli sums. The algorithm samples terms with probability
proportional to coefficient magnitude and replaces a long deterministic product
formula with a stochastic sequence of fixed-angle Pauli evolutions.

The module is backend independent. sample_qdrift returns QDriftRotation records
with a Pauli string and signed exponent. A PyQPanda backend can translate each
record into its preferred Pauli-rotation circuit, while remote executors can
batch or stream the same schedule.

The segment helper implements the qDRIFT first-order diamond-norm bound

2 * lambda^2 * time^2 / N

where lambda is the L1 norm of the Pauli coefficients.

For reference-sized problems the module also provides dense Hamiltonian
construction, exact time evolution, direct execution of a sampled schedule,
pure-state fidelity, Hermitian observable expectation values, and independent
trajectory averaging with standard-error reporting.
