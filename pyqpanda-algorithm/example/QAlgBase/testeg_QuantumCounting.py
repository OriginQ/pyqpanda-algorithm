"""Estimate the number of marked basis states with Quantum Counting.

This example marks two states in an eight-state search space. Increase
phase_bits for finer count resolution, at the cost of more controlled Grover
applications.
"""

from pyqpanda_alg.QuantumCounting import QuantumCounting


def main():
    counter = QuantumCounting(
        qnumber=3,
        phase_bits=6,
        mark_data=["101", "111"],
        shots=2048,
    )
    result = counter.run()

    print("search space:", result.search_space_size)
    print("estimated marked states:", result.estimated_count)
    print("nearest integer count:", result.rounded_count)
    print("dominant phase state:", result.phase_state)
    print("dominant phase probability:", result.phase_probability)


if __name__ == "__main__":
    main()
