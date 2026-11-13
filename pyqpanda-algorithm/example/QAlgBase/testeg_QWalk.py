"""Small discrete-time coined quantum-walk example."""

from pyqpanda_alg.QWalk import CoinedQuantumWalkCycle


walk = CoinedQuantumWalkCycle(
    num_position_qubits=3,
    steps=4,
    initial_position=0,
    initial_coin=0,
)

print("Independent exact reference distribution:")
for position, probability in enumerate(walk.reference()):
    if probability > 1e-12:
        print(f"  position {position}: {probability:.6f}")

# Requires PyQPanda3, like the other circuit examples in this repository.
print("PyQPanda3 exact circuit distribution:")
for position, probability in enumerate(walk.run_exact()):
    if probability > 1e-12:
        print(f"  position {position}: {probability:.6f}")
