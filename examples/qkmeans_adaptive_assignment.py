"""Explicit software measurement backend; the algorithm sees only counts."""
from pyqpanda3.core import CPUQVM
from pyqpanda_alg.QKmeans import adaptive_assign


def main():
    machine = CPUQVM()

    def run_counts(program, shots):
        machine.run(program, shots)
        return machine.result().get_counts()

    result = adaptive_assign(
        (0.0, -1.0),
        [(0.0, -1.0), (0.0, 1.0)],
        run_counts,
        batch_shots=64,
        max_shots=1024,
        alpha=0.05,
    )
    print(result)
    if result.winner is None:
        print("Budget exhausted without separation; no winner is asserted.")
    else:
        print("Selected candidate index:", result.winner)


if __name__ == "__main__":
    main()
