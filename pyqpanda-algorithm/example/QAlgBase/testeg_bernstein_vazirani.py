"""Minimal Bernstein-Vazirani example."""

from pyqpanda_alg.BernsteinVazirani import BernsteinVazirani


def main():
    hidden = "101101"
    algorithm = BernsteinVazirani(hidden)
    result = algorithm.run(shots=1)

    print("hidden string:   ", hidden)
    print("measured string: ", result.measured_secret)
    print("oracle queries:  ", algorithm.quantum_oracle_queries)
    print("success prob.:   ", result.success_probability)


if __name__ == "__main__":
    main()
