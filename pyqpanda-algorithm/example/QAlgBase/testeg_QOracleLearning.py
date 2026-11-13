"""Deutsch-Jozsa and Bernstein-Vazirani example."""

from pyqpanda_alg import QOracleLearning


if __name__ == "__main__":
    secret = "1011"  # q0-first
    recovered, probabilities = QOracleLearning.run_bernstein_vazirani(secret)
    print("Bernstein-Vazirani secret:", recovered)
    print("peak probability:", max(probabilities))

    balanced_oracle = QOracleLearning.affine_truth_table("101")
    classification, probabilities = QOracleLearning.run_deutsch_jozsa(balanced_oracle)
    print("Deutsch-Jozsa:", classification)
    print("P(0...0):", probabilities[0])
