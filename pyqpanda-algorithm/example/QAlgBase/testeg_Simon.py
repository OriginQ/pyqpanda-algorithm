"""Small Simon hidden-XOR example using the pure reference path."""

from pyqpanda_alg import Simon


if __name__ == "__main__":
    secret = "10110"
    probabilities = Simon.simon_reference_probabilities(secret)
    recovered = Simon.recover_secret_from_probabilities(probabilities, len(secret))

    print("secret:", secret)
    print("recovered:", recovered)
    print("non-zero support:")
    for index, probability in enumerate(probabilities):
        if probability > 0:
            bits = "".join(str(bit) for bit in Simon.index_to_q0_bits(index, len(secret)))
            print(f"  {bits}: {probability:.6f}")
