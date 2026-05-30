from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "mnist_like_3_vs_8"
    n_samples: int = 120
    n_pixels: int = 64
    n_qubits: int = 8
    ansatz_layers: int = 2
    seed: int = 2020
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    logistic_epochs: int = 180
    vqc_epochs: int = 12
    shots: int = 512
    eval_noise_prob: float = 0.04
    train_noise_prob: float = 0.04
    angle_jitter: float = 0.04
