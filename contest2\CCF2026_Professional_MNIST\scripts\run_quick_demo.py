import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccf2026_mnist_qml import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CCF2026 professional MNIST QML demo.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "quick_demo")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    config = ExperimentConfig(
        n_samples=args.samples,
        vqc_epochs=args.epochs,
        shots=args.shots,
        seed=args.seed,
    )
    result = run_experiment(args.output_dir, config)
    print(f"Output: {args.output_dir}")
    print(f"Models: {len(result['resources'])}")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
