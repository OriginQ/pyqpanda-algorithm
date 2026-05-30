import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccf2026_mnist_qml import ExperimentConfig, run_experiment


class SmokeTest(unittest.TestCase):
    def test_experiment_writes_submission_outputs(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ccf2026_mnist_"))
        try:
            config = ExperimentConfig(n_samples=32, vqc_epochs=2, shots=64, seed=9)
            result = run_experiment(tmp, config)
            self.assertTrue((tmp / "metrics.csv").exists())
            self.assertTrue((tmp / "resource_table.csv").exists())
            self.assertTrue((tmp / "robustness.csv").exists())
            self.assertTrue((tmp / "report.md").exists())
            self.assertTrue((tmp / "figures" / "test_accuracy.svg").exists())
            vqc_rows = [row for row in result["resources"] if str(row["model"]).startswith("vqc")]
            self.assertTrue(vqc_rows)
            self.assertTrue(all(int(row["qubits"]) == 8 for row in vqc_rows))
            self.assertTrue(all(int(row["total_params"]) <= 100 for row in vqc_rows))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

