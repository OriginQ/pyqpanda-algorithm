from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyqpanda_demo import main as pyqpanda_demo_main


if __name__ == "__main__":
    default_out = str(Path(__file__).resolve().parent / "outputs" / "pyqpanda_counts.json")
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", default_out])
    pyqpanda_demo_main()
