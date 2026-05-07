from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyqpanda_demo import main as pyqpanda_demo_main


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    default_out = str(Path(__file__).resolve().parent / "outputs" / "pyqpanda_counts.json")
    if not has_option("--out"):
        sys.argv.extend(["--out", default_out])
    pyqpanda_demo_main()
