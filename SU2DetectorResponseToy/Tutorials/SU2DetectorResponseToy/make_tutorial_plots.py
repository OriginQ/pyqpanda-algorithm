from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from make_plots import main as make_plots_main


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(here / "outputs" / "summary.json")])
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(here / "outputs" / "figures")])
    make_plots_main()
