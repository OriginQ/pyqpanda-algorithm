from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from make_plots import main as make_plots_main


def has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    if not has_option("--input"):
        sys.argv.extend(["--input", str(here / "outputs" / "summary.json")])
    if not has_option("--out"):
        sys.argv.extend(["--out", str(here / "outputs" / "figures")])
    make_plots_main()
