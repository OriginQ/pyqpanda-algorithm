#!/usr/bin/env python
"""Coverage-baseline regression checker for the pyqpanda_alg packages.

Compares a fresh pytest-cov JSON report (``--cov=pyqpanda_alg
--cov-branch --cov-report=json:...``) with the committed per-package
baseline in ``test/coverage-baseline.json`` and exits nonzero when any
existing algorithm package's line or branch coverage decreases.

Usage::

    python tools/check_coverage_baseline.py test/coverage-baseline.json test/coverage-current.json
    python tools/check_coverage_baseline.py --generate test/coverage-current.json > test/coverage-baseline.json

New packages without a committed baseline (for example the planned VQE,
HHL, and Shor modules) are reported as informational only, except the
``execution`` module which is held to its explicit plan threshold
(``--cov-fail-under=85`` in the execution plan).  Add explicit
thresholds below when the corresponding plan lands.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_PREFIX = "pyqpanda-algorithm"
SOURCE_PACKAGE = "pyqpanda_alg"
EXCLUDED_FILENAMES = {"__init__.py", "_version.py"}

#: Packages covered by explicit plan thresholds instead of a baseline.
#: The execution plan pins 85% line coverage; the VQE/HHL/Shor plans have
#: not specified numeric thresholds yet, so those modules stay
#: informational until they do.
EXPLICIT_THRESHOLDS = {
    "execution": {"line": 85.0},
}

#: Aggregated percentages are rounded to two decimals; absorb float noise
#: from re-aggregation of identical data.
FLOAT_TOLERANCE = 0.05


def _aggregate(coverage_data):
    """Aggregate per-unit line/branch percentages from a pytest-cov JSON.

    Keys look like ``pyqpanda-algorithm\\pyqpanda_alg\\Grover\\Grover_core.py``
    on Windows; a unit is the top-level algorithm package (``Grover``,
    ``QAOA``, ...) or a top-level module (``plugin``).  Package
    ``__init__``/``_version`` files are excluded so the mapping reflects
    the algorithm code itself.
    """
    totals = {}
    for key, file_info in coverage_data.get("files", {}).items():
        parts = key.replace("\\", "/").split("/")
        if (
            len(parts) < 3
            or parts[0] != ROOT_PREFIX
            or parts[1] != SOURCE_PACKAGE
        ):
            continue
        if Path(parts[-1]).name in EXCLUDED_FILENAMES:
            continue
        unit = Path(parts[-1]).stem if len(parts) == 3 else parts[2]
        summary = file_info["summary"]
        total = totals.setdefault(
            unit,
            {
                "num_statements": 0,
                "covered_lines": 0,
                "num_branches": 0,
                "covered_branches": 0,
            },
        )
        total["num_statements"] += summary["num_statements"]
        total["covered_lines"] += summary["covered_lines"]
        total["num_branches"] += summary["num_branches"]
        total["covered_branches"] += summary["covered_branches"]

    units = {}
    for unit, total in totals.items():
        line = (
            total["covered_lines"] / total["num_statements"] * 100.0
            if total["num_statements"]
            else 100.0
        )
        branch = (
            total["covered_branches"] / total["num_branches"] * 100.0
            if total["num_branches"]
            else None
        )
        units[unit] = {
            "line": round(line, 2),
            "branch": round(branch, 2) if branch is not None else None,
        }
    return units


def _generate(current_path):
    """Write the committed baseline mapping for the given coverage JSON."""
    with open(current_path, "r", encoding="utf-8") as handle:
        coverage_data = json.load(handle)
    units = _aggregate(coverage_data)
    packages = {
        unit: {"line": info["line"], "branch": info["branch"]}
        for unit, info in sorted(units.items())
        if unit not in EXPLICIT_THRESHOLDS
    }
    baseline = {
        "packages": packages,
        "explicit_thresholds": EXPLICIT_THRESHOLDS,
        "note": (
            "Measured by `python -m pytest test --cov=pyqpanda_alg "
            "--cov-branch --cov-report=json:test/coverage-current.json` "
            "during Plan 3 Task 6; check_coverage_baseline.py exits "
            "nonzero when an existing package drops below these numbers. "
            "New modules without a baseline entry (VQE/HHL/Shor plans) "
            "use explicit plan thresholds instead."
        ),
    }
    json.dump(baseline, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _check(baseline_path, current_path):
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)["packages"]
    with open(current_path, "r", encoding="utf-8") as handle:
        current = _aggregate(json.load(handle))

    failures = []
    for unit, expected in sorted(baseline.items()):
        if unit not in current:
            failures.append(f"{unit}: package missing from current coverage")
            continue
        measured = current[unit]
        if measured["line"] + FLOAT_TOLERANCE < expected["line"]:
            failures.append(
                f"{unit}: line coverage dropped "
                f"{expected['line']:.2f}% -> {measured['line']:.2f}%"
            )
        if (
            expected["branch"] is not None
            and measured["branch"] is not None
            and measured["branch"] + FLOAT_TOLERANCE < expected["branch"]
        ):
            failures.append(
                f"{unit}: branch coverage dropped "
                f"{expected['branch']:.2f}% -> {measured['branch']:.2f}%"
            )

    for unit, thresholds in EXPLICIT_THRESHOLDS.items():
        if unit in current and current[unit]["line"] < thresholds["line"]:
            failures.append(
                f"{unit}: line coverage {current[unit]['line']:.2f}% below "
                f"explicit plan threshold {thresholds['line']:.2f}%"
            )

    for unit in sorted(set(current) - set(baseline) - set(EXPLICIT_THRESHOLDS)):
        info = current[unit]
        branch = f"{info['branch']:.2f}%" if info["branch"] is not None else "n/a"
        print(f"info: {unit}: new module, line {info['line']:.2f}%, branch {branch}",
              file=sys.stderr)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print("coverage baseline regression detected", file=sys.stderr)
        return 1
    print("PASS: no coverage regression against the committed baseline")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="emit a baseline mapping for the given coverage JSON",
    )
    parser.add_argument("baseline", nargs="?", help="committed baseline JSON")
    parser.add_argument("current", nargs="?", help="fresh pytest-cov JSON")
    args = parser.parse_args()

    if args.generate:
        _generate(args.baseline)
        return 0
    if not args.baseline or not args.current:
        parser.error("baseline and current JSON paths are required")
    return _check(args.baseline, args.current)


if __name__ == "__main__":
    sys.exit(main())
