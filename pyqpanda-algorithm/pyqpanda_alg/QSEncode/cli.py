"""Minimal command-line interface for QSEncode-Insight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .exceptions import QSEncodeInsightError
from .insight import QSEncodeInsight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyqpanda_alg.QSEncode.cli",
        description="QSEncode-Insight compiled-resource analysis",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    analyze = subcommands.add_parser(
        "analyze", help="analyze one probability distribution"
    )
    analyze.add_argument("--basis", choices=("walsh", "fourier"), required=True)
    analyze.add_argument("--fidelity-target", type=float, default=0.99)
    analyze.add_argument(
        "--verification", choices=("standard", "audit"), default="standard"
    )
    source = analyze.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", help="inline JSON probability list")
    source.add_argument("--input-file", type=Path, help="UTF-8 JSON probability file")
    analyze.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def _probabilities(arguments: argparse.Namespace, parser: argparse.ArgumentParser):
    try:
        text = (
            arguments.input_json
            if arguments.input_json is not None
            else arguments.input_file.read_text(encoding="utf-8")
        )
        values = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.error(f"invalid probability input: {error}")
    if not isinstance(values, list):
        parser.error("invalid probability input: expected a JSON list")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    probabilities = _probabilities(arguments, parser)
    try:
        result = QSEncodeInsight(
            basis=arguments.basis,
            fidelity_target=arguments.fidelity_target,
            verification=arguments.verification,
        ).analyze(probabilities)
    except QSEncodeInsightError as error:
        parser.error(f"analysis failed [{error.code}]: {error}")
    print(result.to_json(indent=2 if arguments.pretty else None))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by smoke command
    raise SystemExit(main())


__all__ = ["main"]
