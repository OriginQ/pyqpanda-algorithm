"""Run with python -m pyqpanda_alg.LabScheduling INPUT.json."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .report import run_experiment


def _report_destination(source: Path, output: Path) -> Path:
    """Resolve links and reject input aliases before starting or saving a run."""
    destination = output.resolve()
    if destination == source.resolve() or (
        destination.exists() and source.exists() and destination.samefile(source)
    ):
        raise ValueError("output must not overwrite the input file (including links)")
    if destination.exists() and not destination.is_file():
        raise ValueError("output must be a regular report file")
    return destination


def _save_report(source: Path, output: Path, text: str) -> None:
    """Replace a report only after a complete same-directory temporary write."""
    destination = _report_destination(source, output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text + "\n")
        # Recheck links in case an output alias changed while writing.
        _report_destination(source, destination)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Print JSON; exit 2 for I/O/input errors, 3/4/5 for infeasible/miss/limit."""
    parser = argparse.ArgumentParser(description="共享实验室 QUBO 最小扰动重排（CPU）")
    parser.add_argument("input", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--mixer", choices=("xy", "x"), default="xy")
    parser.add_argument(
        "--reduce", action="store_true", help="exact propagation and component solving"
    )
    parser.add_argument("--output", type=Path, help="also save the JSON report here")
    parser.add_argument(
        "--pruning",
        choices=("singleton", "arc"),
        default="singleton",
        help="propagation strength; arc requires --reduce",
    )
    args = parser.parse_args(argv)
    try:
        if args.output:
            _report_destination(args.input, args.output)
        report = run_experiment(
            args.input,
            args.seed,
            args.layers,
            args.shots,
            args.maxiter,
            args.restarts,
            args.mixer,
            reduce=args.reduce,
            pruning=args.pruning,
        )
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            _save_report(args.input, args.output, text)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(text)
    return {
        "feasible": 0,
        "infeasible": 3,
        "no_feasible_sample": 4,
        "resource_limit": 5,
    }[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
