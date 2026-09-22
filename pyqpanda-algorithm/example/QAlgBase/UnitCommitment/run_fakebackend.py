"""Run the case5 QAOA circuit through the official Runtime FakeBackend.

This is the Runtime-side pre-QPU rehearsal.  It optimizes the QAOA parameters
with the repository's local state-vector path, compiles the final measured
``QProg`` with the Runtime FakeBackend transpiler, and samples that compiled
program locally with device-derived noise.  No real-QPU task is created.  A
Bell smoke is optional and is not part of the default case5 validation path.

The API key is read from the environment variable named by ``--api-key-env``.
Only the variable name and a boolean presence flag are written to the result
package; the key value is never printed or persisted.

Example (PowerShell)::

    $env:QPANDA3_API_KEY = "<your-OriginQ-api-key>"
    python scripts\\run_fakebackend.py `
      --hardware-name WK_C180 --shots 256 --maxiter 4

The optional ``qpanda3_runtime`` package is intentionally not imported by the
ordinary state-vector/QCloud code paths.  Run this script with the environment
that contains QPanda3 Runtime 1.0.1 (the local validation environment used by
this project is ``probability_quantum/.venv``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqpanda_alg.QAOA.qaoa import QAOA  # noqa: E402
from pyqpanda_alg.UnitCommitment import (  # noqa: E402
    Case5CommitmentQuboBuilder,
    Case5Candidate,
    Case5UCInstance,
    RuntimeFakeBackendRunner,
    decode_commitment,
    commitment_capacity,
    evaluate_schedule,
    load_case5_uc,
    solve_milp_uc,
)
from pyqpanda_alg.UnitCommitment.plugin import measure_all  # noqa: E402


DEFAULT_BLOCKS = {
    "WK_C180": [85, 94, 95, 96, 103, 104, 105, 113, 114, 123],
    "WK_C180_2": [39, 40, 48, 49, 50, 57, 58, 59, 67, 68],
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware-name",
        action="append",
        default=None,
        help="Runtime QDevice id; repeat for WK_C180 and WK_C180_2",
    )
    parser.add_argument("--api-key-env", default="QPANDA3_API_KEY")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--bell-shots", type=int, default=128)
    parser.add_argument(
        "--include-bell-smoke",
        action="store_true",
        help="Also run the optional local Bell diagnostic; omitted by default for the case5 mainline.",
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--capacity-weight", type=float, default=200.0)
    parser.add_argument(
        "--initial-parameters",
        type=float,
        nargs="+",
        default=None,
        help="Optional gamma..., beta... warm start with 2*layers values",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Use --initial-parameters directly; no local optimizer call",
    )
    parser.add_argument(
        "--specified-block",
        type=int,
        nargs="+",
        default=None,
        help="Explicit physical qubit block; defaults to a recorded connected block",
    )
    parser.add_argument(
        "--compensation-mode",
        choices=("neutral", "device"),
        default="neutral",
        help="Runtime coherent-angle compensation policy (neutral is the auditable default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "runtime_fakebackend",
    )
    return parser


def _build_program(qaoa: QAOA, parameters: Sequence[float], layers: int, n_qubits: int):
    from pyqpanda3.core import QProg

    values = np.asarray(parameters, dtype=float).reshape(-1)
    if values.size != 2 * int(layers):
        raise ValueError("the QAOA parameter vector must contain 2*layers values")
    program = QProg(int(n_qubits))
    qubits = program.qubits()
    qaoa.layer = int(layers)
    program << qaoa._qaoa_circuit(
        qubits,
        values[:layers],
        values[layers:],
    )
    program << measure_all(qubits, qubits)
    return program


def _candidate_records(
    instance: Case5UCInstance,
    model: Any,
    probabilities: dict[str, float],
    *,
    top_k: int,
    evaluate_method: str = "linprog",
) -> tuple[list[Case5Candidate], float, Case5Candidate | None]:
    base: list[Case5Candidate] = []
    for bitstring, probability in probabilities.items():
        if len(str(bitstring)) != model.n_variables or any(
            char not in "01" for char in str(bitstring)
        ):
            raise ValueError(f"invalid Runtime bitstring: {bitstring!r}")
        commitments = decode_commitment(instance, model.registry, str(bitstring))
        capacities = tuple(
            commitment_capacity(instance, commitments, period)
            for period in instance.periods
        )
        capacity_feasible = all(
            capacity + 1.0e-7 >= instance.demand_mw[period] + instance.reserve_mw[period]
            for period, capacity in enumerate(capacities)
        )
        base.append(
            Case5Candidate(
                bitstring=str(bitstring),
                probability=float(probability),
                commitments=commitments,
                committed_capacity_mw=capacities,
                capacity_feasible=capacity_feasible,
                qubo_energy=float(model.energy(model.registry.bits_from_bitstring(str(bitstring)))),
            )
        )
    base.sort(key=lambda item: (-item.probability, item.qubo_energy, item.bitstring))
    feasible = [item for item in base if item.capacity_feasible]
    keys = {item.bitstring for item in feasible[: int(top_k)]}
    if feasible:
        keys.add(min(feasible, key=lambda item: (item.qubo_energy, item.bitstring)).bitstring)
    evaluated: list[Case5Candidate] = []
    for item in base:
        if item.bitstring in keys:
            schedule = evaluate_schedule(
                instance,
                item.commitments,
                method=evaluate_method,
                enforce_network=True,
            )
            item = Case5Candidate(
                bitstring=item.bitstring,
                probability=item.probability,
                commitments=item.commitments,
                committed_capacity_mw=item.committed_capacity_mw,
                capacity_feasible=item.capacity_feasible,
                qubo_energy=item.qubo_energy,
                schedule=schedule,
            )
        evaluated.append(item)
    evaluated.sort(
        key=lambda item: (
            not item.physically_feasible,
            float("inf") if item.schedule is None else item.schedule.total_cost,
            -item.probability,
            item.bitstring,
        )
    )
    best = next((item for item in evaluated if item.physically_feasible), None)
    return evaluated[: int(top_k)], float(sum(item.probability for item in feasible)), best


def _optimise_parameters(
    model: Any,
    *,
    layers: int,
    maxiter: int,
    seed: int,
    initial_parameters: Sequence[float] | None,
    skip_optimization: bool,
) -> tuple[QAOA, np.ndarray, float, int, str]:
    np.random.seed(int(seed))
    qaoa = QAOA(model.expression)
    if skip_optimization:
        if initial_parameters is None:
            raise ValueError("--skip-optimization requires --initial-parameters")
        parameters = np.asarray(initial_parameters, dtype=float).reshape(-1)
        if parameters.size != 2 * int(layers):
            raise ValueError("--initial-parameters must contain 2*layers values")
        qaoa.layer = int(layers)
        ideal_probabilities = qaoa.run_qaoa_circuit(
            parameters[:layers], parameters[layers:], shots=-1
        )
        loss = float(qaoa._loss_function_default(ideal_probabilities))
        return qaoa, parameters, loss, 1, "provided warm start"
    _, parameters, loss = qaoa.run(
        layer=int(layers),
        initial_para=(
            None
            if initial_parameters is None
            else np.asarray(initial_parameters, dtype=float)
        ),
        shots=-1,
        optimizer="SLSQP",
        optimizer_option={"options": {"maxiter": int(maxiter), "disp": False}},
    )
    return qaoa, np.asarray(parameters, dtype=float), float(loss), int(qaoa.circuit_iter), "local state-vector SLSQP"


def _run_one(
    *,
    instance: Case5UCInstance,
    model: Any,
    qaoa: QAOA,
    parameters: np.ndarray,
    optimizer_loss: float,
    circuit_evaluations: int,
    parameter_source: str,
    hardware_name: str,
    args: argparse.Namespace,
    reference_cost: float,
) -> dict[str, Any]:
    block = args.specified_block
    if block is None:
        block = DEFAULT_BLOCKS.get(hardware_name)
    runner = RuntimeFakeBackendRunner(
        real_device_id=hardware_name,
        api_key_env=args.api_key_env,
        specified_block=block,
        compensation_mode=args.compensation_mode,
    )
    bell = (
        runner.bell_smoke(shots=args.bell_shots)
        if args.include_bell_smoke
        else {
            "status": "not_run_by_configuration",
            "reason": "Bell diagnostics are not part of the default case5 mainline.",
            "real_qpu_submitted": False,
        }
    )
    program = _build_program(qaoa, parameters, args.layers, model.n_variables)
    sample = runner.sample(program, shots=args.shots)
    candidates, feasible_probability, best = _candidate_records(
        instance,
        model,
        sample.probabilities,
        top_k=args.top_k,
    )
    best_cost = None if best is None or best.schedule is None else float(best.schedule.total_cost)
    gap = None if best_cost is None else 100.0 * (best_cost - reference_cost) / reference_cost
    return {
        "hardware_name": hardware_name,
        "status": "finished",
        "runtime": sample.metadata,
        "bell_smoke": bell,
        "qaoa": {
            "layers": int(args.layers),
            "logical_qubits": int(model.n_variables),
            "shots": int(args.shots),
            "parameters": [float(value) for value in parameters],
            "parameter_source": parameter_source,
            "optimizer": "SLSQP" if parameter_source.startswith("local") else "none",
            "optimizer_loss": float(optimizer_loss),
            "circuit_evaluations": int(circuit_evaluations),
            "probability_sum": float(sum(sample.probabilities.values())),
            "feasible_probability": float(feasible_probability),
            "best_cost": best_cost,
            "best_candidate": None if best is None else best.as_dict(),
            "candidates": [item.as_dict() for item in candidates],
        },
        "comparison": {
            "reference_total_cost": float(reference_cost),
            "runtime_fakebackend_best_cost": best_cost,
            "runtime_fakebackend_gap_percent": gap,
            "dispatch_solver": "scipy.optimize.linprog (HiGHS)",
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if min(args.shots, args.layers, args.top_k) < 1:
        raise SystemExit("shots, layers, and top-k must be positive")
    if args.include_bell_smoke and args.bell_shots < 1:
        raise SystemExit("bell-shots must be positive when --include-bell-smoke is used")
    if args.maxiter < 1 and not args.skip_optimization:
        raise SystemExit("maxiter must be positive unless --skip-optimization is used")
    hardware_names = list(dict.fromkeys(args.hardware_name or ["WK_C180"]))
    instance = load_case5_uc()
    model = Case5CommitmentQuboBuilder(capacity_weight=args.capacity_weight).build(instance)
    reference = solve_milp_uc(instance, evaluate_method="linprog", enforce_network=True)
    if not reference.success or not reference.schedule.success:
        raise RuntimeError(f"classical MILP reference failed: {reference.message}")
    qaoa, parameters, loss, circuit_evaluations, parameter_source = _optimise_parameters(
        model,
        layers=args.layers,
        maxiter=args.maxiter,
        seed=args.seed,
        initial_parameters=args.initial_parameters,
        skip_optimization=args.skip_optimization,
    )
    runs: list[dict[str, Any]] = []
    for hardware_name in hardware_names:
        try:
            runs.append(
                _run_one(
                    instance=instance,
                    model=model,
                    qaoa=qaoa,
                    parameters=parameters,
                    optimizer_loss=loss,
                    circuit_evaluations=circuit_evaluations,
                    parameter_source=parameter_source,
                    hardware_name=hardware_name,
                    args=args,
                    reference_cost=float(reference.schedule.total_cost),
                )
            )
        except Exception as exc:  # keep per-device diagnostics without leaking credentials
            runs.append(
                {
                    "hardware_name": hardware_name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "runtime_fakebackend_audit_v1",
        "scope": {
            "real_qpu_submitted": False,
            "remote_quantum_task_submitted": False,
            "execution_environment": "OriginQ qpanda3_runtime local FakeBackend",
            "execution_path": "direct FakeBackend transpile/sample",
        },
        "credential_policy": {
            "api_key_environment_variable": args.api_key_env,
            "api_key_present": bool(os.environ.get(args.api_key_env)),
            "api_key_value_saved": False,
        },
        "case5": {
            "source_url": instance.case.source_url,
            "logical_qubits": model.n_variables,
            "layers": args.layers,
            "capacity_weight": args.capacity_weight,
        },
        "reference": _clean(reference.as_dict()),
        "optimization": {
            "parameter_source": parameter_source,
            "parameters": [float(value) for value in parameters],
            "loss": float(loss),
            "circuit_evaluations": int(circuit_evaluations),
        },
        "runs": _clean(runs),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "case5_runtime_fakebackend_audit.json"
    md_path = args.output / "case5_runtime_fakebackend_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Case 5 Runtime FakeBackend rehearsal",
        "",
        "This report uses the official `qpanda3_runtime` `RuntimeService → QDevice → FakeBackend` path.",
        "It is a local calibration-derived finite-shot rehearsal; no remote quantum task and no real-QPU job was submitted.",
        "",
        f"- API-key variable: `{args.api_key_env}`; present=true; value saved=false.",
        f"- Reference MILP/LP cost: `{reference.schedule.total_cost:.8f}`.",
        f"- Parameter source: `{parameter_source}`.",
        "",
        "## Device runs",
        "",
        "| Device | Status | Bell diagnostic | Best cost | Gap to MILP/LP | Feasible probability | Physical block |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for run in runs:
        if run.get("status") != "finished":
            lines.append(f"| `{run.get('hardware_name')}` | **failed** | — | — | — | — | `{run.get('error_type')}: {run.get('error')}` |")
            continue
        bell = run["bell_smoke"]
        qaoa = run["qaoa"]
        comparison = run["comparison"]
        runtime = run["runtime"]
        bell_status = (
            f"even parity {bell['bell_even_parity_probability']:.6f}; passed={bell['passed']}"
            if bell.get("status") != "not_run_by_configuration"
            else "not run (not part of case5 mainline)"
        )
        lines.append(
            f"| `{run['hardware_name']}` | **finished** | {bell_status} | "
            f"{qaoa['best_cost'] if qaoa['best_cost'] is not None else '—'} | "
            f"{comparison['runtime_fakebackend_gap_percent'] if comparison['runtime_fakebackend_gap_percent'] is not None else '—'}% | "
            f"{qaoa['feasible_probability']:.6f} | `{runtime['specified_block']}` |"
        )
        lines.extend(
            [
                "",
                f"### `{run['hardware_name']}` details",
                "",
                f"- Runtime package/version: `{runtime['runtime_package']}` / `{runtime['runtime_version']}`; pyqpanda3 `{runtime.get('pyqpanda3_version')}`.",
                f"- Compiled program SHA-256: `{runtime['transpiled_program_sha256']}`.",
                f"- Compiled gate counts: `{runtime['compiled_gate_counts']}`.",
                f"- Compensation policy: `{runtime['compensation_mode']}`; used entries `{runtime['compensation_entries_used']}`.",
                f"- Bell diagnostic: `{bell.get('status', 'finished')}`; QAOA probability sum: `{qaoa['probability_sum']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The result validates the Runtime FakeBackend transport, native compilation, physical-block mapping, finite-shot result parsing, and the classical economic-dispatch post-processing. It is not a real-device accuracy claim. A real-QPU run remains separately confirmation-gated.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    completed = [run for run in runs if run.get("status") == "finished" and run["qaoa"].get("best_cost") is not None]
    if completed:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [str(run["hardware_name"]) for run in completed]
            values = [float(run["qaoa"]["best_cost"]) for run in completed]
            figure, axis = plt.subplots(figsize=(6.8, 3.8))
            axis.bar(labels, values, color="#557a95", width=0.55, label="Runtime FakeBackend")
            axis.axhline(float(reference.schedule.total_cost), color="#b23a48", linestyle="--", linewidth=1.6, label="MILP/LP reference")
            axis.set_ylabel("Total cost")
            axis.set_title("Case5 Runtime FakeBackend cost check")
            axis.grid(axis="y", alpha=0.25)
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(args.output / "case5_runtime_fakebackend_costs.png", dpi=180)
            plt.close(figure)
        except Exception:
            # The numerical JSON/Markdown record is the required artifact.
            pass
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "run_status": [run["status"] for run in runs]}, ensure_ascii=False))
    return 0 if all(run.get("status") == "finished" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

