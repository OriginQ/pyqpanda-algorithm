"""Persist a hardware experiment and join its actual shots to the comparison.

The manifest contains the resolved input and locally optimized parameters so
that querying a queued job never retrains a forecast or submits another job.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import hardware
from .model import IrrigationInstance, build_qubo
from .solvers import enumerate_spectrum, qaoa_probabilities


MANIFEST_VERSION = 1


def _save(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _record_result(model, report, circuit, result):
    from .cli import summarize_distribution

    report["hardware"].update({key: value for key, value in result.items() if key != "counts"})
    report.pop("qaoa", None)
    report["solutions"].pop("QAOA", None)
    if result["status"] != "finished":
        report["execution"] = "Local ideal QAOA optimization; QPU status: " + result["status"]
        return
    # Recompute only the small ideal reference at the saved angles. There is
    # no optimizer call and no random resampling on the hardware result path.
    reference = report["qaoa_reference"]
    parameters = reference["parameters"]
    probabilities = qaoa_probabilities(model, parameters["gammas"], parameters["betas"], backend="numpy",
                                       max_qubits=report["configuration"]["max_qubits"])
    bits, energies = enumerate_spectrum(model, report["configuration"]["max_qubits"])
    counts = result["counts"]
    if sum(counts.values()) != report["configuration"]["shots"]:
        raise ValueError("Hardware counts do not match the saved requested shots")
    qaoa = summarize_distribution(model, bits, energies, probabilities, counts,
                                  report["exact"]["minimum_qubo_energy"])
    empirical = np.zeros(len(probabilities))
    for index, count in counts.items():
        empirical[index] = count / qaoa["shots"]
    qaoa.update({
        "backend": "qcloud", "engine": "pyqpanda3.qcloud.QCloudService (QPU)",
        "circuit_evaluations": 1, "parameters": parameters,
        "normalization": circuit["normalization"], "ising_offset": reference["ising_offset"],
        "job_id": result["job_id"], "chip": result["chip"],
        "total_variation_distance": float(0.5 * np.abs(empirical - probabilities).sum()),
    })
    report["qaoa"] = qaoa
    if qaoa["best_budget_feasible_sample"] is not None:
        report["solutions"]["QAOA"] = qaoa["best_budget_feasible_sample"]
    report["execution"] = "Local ideal QAOA optimization followed by real QPU measurement; classical exact reference"


def _query(model, manifest, path, args):
    report = manifest["report"]
    started = time.perf_counter()

    def on_submitted(job_id):
        report["hardware"].update(job_id=job_id, status="pending",
                                  submitted_at=datetime.now(timezone.utc).isoformat())
        print(f"QCloud 任务已提交：{job_id}；任务文件：{path}", flush=True)
        _save(path, manifest)

    try:
        result = hardware.run_hardware(
            manifest["circuit"], chip=report["hardware"]["chip"],
            shots=report["configuration"]["shots"], token_env=args.qcloud_token_env,
            timeout=args.qcloud_timeout, poll_interval=args.qcloud_poll_interval,
            job_id=report["hardware"].get("job_id"), on_submitted=on_submitted,
        )
        # Save received counts before deriving agricultural metrics. A local
        # reporting failure must not discard a successful QPU measurement.
        manifest["result"] = result
        _save(path, manifest)
        _record_result(model, report, manifest["circuit"], result)
        report["hardware_query_elapsed_seconds"] = time.perf_counter() - started
        _save(path, manifest)
    except (Exception, KeyboardInterrupt):
        # Keep the job ID from the callback, including when the user stops
        # waiting. Never store SDK exception text, which may include secrets.
        report["hardware"]["query_interrupted"] = True
        _save(path, manifest)
        raise


def start_hardware_test(model, report, args):
    """Export the reproducible plan, optionally submit one final circuit."""
    parameters = report["qaoa_reference"]["parameters"]
    circuit = hardware.prepare_qaoa_circuit(model, parameters["gammas"], parameters["betas"],
                                            max_qubits=args.max_qubits)
    report["hardware"] = {
        "chip": args.qcloud_chip, "shots": args.shots, "job_id": None,
        "status": "dry_run" if args.qcloud_dry_run else "prepared",
        "circuit_sha256": circuit["sha256"], "gate_counts": circuit["gate_counts"],
        "mapping": True, "optimization": True, "amend": False,
        "counts_source": "QCloud measured counts; no probability-to-count conversion",
    }
    manifest = {"schema_version": MANIFEST_VERSION, "circuit": circuit, "report": report}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "hardware_job.json"
    if path.exists():
        raise ValueError("hardware_job.json already exists; resume it or choose a new output directory")
    _save(path, manifest)
    (args.output / "qaoa.originir").write_text(circuit["originir"], encoding="utf-8")
    if args.qcloud_dry_run:
        report["execution"] = "Offline QCloud dry run: local ideal QAOA optimization only; no QPU submission"
        _save(path, manifest)
    else:
        _query(model, manifest, path, args)


def resume_hardware_test(args):
    """Load a saved experiment and query its original job, without resubmission."""
    try:
        manifest = json.loads(args.qcloud_resume.read_text(encoding="utf-8"))
        if manifest["schema_version"] != MANIFEST_VERSION:
            raise ValueError("Unsupported hardware manifest version")
        report = manifest["report"]
        job_id = report["hardware"]["job_id"]
        if not job_id:
            raise ValueError("Manifest has no submitted job; use a new --backend qcloud run and a new output directory")
        model = build_qubo(IrrigationInstance.from_dict(report["input_data"]))
        parameters = report["qaoa_reference"]["parameters"]
        rebuilt = hardware.prepare_qaoa_circuit(model, parameters["gammas"], parameters["betas"],
                                                max_qubits=report["configuration"]["max_qubits"])
        if rebuilt != manifest["circuit"] or rebuilt["sha256"] != report["hardware"]["circuit_sha256"]:
            raise ValueError("Saved circuit does not match the saved model and QAOA parameters")
        if report["hardware"]["shots"] != report["configuration"]["shots"]:
            raise ValueError("Saved hardware shots do not match experiment configuration")
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid hardware_job.json structure") from exc
    destination = args.output / "hardware_job.json"
    if destination.resolve() != args.qcloud_resume.resolve() and destination.exists():
        raise ValueError("Resume output already contains another hardware_job.json; choose a new output directory")
    # Keep the original manifest updated even if reports go to a new directory.
    _query(model, manifest, args.qcloud_resume, args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qaoa.originir").write_text(rebuilt["originir"], encoding="utf-8")
    if destination.resolve() != args.qcloud_resume.resolve():
        _save(destination, deepcopy(manifest))
    return model, report
