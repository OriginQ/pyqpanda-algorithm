"""Validate and analyze QCloud probabilities without manufacturing shot counts.

The current cloud response exports probabilities to seven decimal places.  A
mass error of at most ``number_of_entries * 1e-7`` is accepted: this conservative
bound covers either rounding or truncation of each reported entry.  It does
not license incomplete distributions.  Both the original weights and their
explicitly normalized distribution are reported.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import math
import re

import numpy as np

from .hardware import prepare_qaoa_circuit
from .model import IrrigationInstance, build_qubo
from .solvers import enumerate_spectrum, qaoa_probabilities


_HEX_KEY = re.compile(r"(?:0[xX])?[0-9a-fA-F]+\Z")
_PROBABILITY_QUANTUM = Decimal("0.0000001")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key in QCloud taskResult")
        result[key] = value
    return result


def _decode_response(response, job_id, chip, width):
    if not isinstance(response, dict) or response.get("success") is not True:
        raise ValueError("QCloud response is not successful")
    if type(response.get("code")) is not int or response["code"] != 10000:
        raise ValueError("QCloud response code is not 10000")
    obj = response.get("obj")
    if not isinstance(obj, dict) or obj.get("taskId") != job_id:
        raise ValueError("QCloud taskId does not match the expected job")
    if obj.get("chipId") != chip:
        raise ValueError("QCloud chipId does not match the prepared chip")
    state = obj.get("taskState")
    if not ((type(state) is int and state == 3) or (type(state) is str and state == "3")):
        raise ValueError("QCloud taskState is not completed (3)")
    if "errCode" in obj and (type(obj["errCode"]) is not int or obj["errCode"] != 0):
        raise ValueError("QCloud completed response includes an error")
    measured = obj.get("measureQubitSize")
    if (not isinstance(measured, list) or len(measured) != 1
            or type(measured[0]) is not int or measured[0] != width):
        raise ValueError("QCloud measurement width does not match the prepared circuit")
    results = obj.get("taskResult")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], str):
        raise ValueError("Expected exactly one JSON-encoded QCloud circuit result")
    try:
        result = json.loads(results[0], object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON in QCloud taskResult") from exc
    if not isinstance(result, dict):
        raise ValueError("QCloud circuit result must be an object")
    keys, values = result.get("key"), result.get("value")
    if (not isinstance(keys, list) or not isinstance(values, list)
            or not keys or len(keys) != len(values)):
        raise ValueError("QCloud probability keys and values must be nonempty equal-length lists")
    decoded = {}
    for key, value in zip(keys, values):
        if not isinstance(key, str) or _HEX_KEY.fullmatch(key) is None:
            raise ValueError("Invalid hexadecimal probability key")
        index = int(key, 16)
        if index >= 1 << width:
            raise ValueError("QCloud probability key exceeds the measurement width")
        if index in decoded:
            raise ValueError("Duplicate probability state, including hexadecimal aliases")
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("Probabilities must be finite numeric values in [0, 1]")
        decoded[index] = value
    mass_decimal = sum((Decimal(str(value)) for value in values), Decimal(0))
    tolerance = len(values) * _PROBABILITY_QUANTUM
    if mass_decimal <= 0 or abs(mass_decimal - 1) > tolerance:
        raise ValueError("QCloud probability mass differs from 1 beyond the seven-decimal export bound")
    return decoded, mass_decimal, tolerance


def analyze_probability_result(prepared_manifest: dict, cloud_response: dict,
                               expected_job_id: str) -> dict:
    """Analyze one completed result, bound to its saved circuit and task ID.

    A shared, unsubmitted prepared manifest may have ``hardware.job_id=None``.
    If the manifest contains a job ID, it must also match ``expected_job_id``.
    All validation and ideal simulation are offline; no credential or SDK is
    loaded.  Invalid/incomplete/mismatched evidence raises ``ValueError``.
    """
    if not isinstance(expected_job_id, str) or not expected_job_id.strip():
        raise ValueError("expected_job_id must be a nonempty string")
    try:
        if type(prepared_manifest["schema_version"]) is not int or prepared_manifest["schema_version"] != 1:
            raise ValueError("Unsupported prepared manifest schema version")
        report = prepared_manifest["report"]
        hardware = report["hardware"]
        if hardware.get("job_id") not in (None, expected_job_id):
            raise ValueError("Prepared manifest job ID differs from expected_job_id")
        chip = hardware["chip"]
        if not isinstance(chip, str) or not chip:
            raise ValueError("Prepared manifest must identify its chip")
        shots = report["configuration"]["shots"]
        if (type(shots) is not int or shots <= 0
                or type(hardware["shots"]) is not int or hardware["shots"] != shots):
            raise ValueError("Prepared shot count must be positive and consistent")
        max_qubits = report["configuration"]["max_qubits"]
        model = build_qubo(IrrigationInstance.from_dict(report["input_data"]))
        parameters = report["qaoa_reference"]["parameters"]
        rebuilt = prepare_qaoa_circuit(model, parameters["gammas"], parameters["betas"],
                                       max_qubits=max_qubits)
        if rebuilt != prepared_manifest["circuit"] or rebuilt["sha256"] != hardware["circuit_sha256"]:
            raise ValueError("Prepared circuit does not match saved inputs, parameters and checksum")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid prepared hardware manifest structure") from exc

    n, nx = model.n_variables, model.n_decisions
    decoded, mass_decimal, tolerance = _decode_response(cloud_response, expected_job_id, chip, n)
    bits, energies = enumerate_spectrum(model, max_qubits=max_qubits)
    probability = np.zeros(len(bits))
    for index, value in decoded.items():
        probability[index] = value
    normalized = probability / float(mass_decimal)
    ideal = qaoa_probabilities(model, parameters["gammas"], parameters["betas"],
                               backend="numpy", max_qubits=max_qubits)
    events = bits[:, :nx].sum(axis=1).astype(int)
    residuals = events + bits[:, nx:] @ model.slack_weights - model.instance.budget_events
    feasible = events <= model.instance.budget_events
    optimal = np.isclose(energies, energies.min(), rtol=0, atol=1e-7)
    observed = np.flatnonzero(probability > 0)
    feasible_observed = observed[feasible[observed]]

    # Evaluate every feasible physical schedule independently of QUBO penalties.
    physical = {index: model.evaluate(bits[index, :nx]) for index in range(1 << nx)
                if feasible[index]}
    exact_index = min(physical, key=lambda index: (physical[index]["objective"], index))
    exact = deepcopy(physical[exact_index])
    exact["decision_state_index"] = exact_index
    exact["decision_bits_x0_first"] = "".join(str(int(bit)) for bit in bits[exact_index, :nx])
    exact_objective = exact["objective"]
    candidate = None
    if len(feasible_observed):
        decision_mask = (1 << nx) - 1
        best = int(min(feasible_observed,
                       key=lambda index: (physical[int(index) & decision_mask]["objective"], int(index))))
        candidate = deepcopy(physical[best & decision_mask])
        gap = candidate["objective"] - exact_objective
        candidate.update({
            "observed_state_index": best,
            "observed_binary_key": format(best, f"0{n}b"),
            "observed_full_bits_x0_first": "".join(str(int(bit)) for bit in bits[best]),
            "decision_bits_x0_first": "".join(str(int(bit)) for bit in bits[best, :nx]),
            "reported_probability": decoded[best],
            "normalized_export_probability": float(normalized[best]),
            "observed_slack_residual": int(residuals[best]),
            "observed_qubo_energy": float(energies[best]),
            "absolute_gap_to_exact": gap,
            "relative_gap_to_exact": gap / abs(exact_objective) if exact_objective else (0.0 if gap == 0 else None),
            "events_by_zone": {zone.name: [day + 1 for day, bit in enumerate(candidate["schedule"][i]) if bit]
                               for i, zone in enumerate(model.instance.zones)},
        })

    def metrics(distribution):
        return {
            "budget_feasible_probability": float(distribution[feasible].sum()),
            "zero_slack_residual_probability": float(distribution[residuals == 0].sum()),
            "optimal_full_qubo_state_probability": float(distribution[optimal].sum()),
            "mean_qubo_energy": float(distribution @ energies),
        }

    return {
        "schema_version": 1,
        "hardware_execution_status": "completed",
        "result_format": "cloud_reported_decimal_probabilities",
        "raw_integer_counts_available": False,
        "counts_synthesized": False,
        "job_id": expected_job_id,
        "chip": chip,
        "requested_shots": shots,
        "configuration": deepcopy(report["configuration"]),
        "parameters": deepcopy(parameters),
        "circuit_sha256": rebuilt["sha256"],
        "measurement_width": n,
        "unique_reported_states": len(decoded),
        "unique_observed_states": len(observed),
        "unique_budget_feasible_observed_states": len(feasible_observed),
        "reported_probability_sum_decimal": str(mass_decimal),
        "probability_mass_shortfall_decimal": str(Decimal(1) - mass_decimal),
        "probability_mass_tolerance_decimal": str(tolerance),
        "probability_mass_tolerance_rule": "reported_entry_count * 1e-7; permits per-entry seven-decimal rounding or truncation",
        "reported_probabilities_by_state_index": {str(index): decoded[index] for index in sorted(decoded)},
        "raw_export_weighted_metrics": metrics(probability),
        "normalized_export_metrics": metrics(normalized),
        "ideal_metrics": metrics(ideal),
        "total_variation_distance_normalized_export_vs_ideal": float(np.abs(normalized - ideal).sum() / 2),
        "best_observed_budget_feasible_candidate": candidate,
        "classical_optimal_candidate": exact,
        "exact_physical_objective": exact_objective,
        "minimum_qubo_energy": float(energies.min()),
        "optimal_full_qubo_states": int(optimal.sum()),
        "interpretation": [
            "Raw cloud probabilities are preserved; integer counts are unavailable and have not been synthesized.",
            "Normalized metrics divide raw probabilities by their reported sum.",
            "Candidate selection uses only positive-probability observed states; slack bits are not repaired.",
            "An absent observed feasible candidate is null; classical results are never substituted.",
            "TVD includes sampling, hardware and export effects; it is not chip fidelity or evidence of quantum advantage.",
        ],
    }
