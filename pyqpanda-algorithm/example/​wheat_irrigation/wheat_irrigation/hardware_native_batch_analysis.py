"""Offline analysis of an ordered, multi-circuit response from one QCloud job.

Result indices follow the SDK's input/result sequence contract. They are not
independent cloud task IDs, and no circuit identity is inferred from measured
probabilities. The platform task-type enum is retained without interpretation.
"""

from __future__ import annotations

from copy import deepcopy
import re

from .hardware_batch_analysis import analyze_probability_result


def _platform_enum(value):
    return ((type(value) is int and value >= 0)
            or (isinstance(value, str) and re.fullmatch(r"[0-9]{1,3}", value) is not None))


def analyze_native_batch_results(
    plans: list[dict], response: dict, job_id: str, scenario_names: list[str],
) -> dict:
    """Validate one parent job and analyze every result in submission order.

    ``plans`` are the prepared manifests in the exact sequence passed to
    ``backend.run``. A result is associated with a plan by that sequence, not
    by a separate task identifier or a guessed probability distribution.
    Missing task-type metadata remains explicitly absent; it is never filled
    with a presumed native-batch enum. No network or credential access occurs.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a nonempty parent task identifier")
    if not isinstance(plans, list) or not plans:
        raise ValueError("plans must be a nonempty ordered list")
    if (not isinstance(scenario_names, list) or len(scenario_names) != len(plans)
            or any(not isinstance(name, str) or not name.strip() for name in scenario_names)):
        raise ValueError("scenario_names must match the prepared plan sequence")
    if not isinstance(response, dict) or response.get("success") is not True:
        raise ValueError("QCloud response is not successful")
    if type(response.get("code")) is not int or response["code"] != 10000:
        raise ValueError("QCloud response code is not 10000")
    obj = response.get("obj")
    if not isinstance(obj, dict) or obj.get("taskId") != job_id:
        raise ValueError("QCloud taskId does not match the expected parent job")
    state = obj.get("taskState")
    if not ((type(state) is int and state == 3) or (type(state) is str and state == "3")):
        raise ValueError("QCloud taskState is not completed (3)")
    if type(obj.get("errCode")) is not int or obj["errCode"] != 0:
        raise ValueError("QCloud completed batch must explicitly report errCode 0")
    for field in ("taskType", "taskCalMethod"):
        if field in obj and obj[field] is not None and not _platform_enum(obj[field]):
            raise ValueError("QCloud platform enum metadata is invalid")

    widths, chips, requested_shots = [], [], []
    try:
        for plan, name in zip(plans, scenario_names):
            width = plan["circuit"]["n_qubits"]
            hardware = plan["report"]["hardware"]
            configuration = plan["report"]["configuration"]
            if type(width) is not int or width <= 0:
                raise ValueError("Prepared measurement widths must be positive integers")
            if configuration.get("scenario", name) != name:
                raise ValueError("Scenario names do not match the prepared submission sequence")
            if hardware.get("job_id") not in (None, job_id):
                raise ValueError("Prepared manifest job ID differs from the parent job")
            widths.append(width)
            chips.append(hardware["chip"])
            requested_shots.append(configuration["shots"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid prepared native batch manifest structure") from exc
    if any(chip != obj.get("chipId") for chip in chips):
        raise ValueError("QCloud chipId must match every prepared circuit")
    if any(shots != requested_shots[0] for shots in requested_shots):
        raise ValueError("All native batch circuits must have the same requested shots")
    measured = obj.get("measureQubitSize")
    if (not isinstance(measured, list) or len(measured) != len(widths)
            or any(type(width) is not int for width in measured) or measured != widths):
        raise ValueError("QCloud measurement width sequence does not match the prepared circuits")
    results = obj.get("taskResult")
    if (not isinstance(results, list) or len(results) != len(plans)
            or any(not isinstance(result, str) for result in results)):
        raise ValueError("QCloud result count must match the complete prepared circuit sequence")

    analyzed = []
    for index, (plan, name, payload, width) in enumerate(zip(plans, scenario_names, results, widths)):
        # This is an offline projection of the authenticated parent response,
        # not a separately queried or submitted child task. Preserve the ID.
        projected_response = {
            "success": response["success"], "code": response["code"],
            "obj": {"taskId": obj["taskId"], "chipId": obj["chipId"],
                    "taskState": state, "errCode": obj["errCode"],
                    "measureQubitSize": [width], "taskResult": [payload]},
        }
        analyzed.append({
            "result_index": index,
            "scenario_name": name,
            "parent_job_id": job_id,
            "analysis": analyze_probability_result(plan, projected_response, job_id),
        })

    return {
        "schema_version": 1,
        "hardware_execution_status": "completed",
        "parent_job_id": job_id,
        "result_count": len(analyzed),
        "chip": obj["chipId"],
        "requested_shots_per_circuit": requested_shots[0],
        "measurement_widths": list(widths),
        "platform_task_type": deepcopy(obj.get("taskType")),
        "platform_task_type_present": "taskType" in obj,
        "platform_task_cal_method": deepcopy(obj.get("taskCalMethod")),
        "platform_task_type_interpretation": "Uninterpreted raw platform enum; no numeric meaning inferred.",
        "ordering_basis": "SDK input/result sequence contract; no independent per-circuit cloud task IDs.",
        "circuit_identity_verified_by_cloud": False,
        "result_format": "cloud_reported_decimal_probabilities",
        "raw_integer_counts_available": False,
        "counts_synthesized": False,
        "circuit_results": analyzed,
        "interpretation": [
            "Each result_index is a zero-based position in one parent job, not an independent task ID.",
            "Circuit association follows SDK input/result sequence; cloud-side circuit fingerprints are not verified.",
            "Measurement widths must match in full and in order; partial batches and cross-job results are rejected.",
            "Original probabilities are analyzed directly; integer counts are never synthesized.",
        ],
    }
