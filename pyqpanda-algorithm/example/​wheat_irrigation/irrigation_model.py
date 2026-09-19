"""Compatibility imports; implementation lives in the standalone package."""

from wheat_irrigation.model import (
    IrrigationInstance, IrrigationQUBO, Zone, bounded_slack_weights,
    build_qubo, greedy_schedule, load_instance,
)

__all__ = ["IrrigationInstance", "IrrigationQUBO", "Zone", "bounded_slack_weights",
           "build_qubo", "greedy_schedule", "load_instance"]
