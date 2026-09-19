"""Quantum irrigation scheduling for water-scarce Yellow River wheat fields."""

__version__ = "0.2.0"

from .model import IrrigationInstance, build_qubo, load_instance

__all__ = ["IrrigationInstance", "build_qubo", "load_instance", "__version__"]
